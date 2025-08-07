import asyncio
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import uuid
from starlette.websockets import WebSocketDisconnect, WebSocketState
import cv2
from ultralytics import YOLO
import time
from pydantic_settings import BaseSettings


# 配置管理
class Settings(BaseSettings):
    camera_width: int = 1280
    camera_height: int = 720
    min_capture_interval: float = 0.05  # 最小捕获间隔(20fps)
    yolo_model_path: str = "yolo11n.pt"
    yolo_confidence: float = 0.4
    jpeg_quality: int = 70
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    max_consecutive_errors: int = 10  # 最大连续错误数
    ssl_keyfile: str = ""  # SSL私钥路径(生产环境配置)
    ssl_certfile: str = ""  # SSL证书路径(生产环境配置)

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

# 日志配置
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("yolo_detector")


# 摄像头管理类：负责摄像头初始化、帧捕获和资源释放
class CameraManager:
    def __init__(self):
        self.cap = None
        self.width = settings.camera_width
        self.height = settings.camera_height
        self.last_capture_time = 0
        self.min_capture_interval = settings.min_capture_interval
        self.lock = asyncio.Lock()  # 摄像头访问锁

    def initialize(self):
        """初始化摄像头，带重试机制"""
        if self.cap and self.cap.isOpened():
            return True
            
        # 尝试多个摄像头索引(0-2)
        for idx in range(3):
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                # 设置并验证分辨率
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # 获取实际分辨率
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"成功打开摄像头 {idx}，分辨率: {self.width}x{self.height}")
                return True
                
            self.release()  # 释放未成功打开的摄像头
        
        raise Exception("无法打开任何摄像头，请检查设备连接")

    async def capture_frame(self):
        """异步捕获一帧图像，含间隔控制和错误处理"""
        async with self.lock:  # 确保摄像头串行访问
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._sync_capture_frame)

    def _sync_capture_frame(self):
        """同步捕获帧实现(在线程池执行)"""
        # 检查摄像头状态，必要时重新初始化
        if not self.cap or not self.cap.isOpened():
            try:
                self.initialize()
            except Exception as e:
                logger.error(f"摄像头初始化失败: {str(e)}")
                return None
        
        # 控制捕获频率
        current_time = time.time()
        if current_time - self.last_capture_time < self.min_capture_interval:
            return None
        
        # 捕获帧并处理异常
        try:
            ret, frame = self.cap.read()
            if ret:
                self.last_capture_time = current_time
                return frame
            else:
                logger.warning("摄像头读取失败，尝试重新初始化")
                self.release()
                self.initialize()
                return None
        except Exception as e:
            logger.error(f"捕获帧时出错: {str(e)}")
            self.release()
            return None

    def release(self):
        """释放摄像头资源"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        logger.info("摄像头资源已释放")


# 图像处理类：使用YOLO模型进行目标检测并绘制结果
class ImageProcessor:
    def __init__(self):
        self.model_path = settings.yolo_model_path
        self.model = None  # 延迟初始化
        self.confidence = settings.yolo_confidence
        self.processing_time = 0  # 处理耗时(毫秒)

    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            try:
                self.model = YOLO(self.model_path)
                logger.info(f"成功加载模型: {self.model_path}")
            except Exception as e:
                logger.warning(f"模型加载失败，使用默认模型: {str(e)}")
                self.model = YOLO("yolo11n.pt")  # 自动下载默认模型

    async def process(self, frame):
        """异步处理帧：目标检测并绘制边界框"""
        if frame is None:
            return None
            
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_process, frame)

    def _sync_process(self, frame):
        """同步处理帧实现(在线程池执行)"""
        self._load_model()  # 首次调用时加载模型
        
        start_time = time.time()
        try:
            # 目标检测
            results = self.model(
                frame, 
                conf=self.confidence, 
                verbose=False,
            )
            
            # 绘制检测结果
            annotated_frame = results[0].plot()
            self.processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
            return annotated_frame
            
        except Exception as e:
            logger.error(f"图像处理错误: {str(e)}")
            return frame  # 出错时返回原始帧

    def set_confidence(self, value):
        """动态调整置信度阈值"""
        if 0 < value < 1:
            self.confidence = value
            logger.info(f"置信度已调整为: {value}")
            return True
        return False


# WebSocket连接管理类：处理会话创建、连接注册和帧发送
class ConnectionManager:
    def __init__(self):
        self.connections = {}  # {session_id: WebSocket}
        self.active_sessions = set()
        self.lock = asyncio.Lock()  # 并发安全锁

    async def create_session(self):
        """创建新会话并返回会话ID"""
        async with self.lock:
            session_id = str(uuid.uuid4())
            self.active_sessions.add(session_id)
            logger.info(f"创建新会话: {session_id}")
            return session_id

    async def is_valid_session(self, session_id):
        """检查会话ID是否有效"""
        async with self.lock:
            return session_id in self.active_sessions

    async def register(self, session_id, websocket):
        """注册新连接"""
        async with self.lock:
            self.connections[session_id] = websocket

    async def unregister(self, session_id):
        """移除连接并清理会话"""
        async with self.lock:
            if session_id in self.connections:
                del self.connections[session_id]
            if session_id in self.active_sessions:
                self.active_sessions.remove(session_id)
                logger.info(f"会话结束: {session_id}")

    async def send_frame(self, websocket, frame):
        """将帧编码为JPEG并发送"""
        # 检查连接状态
        if frame is None or websocket.client_state != WebSocketState.CONNECTED:
            return False
        
        try:
            # JPEG编码参数
            encode_param = [
                int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality,
                int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
            ]
            
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            if ret:
                await websocket.send_bytes(buffer.tobytes())
                return True
            else:
                logger.warning("帧编码失败")
                return False
                
        except Exception as e:
            logger.error(f"发送帧错误: {str(e)}")
            return False


# 核心组件实例化
camera_manager = CameraManager()
image_processor = ImageProcessor()
connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：初始化和清理资源"""
    # 启动时初始化
    try:
        camera_ready = camera_manager.initialize()
        if camera_ready:
            logger.info("摄像头初始化成功")
        else:
            logger.warning("摄像头初始化失败，将在首次请求时重试")
    except Exception as e:
        logger.error(f"启动时出错: {str(e)}")
    
    yield  # 应用运行期间
    
    # 关闭时清理
    camera_manager.release()
    logger.info("应用已关闭，资源已释放")


# 初始化FastAPI应用
app = FastAPI(
    title="YOLO实时目标检测与WebSocket推流服务",
    lifespan=lifespan
)

# 配置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


async def frame_producer(websocket: WebSocket, session_id: str):
    """视频帧生产任务：持续捕获、处理并发送帧"""
    frame_interval = settings.min_capture_interval
    consecutive_errors = 0  # 连续错误计数器
    
    try:
        while True:
            # 检查会话和连接有效性
            if (not await connection_manager.is_valid_session(session_id) or 
                connection_manager.connections.get(session_id) != websocket or
                websocket.client_state != WebSocketState.CONNECTED):
                break
            
            # 捕获帧
            frame = await camera_manager.capture_frame()
            if frame is None:
                consecutive_errors += 1
                if consecutive_errors % 5 == 0:
                    await websocket.send_text(json.dumps({
                        "type": "warning",
                        "message": f"连续{consecutive_errors}次获取帧失败"
                    }))
                # 超过最大连续错误数则退出
                if consecutive_errors >= settings.max_consecutive_errors:
                    logger.error(f"超过最大连续错误数，关闭会话: {session_id}")
                    break
                await asyncio.sleep(frame_interval)
                continue
            
            consecutive_errors = 0  # 重置错误计数器
            
            # 处理帧
            processed_frame = await image_processor.process(frame)
            if processed_frame is None:
                await asyncio.sleep(frame_interval)
                continue
            
            # 发送帧
            send_success = await connection_manager.send_frame(websocket, processed_frame)
            if not send_success:
                consecutive_errors += 1
            
            # 动态调整间隔，匹配处理能力
            adjusted_interval = max(frame_interval, image_processor.processing_time / 1000)
            await asyncio.sleep(adjusted_interval)
            
    except WebSocketDisconnect:
        logger.info(f"客户端主动断开连接: {session_id}")
    except asyncio.CancelledError:
        logger.info(f"帧生产任务已取消: {session_id}")
    except Exception as e:
        logger.error(f"帧处理循环错误: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"服务器内部错误: {str(e)}"
            }))
        except:
            pass
    finally:
        await connection_manager.unregister(session_id)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket端点：处理客户端连接和帧传输"""
    try:
        await websocket.accept()
    except Exception as e:
        logger.error(f"无法接受WebSocket连接: {str(e)}")
        return
    
    # 验证会话ID
    if not await connection_manager.is_valid_session(session_id):
        try:
            await websocket.close(code=1007, reason="无效的会话ID")  # 使用更合适的关闭代码
        except:
            pass
        return
        
    # 注册连接
    await connection_manager.register(session_id, websocket)
    logger.info(f"客户端连接成功: {session_id}")
    
    try:
        # 发送连接成功消息
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "已成功连接到视频流服务器",
            "resolution": f"{camera_manager.width}x{camera_manager.height}"
        }))
        
        # 启动帧生产者任务
        producer_task = asyncio.create_task(frame_producer(websocket, session_id))
        
        # 保持连接，监听客户端消息
        while True:
            try:
                # 非阻塞接收客户端消息
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                message = json.loads(data)
                logger.info(f"收到客户端消息: {message}")
                
                # 处理客户端命令
                response = {"type": "ack", "message": "消息已接收", "received": message}
                if message.get("type") == "set_confidence":
                    success = image_processor.set_confidence(message.get("value", 0.4))
                    response = {
                        "type": "confidence_updated" if success else "error",
                        "message": f"置信度已更新为{image_processor.confidence}" if success else "无效的置信度值"
                    }
                
                await websocket.send_text(json.dumps(response))
            except asyncio.TimeoutError:
                continue  # 超时正常，继续发送帧
            except Exception as e:
                logger.error(f"接收消息错误: {str(e)}")
                break
            
    except WebSocketDisconnect:
        logger.info(f"客户端断开连接: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {str(e)}")
    finally:
        # 清理任务
        if 'producer_task' in locals():
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                logger.info(f"任务已成功取消: {session_id}")
        await connection_manager.unregister(session_id)


@app.get("/", response_class=HTMLResponse)
async def get_client(request: Request):
    """返回客户端页面，创建新会话"""
    session_id = await connection_manager.create_session()
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "session_id": session_id
    })


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    uvicorn.run(
        "server:app", 
        host=settings.server_host, 
        port=settings.server_port, 
        reload=False,
        log_level="info",
        ssl_keyfile=settings.ssl_keyfile if settings.ssl_keyfile else None,
        ssl_certfile=settings.ssl_certfile if settings.ssl_certfile else None,
    )