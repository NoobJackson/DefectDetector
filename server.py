import asyncio
import uvicorn
import os
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


# 摄像头管理类：负责摄像头初始化、帧捕获和资源释放
class CameraManager:
    def __init__(self):
        self.cap = None
        self.width = 1280
        self.height = 720
        self.last_capture_time = 0
        self.min_capture_interval = 0.05  # 最小捕获间隔(20fps)，避免摄像头过载

    def initialize(self):
        """初始化摄像头，带重试机制"""
        if self.cap and self.cap.isOpened():
            return True
            
        # 尝试多个摄像头索引(0-2)，解决索引不正确问题
        for idx in range(3):
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                # 设置并验证分辨率
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # 获取实际分辨率
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"成功打开摄像头 {idx}，分辨率: {self.width}x{self.height}")
                return True
                
            self.release()  # 释放未成功打开的摄像头
        
        raise Exception("无法打开任何摄像头，请检查设备连接")

    def capture_frame(self):
        """捕获一帧图像，含间隔控制和错误处理"""
        # 检查摄像头状态，必要时重新初始化
        if not self.cap or not self.cap.isOpened():
            if not self.initialize():
                return None
        
        # 控制捕获频率，避免过载
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
                print("摄像头读取失败，尝试重新初始化")
                self.release()
                self.initialize()
                return None
        except Exception as e:
            print(f"捕获帧时出错: {str(e)}")
            self.release()
            return None

    def release(self):
        """释放摄像头资源"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None


# 图像处理类：使用YOLO模型进行目标检测并绘制结果
class ImageProcessor:
    def __init__(self, model_path="yolo11n.pt"):
        # 加载模型，失败时使用默认模型
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"模型加载失败，使用默认模型: {str(e)}")
            self.model = YOLO("yolo11n.pt")  # 自动下载默认模型
        
        self.confidence = 0.4  # 置信度阈值
        self.processing_time = 0  # 处理耗时(毫秒)，用于调试

    def process(self, frame):
        """处理帧：目标检测并绘制边界框"""
        if frame is None:
            return None
            
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
            print(f"图像处理错误: {str(e)}")
            return frame  # 出错时返回原始帧


# WebSocket连接管理类：处理会话创建、连接注册和帧发送
class ConnectionManager:
    def __init__(self):
        self.connections = {}  # {session_id: WebSocket}
        self.active_sessions = set()

    def create_session(self):
        """创建新会话并返回会话ID"""
        session_id = str(uuid.uuid4())
        self.active_sessions.add(session_id)
        print(f"创建新会话: {session_id}")
        return session_id

    def is_valid_session(self, session_id):
        """检查会话ID是否有效"""
        return session_id in self.active_sessions

    def register(self, session_id, websocket):
        """注册新连接"""
        self.connections[session_id] = websocket

    def unregister(self, session_id):
        """移除连接并清理会话"""
        if session_id in self.connections:
            del self.connections[session_id]
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)
            print(f"会话结束: {session_id}")

    async def send_frame(self, websocket, frame):
        """将帧编码为JPEG并发送"""
        # 检查连接状态
        if frame is None or websocket.client_state != WebSocketState.CONNECTED:
            return False
        
        try:
            # JPEG编码参数（质量70，平衡传输效率和图像质量）
            encode_param = [
                int(cv2.IMWRITE_JPEG_QUALITY), 70,
                int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
            ]
            
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            if ret:
                await websocket.send_bytes(buffer.tobytes())
                return True
            else:
                print("帧编码失败")
                return False
                
        except Exception as e:
            print(f"发送帧错误: {str(e)}")
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
            print("摄像头初始化成功")
        else:
            print("警告: 摄像头初始化失败，将在首次请求时重试")
    except Exception as e:
        print(f"启动时出错: {str(e)}")
    
    yield  # 应用运行期间
    
    # 关闭时清理
    camera_manager.release()
    print("应用已关闭，资源已释放")


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
    frame_interval = 0.05  # 目标帧率20fps
    consecutive_errors = 0  # 连续错误计数器
    
    try:
        while True:
            # 检查会话和连接有效性
            if (not connection_manager.is_valid_session(session_id) or 
                connection_manager.connections.get(session_id) != websocket or
                websocket.client_state != WebSocketState.CONNECTED):
                break
            
            # 捕获帧
            frame = camera_manager.capture_frame()
            if frame is None:
                consecutive_errors += 1
                if consecutive_errors % 5 == 0:
                    await websocket.send_text(json.dumps({
                        "type": "warning",
                        "message": f"连续{consecutive_errors}次获取帧失败"
                    }))
                await asyncio.sleep(frame_interval)
                continue
            
            consecutive_errors = 0  # 重置错误计数器
            
            # 处理帧
            processed_frame = image_processor.process(frame)
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
        print(f"客户端主动断开连接: {session_id}")
    except asyncio.CancelledError:
        print(f"帧生产任务已取消: {session_id}")
    except Exception as e:
        print(f"帧处理循环错误: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"服务器内部错误: {str(e)}"
            }))
        except:
            pass
    finally:
        connection_manager.unregister(session_id)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket端点：处理客户端连接和帧传输"""
    try:
        await websocket.accept()
    except Exception as e:
        print(f"无法接受WebSocket连接: {str(e)}")
        return
    
    # 验证会话ID
    if not connection_manager.is_valid_session(session_id):
        try:
            await websocket.close(code=1008, reason="无效的会话ID")
        except:
            pass
        return
        
    # 注册连接
    connection_manager.register(session_id, websocket)
    print(f"客户端连接成功: {session_id}")
    
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
                print(f"收到客户端消息: {message}")
                
                # 响应客户端
                await websocket.send_text(json.dumps({
                    "type": "ack",
                    "message": "消息已接收",
                    "received": message
                }))
            except asyncio.TimeoutError:
                continue  # 超时正常，继续发送帧
            except Exception as e:
                print(f"接收消息错误: {str(e)}")
                break
            
    except WebSocketDisconnect:
        print(f"客户端断开连接: {session_id}")
    except Exception as e:
        print(f"WebSocket处理错误: {str(e)}")
    finally:
        # 清理任务
        if 'producer_task' in locals():
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                print(f"任务已成功取消: {session_id}")
        connection_manager.unregister(session_id)


@app.get("/", response_class=HTMLResponse)
async def get_client(request: Request):
    """返回客户端页面，创建新会话"""
    session_id = connection_manager.create_session()
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "session_id": session_id
    })


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )