# TDD：手机 APP 对接 OmniVLA 实时轨迹服务（v1）

## 1. 文档信息
- **状态**：Draft（基于 PRD 评审结果）
- **版本**：v1.0
- **关联文档**：`docs/PRD_mobile_app_omnivla_realtime_v1_zh.md`

## 2. 本版拍板结论（已冻结）
- **手机端**：Android Kotlin（先安卓 MVP）
- **协议**：仅 WebSocket（v1）
- **并发目标**：单会话优先
- **返回结构**：`waypoints + linear_vel + angular_vel`
- **部署**：内网单机 GPU 服务

## 3. 技术目标
- 模型加载一次后常驻，不随帧重复初始化
- 单会话 3~5 FPS 连续输入稳定运行
- 端到端 `p95 <= 450ms`
- 对手机端暴露稳定 JSON 协议，便于后续扩展

## 4. 系统架构

### 4.1 逻辑分层
1. **Android App**
   - 相机采集（CameraX）
   - 抽帧与压缩（JPEG）
   - WebSocket 客户端
   - 轨迹渲染（2D 叠加）与速度显示
2. **OmniVLA Inference Server**
   - FastAPI + WebSocket 路由
   - SessionManager（单会话状态）
   - FrameBuffer（`maxsize=1` 最新帧覆盖）
   - OmniVLAEngine（模型常驻、GPU 前向）
   - Postprocessor（轨迹/速度结果封装）
3. **Observability**
   - 日志：会话、帧序号、错误码
   - 指标：`queue_wait/infer/post/total`

### 4.2 部署拓扑（v1）
- 1 台内网 GPU 主机（RTX 4080 级别）
- 1 个推理进程（建议 1 worker，避免模型重复加载）
- Android 端通过内网 Wi-Fi 直连 WebSocket

## 5. 服务端模块设计

### 5.1 目录结构（建议）
```text
OmniVla/
  server/
    app.py
    ws_router.py
    schemas.py
    session_manager.py
    frame_buffer.py
    omnivla_engine.py
    postprocess.py
    metrics.py
    settings.py
```

### 5.2 模块职责
- `omnivla_engine.py`
  - 进程启动即初始化（`define_model()`）
  - 对外 `infer(current_image, goal_image, instruction, session_ctx)`
  - 返回 `waypoints/linear_vel/angular_vel/modality_id`
- `frame_buffer.py`
  - `maxsize=1`
  - 新帧到达时覆盖旧帧（丢旧保新）
- `session_manager.py`
  - 单会话元数据：`session_id/goal_image/instruction/last_seq`
  - 生命周期：创建、更新、销毁
- `ws_router.py`
  - 消息分发：`start_session/frame/stop_session`
  - 异常统一封装 `error`
- `postprocess.py`
  - 将模型输出标准化为协议字段

### 5.3 与现有代码衔接
- 从 `inference/run_omnivla.py` 抽出可复用 `OmniVLAEngine`
- 保留既有前向路径和后处理逻辑（waypoint 选取、速度计算）
- CLI 脚本继续保留用于离线调试，不影响服务化模块

## 6. WebSocket 协议设计（v1）

### 6.1 连接
- URL：`ws://<server-host>:<port>/ws/v1/omnivla`
- 编码：JSON 文本（图像字段为 base64，后续可升级二进制帧）

### 6.2 上行消息（APP -> Server）

#### 6.2.1 `start_session`
```json
{
  "type": "start_session",
  "session_id": "sess_001",
  "instruction": "沿走廊前进，避开右侧障碍物",
  "goal_image_jpeg_b64": "<base64>",
  "config": {
    "fps": 4
  }
}
```

#### 6.2.2 `frame`
```json
{
  "type": "frame",
  "session_id": "sess_001",
  "seq": 1001,
  "timestamp_ms": 1710000000123,
  "current_image_jpeg_b64": "<base64>"
}
```

#### 6.2.3 `stop_session`
```json
{
  "type": "stop_session",
  "session_id": "sess_001"
}
```

### 6.3 下行消息（Server -> APP）

#### 6.3.1 `trajectory`
```json
{
  "type": "trajectory",
  "session_id": "sess_001",
  "seq": 1001,
  "linear_vel": 0.23,
  "angular_vel": -0.05,
  "modality_id": 6,
  "waypoints": [
    [0.10, -0.12, 0.92, 0.15],
    [0.18, -0.18, 0.90, 0.19]
  ],
  "timing_ms": {
    "queue_wait": 5,
    "infer": 162,
    "post": 7,
    "total": 206
  }
}
```

#### 6.3.2 `error`
```json
{
  "type": "error",
  "session_id": "sess_001",
  "seq": 1001,
  "error_code": "E_IMAGE_DECODE",
  "message": "invalid jpeg payload"
}
```

## 7. 时序设计

### 7.1 会话建立
1. APP 建立 WS 连接
2. APP 发送 `start_session`
3. 服务端解码 `goal_image`，初始化会话上下文
4. 服务端返回 `ack`（可选）或首个状态消息

### 7.2 帧推理循环
1. APP 定时发送 `frame(seq)`
2. 服务端写入 `FrameBuffer(maxsize=1)`，必要时覆盖旧帧
3. 推理 worker 读取最新帧
4. `OmniVLAEngine.infer()` 执行前向
5. 服务端回发 `trajectory(seq)`

### 7.3 会话销毁
1. APP 发送 `stop_session` 或 WS 断开
2. 服务端清理会话缓存（保留模型常驻）

## 8. 关键实现细节

### 8.1 模型常驻策略
- 进程启动时加载模型，加载完成后设置 `ready=true`
- WS 请求未就绪时返回 `E_MODEL_NOT_READY`
- 禁止每消息动态创建模型对象

### 8.2 帧队列与背压
- 使用 `maxsize=1`，新帧覆盖旧帧
- 不做无界排队，避免延迟累积
- APP 端收到高延迟可主动降 FPS

### 8.3 线程/协程模型
- WebSocket I/O：async 协程
- 推理执行：单独 worker（线程或异步任务串行）
- 单会话场景下采用串行推理，逻辑最简单稳定

### 8.4 图像与内存
- JPEG 解码后立即转 RGB
- 尽量复用预处理对象，减少反复创建
- 严格控制临时张量生命周期，避免显存碎片

## 9. 配置项（v1）
- `MODEL_PATH`：默认 `./omnivla-original`
- `DEVICE`：默认 `cuda:0`
- `WS_HOST` / `WS_PORT`
- `MAX_FPS`：默认 5
- `FRAME_BUFFER_SIZE`：固定 1
- `LOG_LEVEL`

## 10. 错误码（首版）
- `E_BAD_REQUEST`：字段缺失/类型错误
- `E_SESSION_NOT_FOUND`：会话不存在
- `E_IMAGE_DECODE`：图像解码失败
- `E_MODEL_NOT_READY`：模型未加载完成
- `E_INFER_TIMEOUT`：推理超时
- `E_INTERNAL`：未知内部错误

## 11. 测试方案

### 11.1 单元测试
- schema 校验
- session 生命周期
- frame 覆盖策略（丢旧保新）

### 11.2 集成测试
- `start_session -> frame -> trajectory -> stop_session` 闭环
- 错误帧输入回包校验
- 连续 5 分钟稳定性测试

### 11.3 性能测试
- 输入 3/4/5 FPS 对比
- 统计 `p50/p95` 总时延
- 输出 GPU 显存曲线（确认无持续增长）

## 12. 开发计划（按模块）
- **阶段 A**：抽象 `OmniVLAEngine`（从脚本迁移到服务层）
- **阶段 B**：实现 WS 协议与 Session/FrameBuffer
- **阶段 C**：接入 Android Kotlin 客户端 MVP
- **阶段 D**：压测与指标看板

## 13. 验收门槛（与 PRD 对齐）
1. 启动后模型仅加载一次
2. 单会话连续 5 分钟无崩溃
3. APP 持续收到 `waypoints + linear/angular`
4. `p95 total_ms <= 450ms`
5. 错误码与错误消息可被 APP 正确处理

## 14. 后续演进（v2+）
- WebRTC 视频通道
- 多会话并发与资源隔离
- 二进制协议（减少 base64 体积）
- 与机器人底盘控制闭环联调
