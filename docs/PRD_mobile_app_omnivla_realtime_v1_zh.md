# PRD：手机 APP 对接 OmniVLA 实时轨迹服务（v1 草案）

## 1. 文档信息
- **状态**：Draft（待评审）
- **版本**：v1.0
- **目标读者**：产品、算法、后端、移动端、测试、运维
- **文档目的**：先确认需求、框架与边界，再进入开发实现

## 2. 背景与问题
当前 `OmniVLA` 推理脚本可完成离线单次推理，但不适合手机端连续视频输入场景：
- 模型加载耗时高，不能每帧重启进程
- 缺少会话管理、实时通信协议、端到端时延指标
- 缺少手机端可消费的标准轨迹返回格式

目标是实现“手机采集视频帧 -> 服务端 OmniVLA 常驻推理 -> 轨迹回传手机”闭环。

## 3. 产品目标（Goals）
- 支持手机 APP 连续上传帧，服务端实时返回轨迹/速度
- 模型常驻 GPU，启动后不重复加载权重
- 端到端时延满足可交互（首版目标 p95 <= 450ms）
- 协议稳定，便于后续接机器人底盘或导航模块

## 4. 非目标（Non-Goals）
- 首版不做多机调度和自动扩缩容
- 首版不做多人多机器人复杂权限体系
- 首版不做完整 SLAM/定位系统，仅消费 current/goal 图像和可选指令
- 首版不做 WebRTC 复杂链路（可列为二期）

## 5. 目标用户与核心场景
- **用户**：算法工程师、机器人应用工程师、测试人员
- **场景 A**：手机拍摄当前画面 + 目标画面，连续得到轨迹建议
- **场景 B**：手机输入语言指令（如“沿走廊前进并避开纸箱”），与视觉目标同步推理
- **场景 C**：调试时查看每帧时延、模型耗时、队列积压情况

## 6. 方案总览（推荐先做 v1）

### 6.1 架构分层
1. **Mobile APP**
   - Camera Capture（采集）
   - Frame Sampler（抽帧 3~5 FPS）
   - WS Client（发送帧/接收轨迹）
   - Trajectory Renderer（轨迹可视化）
2. **Inference Server（FastAPI + WebSocket）**
   - Session Manager（会话状态）
   - Frame Queue（每会话 `maxsize=1`，只保留最新帧）
   - OmniVLA Engine（模型常驻单例）
   - Postprocess（轨迹与速度输出）
3. **Observability**
   - per-frame timing（decode/infer/post/total）
   - 错误码统计、会话状态、GPU 内存监控

### 6.2 推荐通信方式
- **v1**：WebSocket 双向通信（实现快）
- **v2**：WebRTC（低延迟、弱网优化）

## 7. 功能需求（FR）

### FR-01 会话管理
- APP 可创建会话 `start_session`
- 会话绑定：`session_id`、`goal_image`、`instruction`、配置项
- 支持 `stop_session` 主动释放资源

### FR-02 连续帧输入
- APP 以 3~5 FPS 上传 `current_image`
- 服务端按会话独立排队，过载时丢旧帧保新帧
- 每帧包含 `seq` 与 `timestamp_ms`

### FR-03 推理与输出
- 服务端调用 OmniVLA 常驻引擎进行单次前向
- 返回：
  - `waypoints`（建议保持 8x4）
  - `linear_vel`、`angular_vel`
  - `modality_id`
  - `timing_ms`

### FR-04 指令同步
- 支持可选 `instruction` 文本
- 协议中显式字段传递，服务端可选择写入 prompt
- 未传指令时走纯视觉目标路径

### FR-05 错误处理
- 图像解码失败、会话不存在、推理超时等需返回标准错误码
- 错误响应必须包含 `session_id`、`seq`、`error_code`、`message`

### FR-06 可观测性
- 每帧记录：
  - `queue_wait_ms`
  - `infer_ms`
  - `post_ms`
  - `total_ms`
- 支持按会话查询统计（平均/95 分位）

## 8. 非功能需求（NFR）
- **时延**：p95 `total_ms <= 450ms`（单会话、目标 3~5 FPS）
- **稳定性**：连续运行 30 分钟无崩溃、无显存泄漏增长
- **恢复性**：单帧失败不影响后续帧
- **可维护性**：推理引擎与通信层解耦，便于后续切换 WebRTC/HTTP
- **安全性**：至少支持 Token 鉴权（内网可先简化）

## 9. 接口草案（WebSocket）

### 9.1 连接地址
- `ws://<server>/ws/v1/omnivla`

### 9.2 消息类型（首版 JSON）
1. `start_session`
2. `frame`
3. `stop_session`
4. `trajectory`（服务端下行）
5. `error`（服务端下行）

### 9.3 示例：start_session（APP -> Server）
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

### 9.4 示例：frame（APP -> Server）
```json
{
  "type": "frame",
  "session_id": "sess_001",
  "seq": 128,
  "timestamp_ms": 1710000000000,
  "current_image_jpeg_b64": "<base64>"
}
```

### 9.5 示例：trajectory（Server -> APP）
```json
{
  "type": "trajectory",
  "session_id": "sess_001",
  "seq": 128,
  "linear_vel": 0.23,
  "angular_vel": -0.05,
  "modality_id": 6,
  "waypoints": [[0.1, -0.1, 0.9, 0.1]],
  "timing_ms": {
    "queue_wait": 6,
    "infer": 162,
    "post": 8,
    "total": 211
  }
}
```

## 10. 时延预算（目标）
| 阶段 | 预算 |
|---|---|
| 手机端编码+发送 | 40~120ms |
| 服务器排队等待 | 0~30ms |
| OmniVLA 前向 | 120~220ms（GPU） |
| 后处理+回传 | 20~60ms |
| **总计** | **220~430ms** |

## 11. 里程碑（建议）
- **M0（0.5 天）**：PRD 评审与范围冻结
- **M1（1~2 天）**：服务端 `OmniVLAEngine` 抽象 + 常驻加载
- **M2（1~2 天）**：WebSocket 协议 + 单会话闭环
- **M3（1 天）**：手机端最小可用 Demo（持续帧 + 轨迹展示）
- **M4（1 天）**：稳定性压测与日志面板

## 12. 验收标准（DoD）
1. 启动后模型仅加载一次，后续连续帧不重复加载权重
2. 单会话连续 5 分钟推理稳定，帧流不中断
3. 可在手机端实时看到 `linear_vel/angular_vel/waypoints`
4. 端到端 `total_ms` 达到目标范围（至少有可量化报告）
5. 出错时 APP 能收到结构化 `error` 响应

## 13. 关键风险与应对
- **风险 1：队列堆积导致延迟越来越大**
  - 应对：`maxsize=1` 丢旧帧，固定“只处理最新帧”
- **风险 2：显存不足或碎片化**
  - 应对：单模型实例、避免反复构造、定期监控显存
- **风险 3：网络抖动**
  - 应对：降 FPS、自适应压缩率、后续切 WebRTC

## 14. 待确认项（请你拍板）
1. **手机端技术栈**：Flutter / Android Kotlin / iOS Swift
2. **首版协议**：仅 WebSocket（推荐）还是直接 WebRTC
3. **首版并发目标**：单会话优先，还是需要多会话并发
4. **返回结构**：是否固定返回 `waypoints + linear/angular`（推荐保留）
5. **部署形态**：先内网单机 GPU 服务，还是需要公网访问

---

如果你确认这份 PRD 方向，我下一步可以基于本文件输出：
- `技术设计文档（TDD）`
- `API 详细字段规范（含错误码）`
- `任务拆解清单（按日程可执行）`
