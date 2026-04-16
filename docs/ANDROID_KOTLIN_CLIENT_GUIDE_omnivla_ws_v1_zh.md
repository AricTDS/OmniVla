# Android Kotlin 接入指南：OmniVLA WebSocket v1

## 1. 目标
- 使用 Android CameraX 采集视频帧
- 通过 WebSocket 持续发送 `frame`
- 接收 `trajectory` 并实时展示速度与轨迹

## 2. 建议工程结构
```text
app/src/main/java/com/example/omnivla/
  data/
    model/               # 协议数据类
    ws/                  # WebSocket 客户端
  camera/
    CameraFrameSampler.kt
  ui/
    MainViewModel.kt
    TrajectoryOverlayView.kt
  util/
    ImageCodec.kt
```

## 3. 依赖建议
- `okhttp`：WebSocket
- `kotlinx-serialization-json` 或 `moshi`：JSON 序列化
- `androidx.camera:camera-camera2` + `camera-lifecycle` + `camera-view`
- 协程：`kotlinx-coroutines-android`

## 4. 协议数据类（示例）
```kotlin
@Serializable
data class StartSessionMsg(
    val type: String = "start_session",
    val session_id: String,
    val instruction: String? = null,
    val goal_image_jpeg_b64: String,
    val config: Config = Config()
) {
    @Serializable
    data class Config(val fps: Int = 4)
}

@Serializable
data class FrameMsg(
    val type: String = "frame",
    val session_id: String,
    val seq: Int,
    val timestamp_ms: Long,
    val current_image_jpeg_b64: String
)

@Serializable
data class TrajectoryMsg(
    val type: String,
    val session_id: String,
    val seq: Int,
    val linear_vel: Double,
    val angular_vel: Double,
    val modality_id: Int,
    val waypoints: List<List<Double>>,
    val timing_ms: TimingMs
) {
    @Serializable
    data class TimingMs(
        val queue_wait: Int,
        val infer: Int,
        val post: Int,
        val total: Int
    )
}
```

## 5. WebSocket 客户端骨架（示例）
```kotlin
class OmniVlaWsClient(
    private val url: String,
    private val onTrajectory: (TrajectoryMsg) -> Unit,
    private val onError: (String) -> Unit
) {
    private val client = OkHttpClient()
    private val json = Json { ignoreUnknownKeys = true }
    private var ws: WebSocket? = null

    fun connect() {
        val req = Request.Builder().url(url).build()
        ws = client.newWebSocket(req, object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val node = json.parseToJsonElement(text).jsonObject
                    when (node["type"]?.jsonPrimitive?.content) {
                        "trajectory" -> onTrajectory(json.decodeFromString(text))
                        "error" -> onError(text)
                    }
                } catch (e: Exception) {
                    onError("parse failed: ${e.message}")
                }
            }
        })
    }

    fun sendStartSession(msg: StartSessionMsg) {
        ws?.send(json.encodeToString(msg))
    }

    fun sendFrame(msg: FrameMsg) {
        ws?.send(json.encodeToString(msg))
    }

    fun close() {
        ws?.close(1000, "bye")
    }
}
```

## 6. 相机抽帧建议
- 使用 CameraX `ImageAnalysis` 回调
- 在回调层加节流，控制到 3~5 FPS
- 每帧转 JPEG 后再 base64
- 图片建议先压到较小分辨率（如宽 640）

## 7. 图像编码建议
- JPEG 质量：`65~80`
- 单帧建议 `< 500KB`
- 若网络抖动明显，可动态降质量或降 FPS

## 8. ViewModel 状态管理建议
- 保存 `latestSeqSent` 与 `latestSeqShown`
- 收到 `trajectory.seq < latestSeqShown` 时丢弃（防乱序闪烁）
- UI 展示：
  - `linear_vel`
  - `angular_vel`
  - `timing_ms.total`

## 9. 与服务端联调步骤
1. 启动服务端：
   - `env -u LD_LIBRARY_PATH -u DYLD_LIBRARY_PATH conda run -n omnivla uvicorn server.app:app --host 0.0.0.0 --port 8000`
2. APP 连接内网地址：`ws://<gpu-host>:8000/ws/v1/omnivla`
3. 先发 `start_session`（带 goal 图与指令）
4. 连续发送 `frame`
5. 确认稳定接收 `trajectory`

## 10. MVP 验收标准
- 可持续发送帧 5 分钟不断连
- 可稳定显示速度与轨迹
- 服务端错误（如坏图）可在 APP 可视提示
- 端到端延迟达到预期区间（建议 < 450ms p95）

## 11. 后续优化方向
- 改二进制帧（替代 base64）
- 引入断线重连与会话恢复
- 支持 WebRTC 视频通道（v2）
