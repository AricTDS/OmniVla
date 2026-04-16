# API 详细规范：OmniVLA 实时轨迹 WebSocket v1

## 1. 概述
- **协议版本**：v1
- **传输协议**：WebSocket（文本 JSON）
- **连接地址**：`ws://<server-host>:<port>/ws/v1/omnivla`
- **适用范围**：Android Kotlin 客户端与 OmniVLA 推理服务

## 2. 设计约定
- 每条消息必须包含 `type`
- `session_id` 在会话建立后所有消息必带
- `seq` 用于帧级关联（请求与响应对齐）
- 图像字段采用 JPEG base64（v1 先易实现，v2 可升级二进制）
- 时间字段单位统一毫秒（`timestamp_ms`）

## 3. 消息总览

### 3.1 上行（Client -> Server）
1. `start_session`
2. `frame`
3. `stop_session`
4. `ping`（可选）

### 3.2 下行（Server -> Client）
1. `ack`
2. `trajectory`
3. `error`
4. `pong`（可选）

## 4. 字段规范

### 4.1 通用字段
| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | 消息类型 |
| `session_id` | string | 大多是 | 会话标识，建议 UUID 或 `sess_xxx` |
| `seq` | integer | 帧相关时是 | 帧序号，客户端单调递增 |
| `timestamp_ms` | integer | 推荐 | 客户端发送帧时间戳 |

### 4.2 图像字段
| 字段 | 类型 | 必填 | 约束 |
|---|---|---|---|
| `goal_image_jpeg_b64` | string | `start_session` 必填 | base64 编码 JPEG，建议 <= 500KB |
| `current_image_jpeg_b64` | string | `frame` 必填 | base64 编码 JPEG，建议 <= 500KB |

### 4.3 轨迹字段
| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `waypoints` | number[][] | 是 | 形状 `[N][4]`，v1 默认 `N=8` |
| `linear_vel` | number | 是 | 线速度（m/s） |
| `angular_vel` | number | 是 | 角速度（rad/s） |
| `modality_id` | integer | 是 | OmniVLA 模态 ID |

## 5. 上行消息定义

### 5.1 `start_session`
#### 请求示例
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

#### 字段规则
| 字段 | 类型 | 必填 | 约束/说明 |
|---|---|---|---|
| `type` | string | 是 | 固定 `start_session` |
| `session_id` | string | 是 | 长度 1~64，`[a-zA-Z0-9_-]` |
| `instruction` | string | 否 | 长度 <= 512 |
| `goal_image_jpeg_b64` | string | 是 | 可解码为 JPEG |
| `config.fps` | integer | 否 | 建议 1~10，默认 4 |

#### 服务端行为
- 校验字段并解码 `goal_image`
- 创建/覆盖单会话上下文
- 返回 `ack(start_session)`

### 5.2 `frame`
#### 请求示例
```json
{
  "type": "frame",
  "session_id": "sess_001",
  "seq": 128,
  "timestamp_ms": 1710000000123,
  "current_image_jpeg_b64": "<base64>"
}
```

#### 字段规则
| 字段 | 类型 | 必填 | 约束/说明 |
|---|---|---|---|
| `type` | string | 是 | 固定 `frame` |
| `session_id` | string | 是 | 必须已存在 |
| `seq` | integer | 是 | >= 0，建议单调递增 |
| `timestamp_ms` | integer | 否 | Unix ms，调试延迟用 |
| `current_image_jpeg_b64` | string | 是 | 可解码为 JPEG |

#### 服务端行为
- 写入 `FrameBuffer(maxsize=1)`，必要时覆盖旧帧
- 推理完成后回发同 `seq` 的 `trajectory`

### 5.3 `stop_session`
#### 请求示例
```json
{
  "type": "stop_session",
  "session_id": "sess_001"
}
```

#### 服务端行为
- 释放该会话缓存（模型仍常驻）
- 返回 `ack(stop_session)`

### 5.4 `ping`（可选）
```json
{
  "type": "ping",
  "session_id": "sess_001",
  "timestamp_ms": 1710000000999
}
```

## 6. 下行消息定义

### 6.1 `ack`
```json
{
  "type": "ack",
  "session_id": "sess_001",
  "op": "start_session",
  "message": "ok"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | 固定 `ack` |
| `session_id` | string | 是 | 会话 ID |
| `op` | string | 是 | `start_session` 或 `stop_session` |
| `message` | string | 否 | 提示信息 |

### 6.2 `trajectory`
```json
{
  "type": "trajectory",
  "session_id": "sess_001",
  "seq": 128,
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

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | 固定 `trajectory` |
| `session_id` | string | 是 | 会话 ID |
| `seq` | integer | 是 | 与输入 `frame.seq` 对齐 |
| `linear_vel` | number | 是 | m/s |
| `angular_vel` | number | 是 | rad/s |
| `modality_id` | integer | 是 | 模态编号 |
| `waypoints` | number[][] | 是 | 轨迹点集合 |
| `timing_ms` | object | 是 | 耗时统计 |

### 6.3 `error`
```json
{
  "type": "error",
  "session_id": "sess_001",
  "seq": 128,
  "error_code": "E_IMAGE_DECODE",
  "message": "invalid jpeg payload"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | 固定 `error` |
| `session_id` | string | 否 | 若可定位会话则返回 |
| `seq` | integer | 否 | 若可定位帧则返回 |
| `error_code` | string | 是 | 错误码 |
| `message` | string | 是 | 人类可读错误信息 |

### 6.4 `pong`（可选）
```json
{
  "type": "pong",
  "session_id": "sess_001",
  "timestamp_ms": 1710000000999
}
```

## 7. 错误码规范
| 错误码 | 触发条件 | 客户端建议处理 |
|---|---|---|
| `E_BAD_REQUEST` | JSON 缺字段/类型错误 | 修复请求结构并重发 |
| `E_SESSION_NOT_FOUND` | 会话不存在 | 先 `start_session` |
| `E_IMAGE_DECODE` | 图像解码失败 | 检查编码与压缩参数 |
| `E_MODEL_NOT_READY` | 模型未加载完成 | 等待后重试 |
| `E_INFER_TIMEOUT` | 推理超时 | 降 FPS 或重连 |
| `E_INTERNAL` | 服务内部异常 | 指数退避重试并上报日志 |

## 8. 服务端校验规则（必须实现）
- `start_session` 必须先于 `frame`
- 不接受非当前会话 ID 的帧（v1 单会话）
- `seq` 可容忍跳号，不要求连续
- base64 解码失败直接返回 `E_IMAGE_DECODE`
- 当服务繁忙时，可丢弃旧帧但不能返回旧 `seq` 结果

## 9. 客户端实现建议（Android Kotlin）
- WebSocket 用 OkHttp
- CameraX 抽帧间隔控制在 200~300ms
- 每帧 JPEG 压缩质量建议 65~80
- 收到 `trajectory.seq` 小于最近发送序号时可丢弃显示（避免乱序闪烁）
- 对 `E_MODEL_NOT_READY` 使用短重试（如 500ms）

## 10. 版本兼容策略
- 新增字段使用向后兼容策略（客户端忽略未知字段）
- `type` 语义不变，避免复用含义
- 版本升级建议路径：`/ws/v2/omnivla`

## 11. 联调检查清单
1. `start_session` 后收到 `ack`
2. 连续 `frame` 均收到对应 `trajectory`
3. `seq` 可对齐并统计时延
4. 错误帧能返回正确 `error_code`
5. `stop_session` 后继续发帧会收到 `E_SESSION_NOT_FOUND`
