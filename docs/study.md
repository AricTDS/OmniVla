# OmniVLA 学习记录（服务化与参数输入）

## 1) 当前服务化链路：数据输入/输出位置

在当前实现中，数据主链路是：**WebSocket 输入 -> 服务端排队/推理 -> WebSocket 输出**。

### 1.1 输入入口（手机/客户端 -> 服务端）
- 位置：`server/ws_router.py` 的 `omnivla_ws()`
- 收包点：`websocket.receive_text()`
- `start_session` 关键输入：`goal_image_jpeg_b64`、`instruction`
- `frame` 关键输入：`current_image_jpeg_b64`
- 图像解码：`decode_jpeg_b64()`
- 入队：`frame_buffer.push(PendingFrame(...))`

### 1.2 送入模型（服务端内部）
- 位置：`server/ws_router.py` 的 `consumer_loop()`
- 取帧：`frame_buffer.pop(...)`
- 推理调用：`engine.infer(...)`

### 1.3 模型推理与结果计算
- 位置：`server/omnivla_engine.py` 的 `OmniVLAEngine.infer()`
- 输入打包：`inference.data_transformer_omnivla(...)`
- 前向：`inference.run_forward_pass(...)`
- 输出：`waypoints`、`linear_vel`、`angular_vel`、`modality_id`

### 1.4 输出回客户端（服务端 -> 手机）
- 位置：`server/ws_router.py`
- 响应构造：`TrajectoryMessage(...)`
- 发送：`send_model(...)` -> `websocket.send_json(...)`

### 1.5 回放脚本（本地模拟手机）
- 输入发送：`tools/ws_replay_client.py` 的 `ws.send(json.dumps(...))`
- 输出接收：`recv_until(...)` 后打印 `seq / linear / angular / timing`

> 备注：当前服务默认 `enable_visualization=False`，因此默认不落盘轨迹图，主要通过 WebSocket 回包和日志观察结果。

---

## 2) 结论：OmniVLA 模型参数输入方式

### 2.1 模态输入组合（共 9 种）
由 `inference/run_omnivla.py` 中四个开关决定：`satellite`、`lan_prompt`、`pose_goal`、`image_goal`，映射到 `modality_id`：

| modality_id | 组合 |
|---|---|
| 0 | satellite only |
| 1 | satellite + pose |
| 2 | satellite + image |
| 3 | satellite + pose + image |
| 4 | pose only |
| 5 | pose + image |
| 6 | image only |
| 7 | language only |
| 8 | language + pose |

### 2.2 当前服务端实际启用的方式
在 `server/omnivla_engine.py` 中固定为：
- `pose_goal=False`
- `satellite=False`
- `image_goal=True`
- `lan_prompt=False`

即当前服务默认走 **`modality_id=6`（image only）**。  
`instruction` 目前是写入 prompt 文本，不会切换到 `lan_prompt=True` 的语言模态分支。

### 2.3 参数输入的 3 个入口层
1. **命令行**：`inference/run_omnivla.py`（`--current`、`--goal`、`-i`）
2. **Python 接口**：`OmniVLAEngine.infer(current_image_pil, goal_image_pil, instruction)`
3. **WebSocket 协议**：`start_session`（goal + instruction）+ `frame`（current）

---

## 3) 模型推理全流程：从输入到路径输出

以默认模态 `image_goal=True`（`modality_id=6`）为例，完整数据流转如下。

### 3.1 原始输入

| 输入 | 来源 | 形式 |
|------|------|------|
| **当前图像** | 机器人自车摄像头 | PIL Image (RGB) |
| **目标图像** | 用户指定的目标场景 | PIL Image (RGB) |
| **目标位姿** (goal_pose) | GPS/UTM 坐标差 → 本体坐标系 | `[rel_y, -rel_x, cos(Δheading), sin(Δheading)]`，4 维 |
| **语言指令** | 用户文本 | 字符串，如 `"move toward blue trash bin"` |

goal_pose 预处理（`run_omnivla()`）：GPS 差分 → UTM 坐标 → 本体坐标系旋转 → 除以 `metric_waypoint_spacing`(0.1) 归一化。

### 3.2 Prompt 构建与 Tokenize

`PurePromptBuilder` 生成固定格式文本：

```
In: What action should the robot take to move toward blue trash bin?
Out: <action_token×32></s>
```

- 偶数轮 human 包装为 `In: ...\nOut: `，奇数轮 gpt 包装为 `...{eos}`
- action tokens 为 **dummy 占位**（32 = 8步 × 4维），推理时不依赖离散 token，而用连续回归头
- `ActionTokenizer` 的离散化机制：`np.digitize` 映射到词表末尾 256 个 bin（`token_id = vocab_size - bin_index`）

两张图分别经 `processor.image_processor.apply_transform` 预处理后 cat 为 `pixel_values`。

### 3.3 视觉特征提取

进入 `vla(...)` → `PrismaticForConditionalGeneration_MMNv1.forward()`：

```
pixel_values (B, 2×C, H, W)
  ↓ 双 ViT 骨干（DINOv2 + SigLIP）
  ↓ 每张图 → 256 个 patch，两个 ViT 的 embed_dim concat
  ↓ 两张图 → 512 个 patch
  = patch_features (B, 512, vision_dim)
```

### 3.4 视觉投影器 (PrismaticProjector)

```
patch_features (B, 512, vision_dim)
  ↓ fc1: vision_dim → 4×vision_dim
  ↓ GELU
  ↓ fc2: 4×vision_dim → llm_dim
  ↓ GELU
  ↓ fc3: llm_dim → llm_dim
  = projected_patch_embeddings (B, 512, llm_dim)
```

### 3.5 目标位姿投影 (ProprioProjector)

```
goal_pose (B, 4)
  ↓ fc1: 4 → llm_dim
  ↓ GELU
  ↓ fc2: llm_dim → llm_dim
  = (B, llm_dim) → unsqueeze(1) → (B, 1, llm_dim)
  ↓ cat 到 patch 序列末尾
  = projected_patch_embeddings (B, 513, llm_dim)
       512 视觉 patches + 1 位姿 token
```

### 3.6 多模态序列拼接

```
[BOS] [视觉patch×512 + 位姿token×1] [文本tokens~20] [动作tokens×32(清零)] [EOS]
  1个          513个                    ~20个              32个             1个
                                                 总计 ≈ 567 tokens
```

- 动作位置的 embedding **乘以 0**（清零），仅保留位置编码占位
- 动作位置识别：`get_current_action_mask` / `get_next_actions_mask` 通过 `labels` 中 `!= IGNORE_INDEX` 的累计和 + `token_id > ACTION_TOKEN_BEGIN_IDX(31743)` 联合判定

### 3.7 MMN Attention Mask（核心设计）

根据 `modality_id` 按模态屏蔽 patch，**同一模型通过 attention mask 实现 9 种模态组合**：

```
modality_id=6 (image only) 时：
  自车图 patches (前256):  ✅ 参与
  目标图 patches (后256):  ✅ 参与 (image_goal=True)
  位姿 token (第513):      ❌ 被 mask (pose_goal=False)
  文本 / 动作 tokens:      ✅ 参与
```

### 3.8 LLM 前向传播

```
multimodal_embeddings (B, ~567, llm_dim)
  ↓ Llama-based Transformer (多层 self-attention + FFN)
  ↓ output_hidden_states=True
  = hidden_states[-1] → last_hidden_states (B, ~567, llm_dim)
```

### 3.9 提取动作隐状态

```python
text_hidden_states = last_hidden_states[:, num_patches:-1]
# num_patches = 513, 跳过视觉段

actions_hidden_states = text_hidden_states[action_mask].reshape(B, 32, -1)
# 32 = NUM_ACTIONS_CHUNK(8) × ACTION_DIM(4)
# 形状 (B, 32, llm_dim)
```

### 3.10 Action Head 回归预测

`L1RegressionActionHead_idcat`：

```
actions_hidden_states (B, 32, llm_dim)
  ↓ reshape(B, 8, 4×llm_dim)     ← 每步 4 个 token 的 hidden 拼成一个向量
  ↓ LayerNorm
  ↓ cat(modality_id)              ← 拼接模态标识，广播到 8 步
  ↓ fc1: 4×llm_dim+1 → hidden_dim
  ↓ Dropout + ReLU
  ↓ ResNet Block × 2 (LayerNorm → Linear → Dropout → ReLU → Linear → Dropout + 残差)
  ↓ LayerNorm → Linear → output_dim=4
  = predicted_actions (B, 8, 4)
       8 个路径点: [dx, dy, cos(heading), sin(heading)]
```

输出含义：
- `dx` / `dy`：前方/左方归一化位移（单位 = `metric_waypoint_spacing`，默认 0.1m）
- `cos(heading)` / `sin(heading)`：该路径点的朝向分量

### 3.11 路径点 → 速度指令

```python
waypoint_select = 4              # 选第 5 个路径点
chosen[:2] *= metric_waypoint_spacing  # 反归一化 → 米
dx, dy, hx, hy = chosen

linear_vel  = dx / DT            # DT = 1/3 s
angular_vel = arctan(dy/dx) / DT
# clip 到 maxv=0.3 m/s, maxw=0.3 rad/s
```

### 3.12 端到端总结

```
┌──────────────────────────────────────────────────────────┐
│                       输入层                              │
│  当前图像(RGB) + 目标图像(RGB) + 目标位姿(4D) + 语言指令   │
└────────┬───────────────┬──────────────┬──────────────────┘
         │               │              │
    ┌────▼────┐    ┌─────▼─────┐  ┌────▼────┐
    │ DINOv2  │    │ProprioProj│  │Tokenizer│
    │+ SigLIP │    │ (2层MLP)  │  │(文本→id)│
    │ 双ViT   │    └─────┬─────┘  └────┬────┘
    └────┬────┘          │              │
    512个patch       1个token       ~20个token
         │               │              │
    ┌────▼────┐          │              │
    │Projector│          │              │
    │(3层MLP) │          │              │
    └────┬────┘          │              │
         │               │              │
    ┌────▼───────────────▼──────────────▼──────────────┐
    │  [BOS][patch×512][pose×1][text~20][action×32]    │
    │        多模态序列 (~567 tokens)                    │
    │        + MMN Attention Mask (按模态屏蔽)           │
    └────────────────────┬─────────────────────────────┘
                         │
                    ┌────▼─────┐
                    │   LLM    │
                    │ (Llama)  │
                    │ 多层     │
                    │Transformer│
                    └────┬─────┘
                         │
                 hidden_states[-1]
                         │
                 取动作段 32 个 token
                         │
                 reshape(8, 4×H)
                         │
                    ┌────▼─────────┐
                    │ Action Head  │
                    │ MLPResNet    │
                    │+modality_id  │
                    │(2 ResBlocks) │
                    └────┬─────────┘
                         │
                    (B, 8, 4)
              8个路径点 [dx,dy,cos,sin]
                         │
                    ┌────▼────┐
                    │PD控制器  │
                    │选第5个点  │
                    │→速度指令  │
                    └─────────┘
              linear_vel, angular_vel
```

**核心设计**：OmniVLA 本质是 **VLM（视觉语言模型）+ 连续回归头**。不用 LLM 自回归生成离散动作 token，而是：
1. 利用 LLM 的 Transformer 作为**多模态融合器**，让视觉/位姿/语言在 attention 中交互
2. 取出动作位置的**隐层表征**（非 logits）
3. 通过独立的 **Action Head MLP** 回归连续路径坐标

兼得预训练 LLM 的表征能力，又避免离散化的精度损失。

---

## 4) 后续可选优化（待做）
- 将模态开关从固定值改为会话配置（如 `start_session.config` 传 `use_language/use_pose/use_satellite`）
- 增加“是否落盘可视化图片”的环境变量开关
- 输出更细粒度时延拆解（解码、排队、前向、回包）

