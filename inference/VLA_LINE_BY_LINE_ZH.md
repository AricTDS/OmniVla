# OmniVLA 推理脚本逐行导读（`run_omnivla.py`）

本文档与 `VLA_CODE_GUIDE_ZH.md` 配套：后者偏「模块/流程」，本文按**源码行号**对 `inference/run_omnivla.py` 做**逐行（或逐小段）**说明，便于你对照编辑器阅读。

- **对照文件**：`inference/run_omnivla.py`（行号以你当前工作区版本为准）。
- **阅读顺序建议**：`__main__` → `define_model` → `Inference.run_omnivla` → `run_forward_pass` → 数据管线（`data_transformer_omnivla` / `collator_custom`）。

---

## 1. 文件头与依赖（约 L1–L41）

| 行号 | 代码要点 | 含义 |
|------|----------|------|
| L1–L9 | 注释块 | 说明这是 OmniVLA 的示例推理脚本；若要接真机，需要在 `run_omnivla` 里更新当前位姿/图像，并去掉 `run` 里的 `break`。 |
| L11–L12 | `import sys, os` + `sys.path.insert(0, '..')` | 把上级目录加入 `PYTHONPATH`，使 `from prismatic...` 能解析到仓库根下的 `prismatic` 包。 |
| L14–L26 | `time/math/json`、`numpy`、`PIL`、`torch`、`matplotlib`、`utm` | 时间循环、数学、JSON（若扩展）、数组与张量、图像 I/O、绘图、经纬度↔UTM 转换。 |
| L22 | `pad_sequence` | 批内 padding（本脚本 batch=1 也会走统一接口）。 |
| L23 | `DDP` | 类型标注/习惯写法：权重可能来自 DDP 训练保存的 `module.` 前缀。 |
| L24 | `torchvision.transforms` | 图像预处理通常由 HF `Processor` 封装；此处间接通过 `processor.image_processor`。 |
| L31–L39 | `prismatic` 各模块 | **ActionTokenizer**：动作离散 token；**ProprioProjector**：把目标位姿向量投影到 LLM 维度；**action_heads**：从隐藏态回归连续动作；**OpenVLA***：VLA 主体与配置/处理；**prompting**：拼对话式 prompt；**train_utils**：从 labels 里切出「当前动作 token / 未来动作 token」的 mask；**constants**：动作维度、chunk 长度、位姿维度等常量。 |
| L41 | `transformers.Auto*` | 用 HuggingFace 的自动类加载 `config/processor/model`。 |

---

## 2. 权重与模块初始化工具（约 L46–L79）

| 行号 | 代码要点 | 含义 |
|------|----------|------|
| L46–L47 | `remove_ddp_in_checkpoint` | 若 checkpoint 的 key 以 `module.` 开头（DDP），去掉前缀以匹配单卡 `nn.Module`。 |
| L49–L55 | `load_checkpoint` | 按 `{module_name}--{step}_checkpoint.pt` 读取子模块权重；**兼容命名**：若找不到 `pose_projector` 则尝试 `proprio_projector`（历史命名）。 |
| L57–L59 | `count_parameters` | 打印可训练参数量（推理时通常只用于确认模块规模）。 |
| L61–L79 | `init_module` | 构造子模块 → 可选加载 ckpt → 可选转 `bfloat16` → `.to(device)`。返回类型标注为 `DDP`，实际返回的是普通 `nn.Module`（历史遗留/误标）。 |

---

## 3. `Inference` 类：状态与主循环（约 L84–L122）

| 行号 | 代码要点 | 含义 |
|------|----------|------|
| L84–L95 | `__init__` | 保存语言提示、目标 UTM/朝向、目标图、`ActionTokenizer`、`Processor`，以及输出目录；`tick_rate` 控制循环频率；`linear/angular` 为控制量缓存。 |
| L99–L107 | `calculate_relative_position` / `rotate_to_local_frame` | 把全局平面位移（UTM 差）旋转到**以当前车头朝向为 x 轴**的局部坐标，供后续归一化与可视化。 |
| L112–L119 | `run` | 以固定周期调用 `tick()`；当前实现**第一次 tick 后就 `break`**，因此是「单步演示」而非连续控制。 |
| L121–L122 | `tick` | 调用 `run_omnivla()` 得到线速度/角速度并写回成员变量。 |

---

## 4. `run_omnivla`：从 GPS/图到动作（约 L127–L244）

| 行号 | 代码要点 | 含义 |
|------|----------|------|
| L128–L129 | `thres_dist` / `metric_waypoint_spacing` | 远距离裁剪阈值（米）与把网络输出格子还原成米制位移的缩放因子。 |
| L131–L136 | 当前 GPS/朝向 + `utm.from_latlon` | 将当前经纬度转 UTM；朝向取负并按 π 换算到弧度（与训练坐标约定一致）。 |
| L138–L146 | 相对位移 + 半径裁剪 | 计算当前到目标的相对向量；若超过 `thres_dist` 则按比例缩小，避免数值过大。 |
| L148–L153 | `goal_pose_loc_norm` | 把相对位置按 `metric_waypoint_spacing` 归一化，并拼接目标朝向相对当前朝向的 **cos/sin**，形成 **4 维**目标位姿特征（与 `POSE_DIM` 对齐）。 |
| L155–L157 | 读取当前相机图 | 默认 `./inference/current_img.jpg`。 |
| L159–L160 | `lan_inst` | 若全局 `lan_prompt` 为真则用 `self.lan_inst_prompt`，否则占位串 `"xxxx"`（与下游 `transform_datatype` 分支对应）。 |
| L162–L168 | `data_transformer_omnivla` | 把图像/语言/目标位姿等打包成模型输入 batch（内部会构造伪动作标签，仅用于对齐训练时的 token 布局）。 |
| L170–L187 | `run_forward_pass` | 前向：VLA 输出隐藏态 → `action_head` 回归动作 chunk；同时返回 `modality_id`（本文件用全局布尔量决定）。 |
| L188 | `count_id` 自增 | 用于保存可视化文件名递增。 |
| L190–L196 | `waypoints` / `waypoint_select` | 将预测张量转 numpy；`waypoint_select` 选取 chunk 中某一帧作为当前控制目标。 |
| L196 | `chosen_waypoint[:2] *= metric_waypoint_spacing` | 把归一化平面位移还原为米（与 L129 一致）。 |
| L198–L209 | 简易运动学映射 | 由 `(dx,dy)` 与 `(hx,hy)` 推线速度/角速度；其中 `clip_angle(...)` **应在别处定义或改为 `np.clip` 等**（当前文件内未检索到定义，若走到该分支会 `NameError`）。 |
| L211–L235 | clip + 饱和分配 | 先粗 clip，再按 `maxv/maxw` 做耦合限制，得到最终 `*_limit`。 |
| L237–L241 | `save_robot_behavior` | 保存三联图：当前图、目标图、轨迹+模态标注；并把速度打印出来。 |
| L243–L244 | `return` | 返回本控制周期的线/角速度标量。 |

---

## 5. 可视化 `save_robot_behavior`（约 L249–L289）

| 行号 | 代码要点 | 含义 |
|------|----------|------|
| L251–L255 | `matplotlib` 布局 | 2×2 网格：左列上下两张图，右列跨两行画轨迹。 |
| L257–L258 | `imshow` | 显示当前/目标 egocentric 图像。 |
| L260–L262 | 轨迹绘制 | 注释说明网络轨迹在**机器人坐标**：x 前、y 左；绘图时对 y 取负以适配图像坐标习惯。 |
| L264–L271 | `mask_type` 文本 | `modality_id` 的整型索引映射到人类可读字符串（与 `run_forward_pass` 分支一致）。 |
| L278–L284 | 目标点绘制 / 坐标轴 | 某些模态下把目标位姿画成红星；随后统一 `xlim/ylim`。 |

---

## 6. 批处理 `collator_custom`（约 L294–L329）

| 行号 | 代码要点 | 含义 |
|------|----------|------|
| L294–L299 | padding `input_ids/labels` | `labels` 用 `-100` 忽略非动作监督位置（与训练一致）。 |
| L301–L314 | `pixel_values` 拼接规则 | 若存在 `pixel_values_goal`，则在**帧维** `dim=1` 把 current+goal 拼起来，喂给双图视觉骨干。 |
| L316–L317 | `actions` / `goal_pose` | 堆叠为张量；其中 `actions` 在本推理脚本里是随机占位（L388），**不参与损失**（因为 `run_forward_pass` 在 `torch.no_grad()` 下且主要用隐藏态回归）。 |
| L319–L325 | `output` 字典 | 组装模型前向所需键；注意 `pixel_values.to()` **通常应带 device/dtype 参数**（若你本地改过请以此为准）。 |

---

## 7. 单样本构造 `transform_datatype` / `data_transformer_omnivla`（约 L334–L406）

| 行号 | 代码要点 | 含义 |
|------|----------|------|
| L337–L343 | 动作 chunk 字符串化 | 用 `ActionTokenizer` 把连续动作伪标签编码进对话的 assistant 段，保证 token 对齐。 |
| L345–L354 | `conversation` | `inst_obj=="xxxx"` 时表示**无语言**的特殊 prompt；否则嵌入自然语言任务描述。 |
| L356–L358 | `PurePromptBuilder("openvla")` | 按 OpenVLA 约定拼多轮对话模板。 |
| L360–L365 | `input_ids` / `labels` mask | 只对「动作 token 区域」计算语言建模损失的位置留真值，其余为 `IGNORE_INDEX`（推理时也会生成同样形状的 `labels` 以复用训练代码路径）。 |
| L367–L369 | 图像张量 | 对 current/goal 各做一遍 `image_processor.apply_transform`。 |
| L371–L381 | 返回 dict | 单样本字段集合，供 collator 使用。 |
| L386–L406 | `data_transformer_omnivla` | `actions = np.random.rand(8, 4)`：**占位**，为了让 tokenizer 路线与训练 shape 对齐；再 `collator_custom` 得到 batch=1。 |

---

## 8. `run_forward_pass`：模态路由 + VLA + 动作头（约 L411–L476）

| 行号 | 代码要点 | 含义 |
|------|----------|------|
| L416 | `metrics = {}` | 预留指标字典（当前返回未用）。 |
| L417 | `noise...` | 扩散相关占位；`use_diffusion=False` 时不参与。 |
| L419–L437 | `modality_id` 分支 | 由四个全局布尔量组合决定整数 ID：`satellite`、`lan_prompt`、`pose_goal`、`image_goal`。这决定 VLA 内部**哪些输入通路被启用/被 mask**。 |
| L439–L453 | `vla(...)` | `torch.autocast(..., bfloat16)` 下前向；关键输入：`pixel_values`（双图）、`modality_id`、`proprio`（目标位姿）、`proprio_projector`（`pose_projector`）。 |
| L456–L458 | 从 `labels` 派生 mask | `get_current_action_mask` / `get_next_actions_mask` 用于定位「动作 token」在序列中的位置。 |
| L461–L470 | 取隐藏态 | 最后一层 hidden states；去掉 vision patch 部分，再在动作 mask 位置 gather，reshape 为 `(B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)`。 |
| L472–L473 | `action_head.predict_action` | 把动作隐变量回归为连续动作（同时吃 `modality_id` 以切换头内行为）。 |
| L475–L476 | `return` | 返回预测动作与 `modality_id`（供可视化）。 |

**`modality_id` 与注释文本对应（摘自 `save_robot_behavior` 的文案顺序）**

- 0：仅卫星（satellite only）
- 1：位姿 + 卫星
- 2：卫星 + 目标图
- 3：卫星 + 位姿 + 目标图（all，在卫星设定下）
- 4：仅位姿
- 5：位姿 + 目标图
- 6：仅目标图
- 7：仅语言
- 8：语言 + 位姿

---

## 9. `InferenceConfig` 与 `define_model`（约 L482–L557）

| 行号 | 代码要点 | 含义 |
|------|----------|------|
| L482–L494 | `InferenceConfig` dataclass 风格字段 | `vla_path` / `resume_step` 指向权重目录与迭代步；`use_l1_regression` 控制动作头类型；`num_images_in_input=2` 表示 current+goal 双图；`use_lora` 等为结构超参。 |
| L496–L498 | 规范化路径 + 打印 | `rstrip("/")` 避免拼接重复斜杠。 |
| L500–L504 | `device_id` | 有 CUDA 用 `cuda:0`，否则 CPU；仅在 CUDA 时 `set_device/empty_cache`。 |
| L506–L512 | 打印常量 | 与权重匹配的架构超参自检。 |
| L514–L518 | `AutoConfig.register...` | 把自定义 OpenVLA 类注册进 Transformers 自动解析表。 |
| L521–L526 | `from_pretrained` | 加载 processor 与 VLA；`torch_dtype=bfloat16`；`trust_remote_code=True` 允许远程/本地自定义代码。 |
| L528–L529 | `set_num_images_in_input` + `vla.to(...)` | 明确视觉输入帧数，并把模型放到目标设备与 dtype。 |
| L531–L538 | `pose_projector` | 位姿投影 MLP；`to_bf16=True` 与 VLA 的 bf16 路径对齐。 |
| L540–L548 | `action_head` | L1 回归头；同样 `to_bf16=True`。 |
| L550–L552 | `NUM_PATCHES` | patch 数 × 图像帧数，并 `+1` 注释说明为 goal pose 相关附加 patch 计数约定（与模型实现一致为准）。 |
| L554–L557 | `ActionTokenizer` + `return` | 返回推理所需全套对象给 `__main__`。 |

---

## 10. `__main__` 入口（约 L562–L590）

| 行号 | 代码要点 | 含义 |
|------|----------|------|
| L563–L567 | 四个全局开关 | **这是本脚本选择 `modality_id` 的唯一入口**（与 `run_forward_pass` 顶部 if-elif 链联动）。 |
| L569–L574 | 目标 GPS/朝向/图像 | 示例场景：目标经纬度、目标朝向、目标图路径。 |
| L576–L578 | `define_model` | 加载 VLA 与子头。 |
| L580–L590 | 构造 `Inference` 并 `run()` | 触发单步演示流程。 |

---

## 11. 阅读这份「逐行版」时的两个实用提醒

1. **全局模态开关**：`pose_goal / satellite / image_goal / lan_prompt` 既影响 `modality_id`，也影响 `lan_inst` 是否走真实语言（见 L159–160 与 `transform_datatype` 分支）。
2. **占位与对齐**：`data_transformer_omnivla` 里的随机 `actions` 是为对齐训练时序列布局；真正用于控制的连续量来自 `action_head` 的回归输出。

---

## 12. 与模块导读的对应关系

更偏「文件职责 / 扩展点 / 与 edge 脚本差异」的说明见：`inference/VLA_CODE_GUIDE_ZH.md`。

若你还希望对 `run_omnivla_edge.py`、`model_omnivla_edge.py` 做同样体量的逐行版，直接点名文件即可。
