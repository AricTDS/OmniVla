# OmniVLA Inference 代码导读（中文）

这份文档面向“刚接触 VLA/机器人策略代码”的同学，重点解释 `inference/` 目录里的核心推理代码如何工作、该从哪里改、改了会影响什么。

---

## 1. 先建立直觉：这个推理程序做什么？

`OmniVLA` 的推理流程可以理解为：

1. 读取当前观测（当前图像 + 位姿/GPS + 可选语言）
2. 构造“目标条件”（目标图像 / 目标位姿 / 语言目标）
3. 模型输出一段动作轨迹（action chunk）
4. 从轨迹里取一个 waypoint，转成线速度和角速度
5. 画图保存结果（轨迹 + 当前图 + 目标图）

在代码里，主入口是：

- `inference/run_omnivla.py`（主模型）
- `inference/run_omnivla_edge.py`（轻量 edge 模型）

---

## 2. 你需要先知道的 5 个术语

### 2.1 VLA（Vision-Language-Action）

- `Vision`：图像观测
- `Language`：文本指令
- `Action`：机器人控制动作（速度/轨迹参数）

### 2.2 Action Chunk

模型一次不是只预测一个动作，而是预测一段动作序列（例如 8 个 waypoint）。

### 2.3 Modality Mask / `modality_id`

代码通过 `modality_id` 指定当前启用了哪些输入模态（例如：只有图像、图像+语言、位姿+语言等）。

### 2.4 Proprio / Goal Pose

机器人“自身相关状态”或目标位姿编码，这里主要是目标相对位姿（x/y + 朝向 cos/sin）。

### 2.5 PD-like 控制映射

模型输出轨迹后，代码会做一个基于几何关系的速度映射，再做限幅，得到最终 `linear` / `angular`。

---

## 3. 目录与文件职责

`inference/` 下你最常看的 5 个文件：

- `run_omnivla.py`：主模型推理脚本（最重要）
- `run_omnivla_edge.py`：edge 版本推理脚本
- `model_omnivla_edge.py`：edge 网络结构定义
- `current_img.jpg`：默认当前帧输入
- `goal_img.jpg`：默认目标帧输入

输出通常是：

- `1_ex.jpg`（主模型）
- `1_ex_omnivla_edge.jpg`（edge）

---

## 4. `run_omnivla.py` 深度导读

## 4.1 入口结构

核心调用链：

- `if __name__ == "__main__":`
- `define_model(cfg)`
- `Inference(...).run()`
- `Inference.tick()`
- `Inference.run_omnivla()`
- `Inference.run_forward_pass(...)`

## 4.2 模型加载：`define_model()`

主要做 4 件事：

1. 选择设备（`cuda` 或 `cpu`）
2. 注册 HF AutoClass（`OpenVLAConfig/Processor/Model`）
3. 从 `cfg.vla_path` 加载主模型和 processor
4. 加载两个关键头：
   - `pose_projector`
   - `action_head`

你后续最常改的地方：

- `InferenceConfig.vla_path`
- `InferenceConfig.resume_step`

## 4.3 推理主流程：`run_omnivla()`

这部分是“策略逻辑 + 控制输出”主干，按顺序：

1. 读取当前位姿（示例里是写死的 GPS + compass）
2. 计算目标相对位姿并归一化
3. 加载 `current_img.jpg`
4. 准备语言（如果 `lan_prompt=False`，会用 `"xxxx"` 占位）
5. 调 `data_transformer_omnivla()` 组 batch
6. 调 `run_forward_pass()` 拿预测轨迹
7. 选第 `waypoint_select` 个点转速度
8. 速度限幅（最终上限 `0.3/0.3`）
9. 可视化并保存结果图

## 4.4 前向推理：`run_forward_pass()`

这是模型“真正 forward”的位置：

- 根据全局开关组合出 `modality_id`
- 调 `vla(...)` 得到 hidden states
- 通过 token mask 提取动作 token 对应的 hidden states
- 用 `action_head.predict_action(...)` 回归动作 chunk

注意：这里不是直接读 logits 生成文本，而是走“隐藏状态 -> 动作头”这条路径。

## 4.5 数据组装：`transform_datatype()` + `collator_custom()`

这两部分模仿训练时数据结构，把推理输入包装成模型习惯的 batch 格式。

你会看到 `data_transformer_omnivla()` 里有：

- `actions = np.random.rand(8, 4)`（dummy）

这不是最终控制动作，而是为了构造 prompt/label 对齐格式，真正动作来自 `run_forward_pass()` 的预测结果。

## 4.6 全局开关（非常关键）

主入口里这几个全局变量会影响 `modality_id`：

- `pose_goal`
- `satellite`
- `image_goal`
- `lan_prompt`

因为它们在多个函数里被直接引用，改模态时要注意这些开关的一致性。

---

## 5. `run_omnivla_edge.py` 导读

edge 脚本和主脚本逻辑相似，但有三点明显不同：

1. 不直接走 `OpenVLAForActionPrediction_MMNv1` 主体 forward
2. 使用 `load_model()` 加载 `omnivla-edge.pth`
3. 输入更“工程化”：
   - context 图像队列
   - map 图像（示例里是 dummy）
   - CLIP 文本特征
   - FiLM 相关视觉-语言融合

入口里关键参数集中在 `model_params` 字典，改模型结构/大小先看这里。

---

## 6. `model_omnivla_edge.py` 网络结构拆解

这个文件可以按“3 层”理解：

### 6.1 编码层

- `obs_encoder`（EfficientNet，处理观测序列）
- `goal_encoder`（处理地图/目标相关图像）
- `goal_encoder_img`（处理目标图像或 obs+goal 拼接）
- `compress_*` 线性层做维度对齐

### 6.2 融合层

- `FiLMNetwork`：把语言特征调制到视觉特征上
- `MultiLayerDecoder_mask3`：Transformer 编码器 + mask 机制
- `all_masks` / `avg_pool_mask`：按模态屏蔽 token

### 6.3 预测层

- `action_predictor`：输出轨迹动作
- `dist_predictor`：输出距离相关量
- 后处理：
  - 对前两维做 `cumsum`（累积位移）
  - 对朝向分量做归一化

---

## 7. 你最常改的地方（按目标）

## 7.1 换输入图片

- 替换 `inference/current_img.jpg`
- 替换 `inference/goal_img.jpg`

## 7.2 换语言指令

- `run_omnivla.py` 里改 `lan_inst_prompt`
- 并确保 `lan_prompt=True`

## 7.3 换模型权重

- 主模型：改 `InferenceConfig.vla_path`
- edge：改 `ckpth_path`（默认 `./omnivla-edge/omnivla-edge.pth`）

## 7.4 调输出运动风格

- 改 `waypoint_select`
- 改限速参数 `maxv/maxw`
- 改 `DT`、角度映射公式

---

## 8. 常见坑（你已经遇到过的）

- **CPU 环境与 CUDA 调用冲突**  
  需要保证脚本在 `torch.cuda.is_available()==False` 时不强制 `set_device(cuda)`

- **dtype 不一致（bf16 vs float）**  
  `pose_projector` / `action_head` 与输入 dtype 需要一致

- **edge 缺权重文件**  
  `./omnivla-edge/omnivla-edge.pth` 不存在会直接报错

- **`pkg_resources` 缺失**（edge + clip 常见）  
  可通过合适版本 `setuptools` 解决

---

## 9. 推荐学习路径（代码阅读顺序）

建议按下面顺序读：

1. `run_omnivla.py`：先通读入口和 `run_omnivla()`
2. `run_forward_pass()`：理解动作是如何从 hidden states 回归的
3. `run_omnivla_edge.py`：对比主模型和 edge 的输入差异
4. `model_omnivla_edge.py`：看融合结构（FiLM + Transformer）
5. 再回到 `prismatic/` 深入底层

---

## 10. 下一步建议（你马上可做）

- 第一步：只改 `lan_inst_prompt`，观察输出变化
- 第二步：用你自己的 `current/goal` 图跑 3 组样例
- 第三步：把写死的 GPS 换成你的实时定位输入
- 第四步：把 `break` 去掉，接入真实机器人控制环

如果你愿意，我下一份可以给你做“函数级别逐段讲解版（按执行顺序逐行解释）”，先从 `run_omnivla.py` 开始。
