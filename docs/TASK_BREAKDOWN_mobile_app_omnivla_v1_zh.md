# 开发任务拆解：手机 APP 对接 OmniVLA 实时轨迹（v1）

## 1. 范围与冻结前提
- 方案：Android Kotlin + WebSocket + 单会话 + 内网单机 GPU
- 输出：`waypoints + linear_vel + angular_vel`
- 目标：`p95 total_ms <= 450ms`

## 2. 角色与分工
- **后端工程师**：推理服务、协议、会话、日志与指标
- **安卓工程师**：采集、压缩、WebSocket、轨迹展示
- **测试工程师**：联调用例、稳定性与性能验证
- **算法工程师**：推理路径一致性与输出结果校验

## 3. 任务分解（A/B/C 按依赖顺序）

### A. 任务排期与文档冻结（0.5 天）
1. 确认 PRD/TDD/API 文档一致性
2. 冻结 v1 字段与错误码
3. 明确联调场景（1 组固定 current/goal + 指令）

**交付物**
- `PRD/TDD/API_SPEC` 最终版
- 本任务拆解文档

### B. 服务端骨架实现（2~3 天）

#### B1. 工程骨架（0.5 天）
1. 新建 `server/` 目录与模块骨架
2. FastAPI 应用入口、WebSocket 路由
3. 配置管理（host/port/device/model_path）

#### B2. 推理引擎常驻化（0.5~1 天）
1. 抽象 `OmniVLAEngine`（单例常驻）
2. 启动时加载模型，失败可观测
3. 禁止重复加载，提供 `ready` 状态

#### B3. 协议与会话（0.5 天）
1. `start_session/frame/stop_session` 校验
2. 单会话 `SessionManager`
3. 标准 `ack/error/trajectory` 回包

#### B4. 背压与时延统计（0.5 天）
1. `FrameBuffer(maxsize=1)` 丢旧保新
2. 每帧统计 `queue_wait/infer/post/total`
3. 输出结构化日志（session_id + seq）

#### B5. 基础测试（0.5 天）
1. 关键模块单元测试（schema/session/buffer）
2. 本地脚本回放帧联调
3. 输出性能报告（p50/p95）

**交付物**
- `server/*.py` 基础可运行版本
- 内网联调可用的 WS 服务
- 服务端联调说明（启动命令 + 示例 payload）

### C. Android Kotlin 客户端 MVP（2~3 天）

#### C1. 采集与抽帧（0.5~1 天）
1. CameraX 采集画面
2. 抽帧（3~5 FPS）
3. JPEG 压缩（质量 65~80）

#### C2. WebSocket 客户端（0.5~1 天）
1. 建立连接与重连机制
2. 发送 `start_session/frame/stop_session`
3. 接收 `trajectory/error` 并做 seq 对齐

#### C3. 结果展示（0.5 天）
1. 显示 `linear_vel/angular_vel`
2. 绘制简化轨迹（2D Overlay）
3. 展示端到端时延

#### C4. 联调与稳定性（0.5 天）
1. 连续 5 分钟压测
2. 弱网/丢包下的重连验证
3. 异常输入（坏图、空会话）验证

**交付物**
- Android MVP（可连服务端实时显示轨迹）
- 联调录像与测试报告

## 4. 并行策略
- 后端 B1/B2 完成后，安卓即可基于 mock 协议先做 C1/C2
- API 字段冻结后，B3 与 C2 并行
- B4 与 C3 并行，最后合流压测

## 5. 风险与缓解
- **加载慢导致首帧等待长**：启动时预热并前端显示“模型加载中”
- **帧堆积**：严格 `maxsize=1`，客户端动态降 FPS
- **端到端超时**：降低分辨率与 JPEG 质量，关闭非必要可视化
- **结果抖动**：客户端只渲染最新 `seq`，丢弃过期返回

## 6. 验收清单（DoD）
1. 服务端模型只加载一次（日志可证明）
2. 单会话 3~5 FPS 连续 5 分钟无崩溃
3. Android 端稳定显示速度与轨迹
4. 错误码全链路可验证（至少 5 类）
5. 性能报告包含 `p50/p95 total_ms`

## 7. 建议日程（5~6 天）
- Day 1：A + B1
- Day 2：B2 + B3
- Day 3：B4 + B5
- Day 4：C1 + C2
- Day 5：C3 + C4
- Day 6（可选）：性能优化与文档收敛
