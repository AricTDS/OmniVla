#!/usr/bin/env bash
# 启动 WebSocket 服务（uvicorn），进程常驻：加载模型后一直监听，客户端每发一帧/一次会话则处理并返回。
# 与 scripts/run_omnivla_once.sh 不同：后者是本地单次跑 inference/run_omnivla.py，跑完即退出。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${OMNIVLA_CONDA_ENV:-omnivla}"
HOST="${OMNIVLA_HOST:-0.0.0.0}"
PORT="${OMNIVLA_PORT:-8000}"
EXAMPLE_GOAL="${OMNIVLA_EXAMPLE_GOAL:-example/20260414-192532_goal.jpg}"
EXAMPLE_CURRENT="${OMNIVLA_EXAMPLE_CURRENT:-example/20260414-192532_current.jpg}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[start_omnivla_server] ERROR: conda command not found"
  exit 1
fi

# 供默认相对路径 ./logs/... 使用；具体日志文件名由 server.settings 决定（带时间戳见 default_log_file_path）
mkdir -p "${REPO_ROOT}/logs"

cd "${REPO_ROOT}"

echo "[start_omnivla_server] repo=${REPO_ROOT}"
echo "[start_omnivla_server] conda_env=${CONDA_ENV}"
echo "[start_omnivla_server] 本进程将保持运行，等待 WebSocket 客户端；停止请 Ctrl+C"
echo "[start_omnivla_server] ws=ws://${HOST}:${PORT}/ws/v1/omnivla"
echo "[start_omnivla_server] log: 未设置 OMNIVLA_LOG_FILE 时由 server.settings.default_log_file_path() 生成 ./logs/omnivla_server_<时间戳>.log"
echo "[start_omnivla_server] example_replay_cmd: conda run -n ${CONDA_ENV} python -u tools/ws_replay_client.py --ws-url ws://${HOST}:${PORT}/ws/v1/omnivla --session-id sess_example --goal ${EXAMPLE_GOAL} --frames ${EXAMPLE_CURRENT} --instruction \"沿走廊前进\" --repeat 1"

# 清理库路径，避免系统 CUDA/cuDNN 与 PyTorch wheel 冲突
exec env -u LD_LIBRARY_PATH -u DYLD_LIBRARY_PATH \
  conda run -n "${CONDA_ENV}" uvicorn server.app:app --host "${HOST}" --port "${PORT}" "$@"
