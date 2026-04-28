#!/usr/bin/env bash
# 本地单次推理：直接运行 inference/run_omnivla.py，加载模型、推理一轮、写可视化后进程结束。
# 需要「服务常驻、客户端发一次回一次」请用 scripts/start_omnivla_server.sh + tools/ws_replay_client.py 或 App。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 允许通过环境变量覆盖 conda 环境名
CONDA_ENV="${OMNIVLA_CONDA_ENV:-omnivla}"
DEFAULT_CURRENT="${OMNIVLA_EXAMPLE_CURRENT:-send/20260414-192532_current.jpg}"
DEFAULT_GOAL="${OMNIVLA_EXAMPLE_GOAL:-example/20260414-192532_goal.jpg}"
DEFAULT_SAVE_DIR="${OMNIVLA_EXAMPLE_SAVE_DIR:-example}"
DEFAULT_INSTRUCTION="${OMNIVLA_EXAMPLE_INSTRUCTION:-沿走廊前进，绕开右侧障碍物并接近目标区域}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[run_omnivla_once] ERROR: conda command not found"
  exit 1
fi

cd "${REPO_ROOT}"

if [ "$#" -eq 0 ]; then
  if [ ! -f "${DEFAULT_CURRENT}" ] || [ ! -f "${DEFAULT_GOAL}" ]; then
    echo "[run_omnivla_once] ERROR: 默认 example 图片不存在，请手动传入 --current/--goal"
    echo "[run_omnivla_once] default_current=${DEFAULT_CURRENT}"
    echo "[run_omnivla_once] default_goal=${DEFAULT_GOAL}"
    exit 1
  fi
  ARGS=(
    --current "${DEFAULT_CURRENT}"
    --goal "${DEFAULT_GOAL}"
    -i "${DEFAULT_INSTRUCTION}"
    --save-dir "${DEFAULT_SAVE_DIR}"
  )
  echo "[run_omnivla_once] 使用默认 example 参数（未传命令行参数）"
else
  ARGS=("$@")
fi

echo "[run_omnivla_once] repo=${REPO_ROOT}"
echo "[run_omnivla_once] conda_env=${CONDA_ENV}"
echo "[run_omnivla_once] command: inference/run_omnivla.py ${ARGS[*]}"

# 清理库路径，避免系统 CUDA/cuDNN 与 PyTorch wheel 冲突
exec env -u LD_LIBRARY_PATH -u DYLD_LIBRARY_PATH \
  conda run -n "${CONDA_ENV}" python -u inference/run_omnivla.py "${ARGS[@]}"
