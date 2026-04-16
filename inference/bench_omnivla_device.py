#!/usr/bin/env python3
"""
OmniVLA 推理耗时基准：模型加载、数据打包、单次前向（与 run_omnivla 一致的路径）。

在仓库根目录执行（与 run_omnivla.py 相同的工作目录约定）::

    conda run -n omnivla python inference/bench_omnivla_device.py --device cpu
    conda run -n omnivla python inference/bench_omnivla_device.py --device cuda
    conda run -n omnivla python inference/bench_omnivla_device.py --both

--both 会依次起两个子进程（CPU 子进程会设置 CUDA_VISIBLE_DEVICES= 以屏蔽 GPU）。

说明：若当前 PyTorch 为 CPU 轮子（如 torch-*+cpu），--device cuda 会直接跳过并提示。
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _prepare_batch(ro, inference):
    """与 run_omnivla.Inference.run_omnivla 中几何与 batch 构造一致（不含前向与控制律）。"""
    import numpy as np
    from PIL import Image

    thres_dist = 30.0
    metric_waypoint_spacing = 0.1

    current_lat = 37.87371258374039
    current_lon = -122.26729417226024
    current_compass = 270.0
    cur_utm = ro.utm.from_latlon(current_lat, current_lon)
    cur_compass = -float(current_compass) / 180.0 * math.pi

    delta_x, delta_y = inference.calculate_relative_position(
        cur_utm[0], cur_utm[1], inference.goal_utm[0], inference.goal_utm[1]
    )
    relative_x, relative_y = inference.rotate_to_local_frame(delta_x, delta_y, cur_compass)
    radius = float(np.sqrt(relative_x**2 + relative_y**2))
    if radius > thres_dist:
        relative_x *= thres_dist / radius
        relative_y *= thres_dist / radius

    goal_pose_loc_norm = np.array(
        [
            relative_y / metric_waypoint_spacing,
            -relative_x / metric_waypoint_spacing,
            np.cos(inference.goal_compass - cur_compass),
            np.sin(inference.goal_compass - cur_compass),
        ]
    )

    current_image_path = os.path.join(_repo_root(), "inference", "current_img.jpg")
    current_image_PIL = Image.open(current_image_path).convert("RGB")

    lan_inst = inference.lan_inst_prompt if ro.lan_prompt else "xxxx"

    return inference.data_transformer_omnivla(
        current_image_PIL,
        lan_inst,
        inference.goal_image_PIL,
        goal_pose_loc_norm,
        prompt_builder=ro.PurePromptBuilder,
        action_tokenizer=inference.action_tokenizer,
        processor=inference.processor,
    )


def run_single_benchmark(device_name: str, warmup: int, repeats: int) -> int:
    # 若 LD_LIBRARY_PATH 优先加载了系统里的 libcudnn，可能与 PyTorch cu121 轮子内置版本混用，
    # 导致 libcudnn_cnn_infer 报 undefined symbol（与 libcudnn_ops_infer 不匹配）。
    if device_name == "cuda":
        os.environ.pop("LD_LIBRARY_PATH", None)
        os.environ.pop("DYLD_LIBRARY_PATH", None)

    import torch

    repo = _repo_root()
    os.chdir(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import inference.run_omnivla as ro

    if device_name == "cuda" and not torch.cuda.is_available():
        print("[bench] skip cuda: torch.cuda.is_available() is False（常见原因：未装 CUDA 版 PyTorch 或无 GPU 驱动）")
        return 0

    ro.pose_goal = False
    ro.satellite = False
    ro.image_goal = True
    ro.lan_prompt = False

    goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0
    goal_utm = ro.utm.from_latlon(goal_lat, goal_lon)
    goal_compass = -float(goal_compass) / 180.0 * math.pi

    goal_path = os.path.join(repo, "inference", "goal_img.jpg")
    from PIL import Image

    goal_image_PIL = Image.open(goal_path).convert("RGB")

    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    print(f"[bench] device={device} warmup={warmup} repeats={repeats}")

    t0 = time.perf_counter()
    cfg = ro.InferenceConfig()
    vla, action_head, pose_projector, device_id, num_patches, action_tokenizer, processor = ro.define_model(
        cfg, device=device
    )
    t_load = time.perf_counter() - t0
    print(f"[bench] load_model_sec={t_load:.3f}")

    ro.vla = vla
    ro.action_head = action_head
    ro.pose_projector = pose_projector
    ro.device_id = device_id
    ro.NUM_PATCHES = num_patches

    inference = ro.Inference(
        save_dir=os.path.join(repo, "inference"),
        lan_inst_prompt="move toward blue trash bin",
        goal_utm=goal_utm,
        goal_compass=goal_compass,
        goal_image_PIL=goal_image_PIL,
        action_tokenizer=action_tokenizer,
        processor=processor,
    )

    t1 = time.perf_counter()
    batch = _prepare_batch(ro, inference)
    t_data = time.perf_counter() - t1
    print(f"[bench] build_batch_sec={t_data:.4f}")

    def one_forward():
        if device_id.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        inference.run_forward_pass(
            vla=vla.eval(),
            action_head=action_head.eval(),
            noisy_action_projector=None,
            pose_projector=pose_projector.eval(),
            batch=batch,
            action_tokenizer=action_tokenizer,
            device_id=device_id,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_patches=num_patches,
            compute_diffusion_l1=False,
            num_diffusion_steps_train=None,
            mode="vali",
            idrun=0,
        )
        if device_id.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t

    for _ in range(warmup):
        one_forward()

    times = [one_forward() for _ in range(repeats)]
    mean_f = sum(times) / len(times)
    print(f"[bench] forward_sec_each={[round(x, 3) for x in times]} mean={mean_f:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], help="单设备模式")
    parser.add_argument("--both", action="store_true", help="依次跑 CPU 与 CUDA（子进程）")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    # 必须在「尚未 import torch」前清掉 LD_LIBRARY_PATH，否则动态链接器可能先加载
    # /usr/local/cuda 下的旧 libcudnn，与 PyTorch 自带的 cuDNN 8.9 混用触发符号错误。
    if args.device == "cuda" and (
        os.environ.get("LD_LIBRARY_PATH") or os.environ.get("DYLD_LIBRARY_PATH")
    ):
        env = os.environ.copy()
        env.pop("LD_LIBRARY_PATH", None)
        env.pop("DYLD_LIBRARY_PATH", None)
        argv = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
        os.execve(sys.executable, argv, env)

    if args.both:
        repo = _repo_root()
        script = os.path.join(repo, "inference", "bench_omnivla_device.py")
        py = sys.executable
        env_base = os.environ.copy()

        print("======== CPU (CUDA_VISIBLE_DEVICES=) ========")
        env_cpu = env_base.copy()
        env_cpu["CUDA_VISIBLE_DEVICES"] = ""
        r_cpu = subprocess.run(
            [py, script, "--device", "cpu", "--warmup", str(args.warmup), "--repeats", str(args.repeats)],
            cwd=repo,
            env=env_cpu,
        )

        print("======== CUDA（去掉 LD_LIBRARY_PATH，避免系统 cuDNN 冲突）========")
        env_cuda = env_base.copy()
        env_cuda.pop("LD_LIBRARY_PATH", None)
        env_cuda.pop("DYLD_LIBRARY_PATH", None)
        r_cuda = subprocess.run(
            [py, script, "--device", "cuda", "--warmup", str(args.warmup), "--repeats", str(args.repeats)],
            cwd=repo,
            env=env_cuda,
        )
        return max(r_cpu.returncode, r_cuda.returncode)

    if not args.device:
        parser.error("请指定 --device 或 --both")
    return run_single_benchmark(args.device, args.warmup, args.repeats)


if __name__ == "__main__":
    raise SystemExit(main())
