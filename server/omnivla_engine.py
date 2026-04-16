from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import utm
from PIL import Image

from inference import run_omnivla as rv


class OmniVLAEngine:
    """
    OmniVLA 常驻推理引擎（v1）。
    - 进程启动后仅加载一次模型
    - 单实例串行 infer，避免并发抢占显存
    """

    def __init__(
        self,
        model_path: str = "./omnivla-original",
        device: str = "cuda:0",
        save_dir: str = "./inference",
        enable_visualization: bool = False,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.save_dir = save_dir
        self.enable_visualization = enable_visualization

        self.metric_waypoint_spacing = 0.1
        self._default_instruction = "move toward blue trash bin"

        # 与 run_omnivla.py 现有示例保持一致的目标定义（后续可改为会话级可配置）
        goal_lat, goal_lon, goal_compass_deg = 37.8738930785863, -122.26746181032362, 0.0
        self.goal_utm = utm.from_latlon(goal_lat, goal_lon)
        self.goal_compass = -float(goal_compass_deg) / 180.0 * math.pi

        self.ready = False
        self._counter = 0
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def load(self) -> None:
        if self.ready:
            return
        with self._load_lock:
            if self.ready:
                return

            # v1 先锁定为 image-goal 路径；instruction 仅写入 prompt 文本
            rv.pose_goal = False
            rv.satellite = False
            rv.image_goal = True
            rv.lan_prompt = False

            cfg = rv.InferenceConfig()
            cfg.vla_path = self.model_path

            target_device: Optional[torch.device]
            if self.device:
                if self.device.startswith("cuda") and not torch.cuda.is_available():
                    print("[OmniVLAEngine] CUDA 不可用，自动回退到 CPU。")
                    target_device = torch.device("cpu")
                else:
                    target_device = torch.device(self.device)
            else:
                target_device = None

            (
                self.vla,
                self.action_head,
                self.pose_projector,
                self.device_id,
                self.num_patches,
                self.action_tokenizer,
                self.processor,
            ) = rv.define_model(cfg, device=target_device)

            # run_omnivla.Inference.run_forward_pass 中会使用这些全局变量
            rv.vla = self.vla
            rv.action_head = self.action_head
            rv.pose_projector = self.pose_projector
            rv.device_id = self.device_id
            rv.NUM_PATCHES = self.num_patches

            self.ready = True

    def _compute_goal_pose_loc_norm(self) -> np.ndarray:
        thres_dist = 30.0
        current_lat = 37.87371258374039
        current_lon = -122.26729417226024
        current_compass = 270.0
        cur_utm = utm.from_latlon(current_lat, current_lon)
        cur_compass = -float(current_compass) / 180.0 * math.pi

        delta_x, delta_y = self.goal_utm[0] - cur_utm[0], self.goal_utm[1] - cur_utm[1]
        relative_x = delta_x * math.cos(cur_compass) + delta_y * math.sin(cur_compass)
        relative_y = -delta_x * math.sin(cur_compass) + delta_y * math.cos(cur_compass)
        radius = float(np.sqrt(relative_x**2 + relative_y**2))
        if radius > thres_dist:
            scale = thres_dist / radius
            relative_x *= scale
            relative_y *= scale

        return np.array(
            [
                relative_y / self.metric_waypoint_spacing,
                -relative_x / self.metric_waypoint_spacing,
                np.cos(self.goal_compass - cur_compass),
                np.sin(self.goal_compass - cur_compass),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _compute_control(waypoints: np.ndarray, metric_waypoint_spacing: float) -> tuple[float, float]:
        waypoint_select = min(4, waypoints.shape[0] - 1)
        chosen_waypoint = waypoints[waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        eps = 1e-8
        dt = 1.0 / 3.0
        if np.abs(dx) < eps and np.abs(dy) < eps:
            linear_vel = 0.0
            angular_vel = rv.clip_angle(float(np.arctan2(hy, hx))) / dt
        elif np.abs(dx) < eps:
            linear_vel = 0.0
            angular_vel = float(np.sign(dy) * np.pi / (2.0 * dt))
        else:
            linear_vel = float(dx / dt)
            angular_vel = float(np.arctan(dy / dx) / dt)

        linear_vel = float(np.clip(linear_vel, 0.0, 0.5))
        angular_vel = float(np.clip(angular_vel, -1.0, 1.0))

        maxv, maxw = 0.3, 0.3
        if np.abs(linear_vel) <= maxv:
            if np.abs(angular_vel) <= maxw:
                linear_limit = linear_vel
                angular_limit = angular_vel
            else:
                rd = linear_vel / angular_vel
                linear_limit = float(maxw * np.sign(linear_vel) * np.abs(rd))
                angular_limit = float(maxw * np.sign(angular_vel))
        else:
            if np.abs(angular_vel) <= 0.001:
                linear_limit = float(maxv * np.sign(linear_vel))
                angular_limit = 0.0
            else:
                rd = linear_vel / angular_vel
                if np.abs(rd) >= maxv / maxw:
                    linear_limit = float(maxv * np.sign(linear_vel))
                    angular_limit = float(maxv * np.sign(angular_vel) / np.abs(rd))
                else:
                    linear_limit = float(maxw * np.sign(linear_vel) * np.abs(rd))
                    angular_limit = float(maxw * np.sign(angular_vel))
        return linear_limit, angular_limit

    def infer(
        self,
        current_image_pil: Image.Image,
        goal_image_pil: Image.Image,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.ready:
            raise RuntimeError("model not ready")

        with self._infer_lock:
            inference = rv.Inference(
                save_dir=self.save_dir,
                lan_inst_prompt=instruction or self._default_instruction,
                goal_utm=self.goal_utm,
                goal_compass=self.goal_compass,
                goal_image_PIL=goal_image_pil,
                action_tokenizer=self.action_tokenizer,
                processor=self.processor,
                current_image_path=None,
                inject_instruction_in_prompt=instruction is not None,
            )

            goal_pose_loc_norm = self._compute_goal_pose_loc_norm()
            lan_inst = (
                inference.lan_inst_prompt if (rv.lan_prompt or inference.inject_instruction_in_prompt) else "xxxx"
            )
            batch = inference.data_transformer_omnivla(
                current_image_pil,
                lan_inst,
                goal_image_pil,
                goal_pose_loc_norm,
                prompt_builder=rv.PurePromptBuilder,
                action_tokenizer=self.action_tokenizer,
                processor=self.processor,
            )

            t0 = time.perf_counter()
            actions, modality_id = inference.run_forward_pass(
                vla=self.vla.eval(),
                action_head=self.action_head.eval(),
                noisy_action_projector=None,
                pose_projector=self.pose_projector.eval(),
                batch=batch,
                action_tokenizer=self.action_tokenizer,
                device_id=self.device_id,
                use_l1_regression=True,
                use_diffusion=False,
                use_film=False,
                num_patches=self.num_patches,
                compute_diffusion_l1=False,
                num_diffusion_steps_train=None,
                mode="vali",
                idrun=self._counter,
            )
            infer_ms = (time.perf_counter() - t0) * 1000.0

            waypoints = actions.float().cpu().numpy()[0]
            linear_vel, angular_vel = self._compute_control(waypoints, self.metric_waypoint_spacing)
            modality_id_arr = modality_id.cpu().numpy()

            if self.enable_visualization:
                inference.count_id = self._counter
                inference.save_robot_behavior(
                    current_image_pil,
                    goal_image_pil,
                    goal_pose_loc_norm,
                    waypoints,
                    linear_vel,
                    angular_vel,
                    self.metric_waypoint_spacing,
                    modality_id_arr,
                )
            self._counter += 1

            return {
                "waypoints": [[float(v) for v in row] for row in waypoints.tolist()],
                "linear_vel": float(linear_vel),
                "angular_vel": float(angular_vel),
                "modality_id": int(modality_id_arr[0]),
                "infer_ms": int(round(infer_ms)),
            }
