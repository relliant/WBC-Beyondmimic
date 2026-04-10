import os
from datetime import datetime

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _upgrade_legacy_train_cfg(env: VecEnv, train_cfg: dict) -> dict:
    """Adapt legacy IsaacLab policy config to the current rsl-rl runner schema."""
    actor_cfg = train_cfg.get("actor")
    critic_cfg = train_cfg.get("critic")
    needs_actor_upgrade = not isinstance(actor_cfg, dict) or "class_name" not in actor_cfg
    needs_critic_upgrade = not isinstance(critic_cfg, dict) or "class_name" not in critic_cfg

    if (needs_actor_upgrade or needs_critic_upgrade) and "policy" in train_cfg:
        policy_cfg = dict(train_cfg["policy"])
        empirical_norm = bool(train_cfg.get("empirical_normalization", False))
        distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": policy_cfg.get("init_noise_std", 1.0),
            "std_type": policy_cfg.get("noise_std_type", "scalar"),
        }

        train_cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg["actor_hidden_dims"],
            "activation": policy_cfg["activation"],
            "obs_normalization": policy_cfg.get("actor_obs_normalization", empirical_norm),
            "distribution_cfg": distribution_cfg,
        }
        train_cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg["critic_hidden_dims"],
            "activation": policy_cfg["activation"],
            "obs_normalization": policy_cfg.get("critic_obs_normalization", empirical_norm),
        }

    algorithm_cfg = train_cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict) and "class_name" not in algorithm_cfg:
        algorithm_cfg["class_name"] = "PPO"

    obs_groups = train_cfg.get("obs_groups")
    if isinstance(obs_groups, dict) and "actor" not in obs_groups:
        train_cfg["obs_groups"] = {
            "actor": obs_groups.get("policy", ["policy"]),
            "critic": obs_groups.get("critic", obs_groups.get("policy", ["policy"])),
        }

    if not train_cfg.get("obs_groups"):
        observations = env.get_observations()
        groups = set(observations.keys()) if hasattr(observations, "keys") else set()
        critic_group = "critic" if "critic" in groups else "policy"
        train_cfg["obs_groups"] = {"actor": ["policy"], "critic": [critic_group]}

    return train_cfg


class MyOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, _upgrade_legacy_train_cfg(env, train_cfg), log_dir, device)

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        logger_type = getattr(self.logger, "logger_type", None)
        if logger_type == "wandb":
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        registry_name: str = None,
        stage_context: dict | None = None,
    ):
        super().__init__(env, _upgrade_legacy_train_cfg(env, train_cfg), log_dir, device)
        self.registry_name = registry_name
        self.stage_context = stage_context or {}
        self._stage_log_dir = log_dir

        stage_name = self.stage_context.get("stage_name")
        if stage_name:
            print(
                f"[STAGED] Runner initialized for {stage_name} "
                f"(amp={self.stage_context.get('enable_amp', False)}, "
                f"distill={self.stage_context.get('enable_distill', False)})"
            )

    def emit_stage_summary(self) -> None:
        """Write a lightweight stage summary to logs for experiment traceability."""
        if not self._stage_log_dir or not self.stage_context:
            return

        summary_path = os.path.join(self._stage_log_dir, "params", "stage_context.txt")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"timestamp={datetime.now().isoformat()}\n")
            for key, value in self.stage_context.items():
                f.write(f"{key}={value}\n")

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        logger_type = getattr(self.logger, "logger_type", None)
        if logger_type == "wandb":
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=getattr(self, "obs_normalizer", None), path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
