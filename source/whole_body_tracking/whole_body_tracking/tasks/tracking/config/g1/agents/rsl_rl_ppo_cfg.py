from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    staged_training = {
        "enabled": False,
        "stage1": {
            "run_name_suffix": "_stage1",
            "max_iterations": 8000,
            "enable_amp": True,
            "enable_distill": False,
            "motion_selector": "uniform",
            "motion_files": [
                "source/motion/unitree_g1/npz/EKUT/265/SLP105_stageii.npz",
            ],
        },
        "stage2": {
            "run_name_suffix": "_stage2",
            "max_iterations": 15000,
            "enable_amp": False,
            "enable_distill": True,
            "teacher_source": "stage1",
            "teacher_checkpoint": None,
            "distill_action_coef": 1.0,
            "distill_feature_coef": 0.5,
            "motion_selector": "curriculum",
            "motion_files": [
                "source/motion/unitree_g1/npz/EKUT/265/SLP105_stageii.npz",
                "source/motion/unitree_g1/npz/EKUT/265/WSTR03_stageii.npz",
                "source/motion/unitree_g1/npz/EKUT/265/WSTR05_stageii.npz",
                "source/motion/unitree_g1/npz/EKUT/265/SLP204_stageii.npz",
                "source/motion/unitree_g1/npz/EKUT/265/BEAMN04_stageii.npz",
            ],
        },
    }


@configclass
class G1MultiMotionPPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_multimotion"


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)


@configclass
class G1FlatStageDistillPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.staged_training = dict(self.staged_training)
        self.staged_training["enabled"] = True
        self.run_name = (self.run_name or "") + "_staged"
