#from utils import make_minigrid_env, make_procgen_env
import torch
import os
import uuid
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, DummyVecEnv
# from stable_baselines3.common.callbacks import BaseCallback # No longer needed
from stable_baselines3.ppo.policies import MlpPolicy
import numpy as np
from stable_baselines3.ppo import CnnPolicy
from shimmy import GymV26CompatibilityV0
import wandb
import time # Import time for elapsed time calculation
# In experiment.py
from gymnasium.wrappers import EnvCompatibility,FrameStack, GrayScaleObservation, TransformObservation
import gymnasium as gym

def _missing_env_maker(name: str):
    def _fn(*_args, **_kwargs):
        raise NotImplementedError(
            f"`{name}` not available. If you expected it, check `utils.py` and your optional dependencies."
        )
    return _fn

# Import optional env makers safely.
# Important: do NOT use a single `from utils import ...` inside one try/except, because a missing symbol
# (e.g. `make_procgen_env`) would mask *all* other env makers (including DMLab/psychlab).
import utils as _utils
make_maze_env = getattr(_utils, "make_maze_env", _missing_env_maker("make_maze_env"))
make_miniworld_env = getattr(_utils, "make_miniworld_env", _missing_env_maker("make_miniworld_env"))
make_dmlab_env = getattr(_utils, "make_dmlab_env", _missing_env_maker("make_dmlab_env"))
make_procgen_env = getattr(_utils, "make_procgen_env", _missing_env_maker("make_procgen_env"))

# --- NEW: Function to be called directly by the trainer ---
def log_wandb_metrics(metrics, total_timesteps, n_updates): 
    """Logs a dictionary of metrics directly to W&B."""
    if metrics:
        # Use total_timesteps as the step index for the chart
        # We ignore n_updates, but the function signature must match the caller!
        wandb.log(metrics, step=total_timesteps)

class Experiment:
    def __init__(self, config, experiment_id=None):

        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if experiment_id is None:
            experiment_id = self._create_exp_id()
        else:
            experiment_id = experiment_id

        self.outpath = os.path.join('./experiments', config['model'], config['env'], experiment_id)
        os.makedirs(self.outpath, exist_ok=True)


    def _create_exp_id(self):
        return str(uuid.uuid4())

    def run(self, seed=None):
        start_time = time.time()
        wandb_run = wandb.init(
            project="shelm-psychlab-xlstm",
            config=self.config,
            name=f"SHELM-XLSTM-Standard-{self.config['env']}",
        )

        # --- NEW: W&B AUTOMATIC CHART SETUP ---
        # 1. Define the x-axis for your charts
        wandb.define_metric("time/total_timesteps")
        
        # 2. Define the training metrics group to force a chart panel
        # We explicitly link the train metrics to the total_timesteps step
        
        train_metrics = [
            "train/policy_gradient_loss", "train/value_loss", "train/loss", 
            "train/entropy_loss", "train/approx_kl", "train/explained_variance",
            "train/adv_mean", "train/adv_std", "train/clip_fraction", 
            "train/learning_rate", "train/ent_coef", "train/clip_range",
            "train/n_updates", "rollout/ep_rew_mean", "rollout/ep_len_mean"
        ]
        
        # This loop explicitly tells W&B about all your training metrics
        for metric in train_metrics:
            wandb.define_metric(metric, step_metric="time/total_timesteps", summary="min")
            
        # create training environments
        if 'RandomMaze' in self.config['env']:
            env = DummyVecEnv([make_maze_env() for _ in range(self.config['n_envs'])])
            env = VecMonitor(env)
        elif 'MiniGrid' in self.config['env']:
            env = DummyVecEnv([make_minigrid_env(self.config['env']) for _ in range(self.config['n_envs'])])
            env = VecNormalize(VecMonitor(env), norm_reward=True, norm_obs=False, clip_reward=1.)
        elif 'MiniWorld' in self.config['env']:
            env = DummyVecEnv([make_miniworld_env(self.config['env']) for _ in range(self.config['n_envs'])])
            env = VecNormalize(VecMonitor(env), norm_reward=True, norm_obs=False, clip_reward=1.)
        elif 'psychlab' in self.config['env']:
                    import gymnasium as gym
                    from shimmy import GymV21CompatibilityV0
                    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize, VecTransposeImage
                    import numpy as np

                    def make_raw_env():
                        raw_env = make_dmlab_env(self.config['env'])()
                        # Wrap it so it becomes a Gymnasium-compatible object
                        return GymV21CompatibilityV0(env=raw_env)

                    env = DummyVecEnv([make_raw_env for _ in range(self.config['n_envs'])])

                    # Use Gymnasium spaces to match the trainer's new imports
                    env.observation_space = gym.spaces.Box(
                        low=0, high=255, shape=(80, 80, 3), dtype=np.uint8
                    )
                    env.action_space = gym.spaces.Discrete(11)

                    env = VecMonitor(env)
                    # Transpose (80,80,3) -> (3,80,80)
                    env = VecTransposeImage(env)
                    # Normalize rewards for the xLSTM
                    env = VecNormalize(env, norm_reward=True, norm_obs=False, clip_reward=1.)
        else:
            # create procgen environment
            env = make_procgen_env(id=self.config['env'], num_envs=self.config['n_envs'], num_levels=0)

        assert self.config['model'] in ['SHELM', 'HELMv2', 'HELM', 'Impala-PPO', 'CNN-PPO'], \
            f"Model type {self.config['model']} not recognized!"

        if self.config['model'] == 'HELM':
            from trainers.helm_trainer import HELMPPO
            trainer = HELMPPO
        elif self.config['model'] == 'HELMv2':
            from trainers.helmv2_trainer import HELMv2PPO
            trainer = HELMv2PPO
        elif self.config['model'] == 'Impala-PPO':
            from trainers.lstm_trainer import LSTMPPO
            trainer = LSTMPPO
        elif self.config['model'] == 'SHELM':
            from trainers.shelm_trainer import SHELMPPO
            trainer = SHELMPPO
        else:
            from trainers.cnn_trainer import CNNPPO
            trainer = CNNPPO

        tb_log_base = os.path.join('./experiments', self.config['model'], self.config['env'])

        # --- IMPORTANT: Pass the W&B function to the trainer as a hook ---
        model = trainer(CnnPolicy, env, verbose=1, tensorboard_log=tb_log_base,
                        lr_decay=self.config['lr_decay'], ent_coef=self.config['ent_coef'],
                        ent_decay=self.config['ent_decay'], learning_rate=self.config['learning_rate'],
                        vf_coef=self.config['vf_coef'], n_epochs=int(self.config['n_epochs']),
                        ent_decay_factor=self.config['ent_decay_factor'], clip_range=self.config['clip_range'],
                        gamma=self.config['gamma'], gae_lambda=self.config['gae_lambda'],
                        n_steps=int(self.config['n_rollout_steps']), n_envs=int(self.config['n_envs']),
                        min_lr=self.config['min_lr'], min_ent_coef=self.config['min_ent_coef'],
                        start_fraction=self.config['start_fraction'], end_fraction=self.config['end_fraction'],
                        device=self.device, clip_decay=self.config['clip_decay'], config=self.config,
                        clip_range_vf=self.config['clip_range_vf'], seed=seed,
                        max_grad_norm=self.config['max_grad_norm'],
                        adv_norm=self.config.get('adv_norm', False),
                        save_ckpt=self.config.get('save_ckpt', False),
                        # --- NEW ARGUMENT ---
                        wandb_log_fn=log_wandb_metrics
                        )
        
        model.learn(
            total_timesteps=self.config['n_steps'], 
            eval_log_path=self.outpath,
        )

        env.close()

        wandb.finish()