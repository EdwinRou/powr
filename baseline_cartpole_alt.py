import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
from pathlib import Path


class TrainingConfig:
    def __init__(self):
        self.config = {
            "env_name": "CartPole-v1",
            "batch_size": 64,
            "buffer_size": 100_000,
            "exploration_final_eps": 0.04,
            "exploration_fraction": 0.16,
            "gamma": 0.99,
            "gradient_steps": 128,
            "learning_rate": 0.0023,
            "learning_starts": 1_000,
            "total_timesteps": 50_000,
            "policy": "MlpPolicy",
            "policy_kwargs": dict(net_arch=[256, 256]),
            "target_update_interval": 10,
            "train_freq": 256,
            "normalize": False,
            "eval_freq": 1000,
            "n_eval_episodes": 10,
            "save_freq": 5000,
        }

    @property
    def run_name(self):
        return f"{self.config['env_name']}_bs{self.config['batch_size']}_lr{self.config['learning_rate']}"

    def initialize_wandb(self):
        return wandb.init(
            project="baseline_CartPole-v1_DQN",
            entity="edwinro-institut-polytechnique-de-paris",
            config=self.config,
            name=self.run_name,
        )


class WandbCallback(BaseCallback):
    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.current_reward = 0
        self.current_step = 0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            wandb.log({
                "train/episode_reward": self.current_reward,
                "train/steps_for_epoch": self.num_timesteps - self.current_step,
                "train/exploration_rate": self.model.exploration_rate,
                "timesteps": self.num_timesteps
            }, step=self.num_timesteps)

            self.current_step = self.num_timesteps
            self.current_reward = 0
        return True


class CartPoleEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        if not self.locals["dones"][0]:  # Only evaluate at episode end
            return True

        super()._on_step()
        return True


def create_env(config, render_mode=None):
    env = gym.make(config["env_name"], render_mode=render_mode)
    env = Monitor(env)
    return DummyVecEnv([lambda: env])


def test_policy(model, env, n_episodes=10):
    """Run testing episodes and return statistics"""
    episodes_rewards = []
    episodes_lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            episode_length += 1

        episodes_rewards.append(total_reward)
        episodes_lengths.append(episode_length)

    return {
        "mean_reward": np.mean(episodes_rewards),
        "std_reward": np.std(episodes_rewards)
    }


def create_evaluation_gif(model, env, gif_path, n_episodes=5):
    frames = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames.append(env.render())

    imageio.mimsave(gif_path, frames, fps=30)
    wandb.log({"eval/gif": wandb.Video(gif_path, fps=30, format="gif")})


def main():
    config = TrainingConfig()
    _ = config.initialize_wandb()

    # Create environments
    env = create_env(config.config)
    eval_env = create_env(config.config)
    test_env = create_env(config.config, render_mode="rgb_array")

    # Setup evaluation callback
    eval_callback = CartPoleEvalCallback(
        eval_env,
        eval_freq=1,  # Check every step, but will skip if epoch has no ended
        n_eval_episodes=config.config["n_eval_episodes"],
        deterministic=True,
        verbose=0
    )

    # Initialize model
    model = DQN(
        policy=config.config["policy"],
        env=env,
        **{k: v for k, v in config.config.items()
           if k not in ["env_name", "policy", "normalize", "eval_freq", "n_eval_episodes"]}
    )

    # Training
    model.learn(
        total_timesteps=config.config["total_timesteps"],
        callback=[WandbCallback(eval_env=eval_env), eval_callback]
    )

    # Final testing
    test_results = test_policy(model, test_env)
    wandb.log({"test/" + k: v for k, v in test_results.items()})
    create_evaluation_gif(model, test_env, "final_evaluation.gif")

    # Cleanup
    env.close()
    eval_env.close()
    test_env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
