import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


config = {
    "env_name": "CartPole-v1",
    "batch_size": 64,
    "buffer_size": 100000,
    "exploration_final_eps": 0.04,
    "exploration_fraction": 0.16,
    "gamma": 0.99,
    "gradient_steps": 128,
    "learning_rate": 0.0023,
    "learning_starts": 1000,
    "total_timesteps": 50000,
    "policy": "MlpPolicy",
    "policy_kwargs": dict(net_arch=[256, 256]),
    "target_update_interval": 10,
    "train_freq": 256,
    "normalize": False,
}


# Create a smart name for the run using key parameters
run_name = f"{config['env_name']}_bs{config['batch_size']}_lr{config['learning_rate']}_arch{config['policy_kwargs']}"


# Initialize Weights & Biases
wandb.init(
    project="baseline_CartPole-v1_DQN",
    entity="edwinro-institut-polytechnique-de-paris",
    config=config,
    name=run_name,  # Use the dynamically generated name
)


# Evaluation function
def evaluate_policy(model, env, n_episodes=10):
    """
    Evaluate a model on a given environment.

    :param model: Trained RL model
    :param env: Gym environment
    :param n_episodes: Number of episodes for evaluation
    :return: Mean and standard deviation of rewards
    """
    all_rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()  # Include the info returned by reset
        terminated, truncated = False, False
        total_reward = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        all_rewards.append(total_reward)
    return np.mean(all_rewards), np.std(all_rewards)


# Custom Callback for wandb Logging
class WandbCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.current_reward = 0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            # Log per-episode reward
            self.episode_rewards.append(self.current_reward)
            wandb.log({
                "train/episode_reward": self.current_reward,
                "train/step": self.num_timesteps,
                "train/exploration_rate": self.model.exploration_rate,
            })
            self.current_reward = 0

        # Perform evaluation periodically
        if self.num_timesteps % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env)
            wandb.log({
                "eval/mean_reward": mean_reward,
                "eval/std_reward": std_reward,
                "eval/step": self.num_timesteps,
            })
        return True


# Create and monitor the environment
env = gym.make("CartPole-v1", render_mode=None)  # Using CartPole-v1
env = Monitor(env)  # Monitor to log rewards and episodes
env = DummyVecEnv([lambda: env])  # Vectorized environment

# Create the DQN agent
model = DQN(
    policy=wandb.config.policy,
    env=env,
    learning_rate=wandb.config.learning_rate,
    buffer_size=wandb.config.buffer_size,
    learning_starts=wandb.config.learning_starts,
    batch_size=wandb.config.batch_size,
    gamma=wandb.config.gamma,
    train_freq=wandb.config.train_freq,
    target_update_interval=wandb.config.target_update_interval,
    exploration_fraction=wandb.config.exploration_fraction,
    exploration_final_eps=wandb.config.exploration_final_eps,
    gradient_steps=wandb.config.gradient_steps,
    policy_kwargs=wandb.config.policy_kwargs,
    verbose=1,
)


# Train the agent with wandb callback
model.learn(
    total_timesteps=wandb.config.total_timesteps,
    callback=WandbCallback(eval_env=gym.make("CartPole-v1"), eval_freq=5000),
)

# Save the model
# model.save("dqn_cartpole")
# wandb.save("dqn_cartpole.zip")

# Test the trained agent
obs, _ = env.reset()  # Include the info returned by reset
for _ in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    # env.render()
    if dones:
        break

env.close()

# End the wandb run
wandb.finish()
