import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import imageio

config = {
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
}

# Create a smart name for the run using key parameters
run_name = f"{config['env_name']}_bs{config['batch_size']}_lr{config['learning_rate']}_arch{config['policy_kwargs']}"

# Initialize Weights & Biases
wandb.init(
    project="baseline_CartPole-v1_DQN",
    entity="edwinro-institut-polytechnique-de-paris",
    config=config,
    name=run_name,
)


# Evaluation function
def evaluate_policy(model, env, n_episodes=10):
    all_rewards = []
    steps_epochs = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0
        number_step = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            number_step += 1
        all_rewards.append(total_reward)
        steps_epochs.append(number_step)
    return np.mean(all_rewards), np.std(all_rewards), np.mean(steps_epochs)


# Custom Callback for wandb Logging
class WandbCallback(BaseCallback):
    def __init__(self, eval_env, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.current_reward = 0
        self.current_step = 0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            # Log per-episode reward
            wandb.log({
                "train/episode_reward": self.current_reward,
                "train/steps_for_epoch": self.num_timesteps - self.current_step,
                "train/exploration_rate": self.model.exploration_rate,
            })

            # Perform evaluation after each episode
            mean_reward, std_reward, mean_steps_for_epoch = evaluate_policy(self.model, self.eval_env)
            wandb.log({
                "eval/mean_reward": mean_reward,
                "eval/std_reward": std_reward,
                "eval/mean_steps_for_epoch": mean_steps_for_epoch,
            })

            self.current_step = self.num_timesteps
            self.current_reward = 0
        return True


def test_policy_with_gif(model, env, n_episodes=7, gif_name="test_env.gif"):
    """
    Test a trained RL model, log results, and create a GIF.

    :param model: The trained RL model.
    :param env: The environment to test on.
    :param n_episodes: Number of episodes to test.
    :param gif_name: Name of the GIF file to save.
    """
    episodes_rewards = []
    frames = []  # For GIF
    for episode in range(n_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Collect frames for the GIF
            frame = env.render()
            frames.append(frame)

        episodes_rewards.append(total_reward)

    # Save and log GIF to wandb
    imageio.mimsave(gif_name, frames, fps=30)
    wandb.log({"test/episode_gif": wandb.Video(gif_name, fps=30, format="gif")})

    print("Results of testing:")
    print(f"Mean reward across {n_episodes} episodes: {np.mean(episodes_rewards)}")
    print(f"Std reward across {n_episodes} episodes: {np.std(episodes_rewards)}")


# Create and monitor the environment
env = gym.make("CartPole-v1", render_mode=None)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

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
    callback=WandbCallback(eval_env=gym.make("CartPole-v1")),
)

# Save the model
# model.save("dqn_cartpole")
# wandb.save("dqn_cartpole.zip")

# Test the trained agent and log results
test_env = gym.make("CartPole-v1", render_mode="rgb_array")
test_policy_with_gif(model, test_env, n_episodes=10, gif_name="cartpole_test.gif")

# Close environments
env.close()
test_env.close()

# End the wandb run
wandb.finish()
