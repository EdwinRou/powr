import gymnasium as gym
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Initialize Weights & Biases
wandb.init(
    project="MountainCar-v0_DQN",
    entity="edwinro-institut-polytechnique-de-paris",  # Replace with your wandb username
    config={
        "env_name": "MountainCar-v0",
        "learning_rate": 1e-3,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 32,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 500,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.02,
        "total_timesteps": 50000,
    },
)


# Custom Callback for wandb Logging
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional metrics to wandb
        wandb.log({
            "step": self.num_timesteps,
            "reward": self.locals.get("rewards", 0),
            "exploration_rate": self.model.exploration_rate,
        })
        return True

    def _on_training_end(self) -> None:
        # Log model at the end of training
        wandb.save("dqn_mountaincar.zip")


# Create and monitor the environment
env = gym.make("MountainCar-v0", render_mode="human")
env = Monitor(env)  # Monitor to log rewards and episodes
env = DummyVecEnv([lambda: env])  # Vectorized environment

# Create the DQN agent
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=wandb.config.learning_rate,
    buffer_size=wandb.config.buffer_size,
    learning_starts=wandb.config.learning_starts,
    batch_size=wandb.config.batch_size,
    gamma=wandb.config.gamma,
    train_freq=wandb.config.train_freq,
    target_update_interval=wandb.config.target_update_interval,
    exploration_fraction=wandb.config.exploration_fraction,
    exploration_final_eps=wandb.config.exploration_final_eps,
    verbose=1,
)

# Train the agent with wandb callback
model.learn(
    total_timesteps=wandb.config.total_timesteps,
    callback=WandbCallback(),
)

# Save the model
model.save("dqn_mountaincar")
wandb.save("dqn_mountaincar.zip")

# Test the trained agent
obs = env.reset()
for _ in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break

env.close()

# End the wandb run
wandb.finish()
