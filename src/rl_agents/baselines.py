from stable_baselines3 import DQN, PPO
import os
from src.utils.logger import setup_logger

logger = setup_logger("baselines")

class BaselineAgents:
    """
    Wrapper for baseline RL agents (DQN, PPO).
    """
    def __init__(self, env, agent_type="DQN", device="cpu"):
        self.env = env
        self.agent_type = agent_type
        self.device = device
        
        if agent_type == "DQN":
            self.model = DQN("MlpPolicy", env, verbose=1, device=device)
        elif agent_type == "PPO":
            self.model = PPO("MlpPolicy", env, verbose=1, device=device)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def train(self, total_timesteps=10000):
        logger.info(f"Training baseline {self.agent_type} for {total_timesteps} steps...")
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"{self.agent_type} model saved to {path}")

    def load(self, path):
        if self.agent_type == "DQN":
            self.model = DQN.load(path, env=self.env)
        elif self.agent_type == "PPO":
            self.model = PPO.load(path, env=self.env)
        logger.info(f"{self.agent_type} model loaded from {path}")
