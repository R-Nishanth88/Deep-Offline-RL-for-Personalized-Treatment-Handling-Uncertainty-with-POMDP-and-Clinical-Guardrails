import os
import torch
import numpy as np
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.preprocessing.process_data import MIMICPreprocessor
from src.temporal_model.lstm_encoder import TemporalEncoder, TemporalTrainer
from src.latent_model.vae import LatentVAE, VAETrainer
from src.environment.icu_env import ICUEnv
from src.rl_agents.cql_agent import CQLAgent
from src.rl_agents.baselines import BaselineAgents

logger = setup_logger("training_pipeline")

def run_pipeline():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Step 1: Preprocessing
    logger.info("--- Step 1: Preprocessing ---")
    preprocessor = MIMICPreprocessor(raw_path="./data/raw", processed_path="./data/processed")
    preprocessor.process_trajectories()

    # Step 2: Train Temporal & Latent Models (Simplified training logic)
    logger.info("--- Step 2: Training Temporal & Latent Models ---")
    # input_dim=16 (7 vitals + 9 labs)
    temporal_model = TemporalEncoder(input_dim=16, hidden_dim=64, latent_dim=32)
    latent_model = LatentVAE(input_dim=16, hidden_dim=64, latent_dim=16)
    
    # Placeholder: In a real run, we'd load data and call trainers here
    logger.info("Models initialized. Skipping training for scaffold demo.")

    # Step 3: Initialize Environment
    logger.info("--- Step 3: Initializing Environment ---")
    env = ICUEnv(trajectory_path="./data/processed/icu_trajectories.parquet")

    # Step 4: Train DQN (Baseline)
    logger.info("--- Step 4: Training DQN ---")
    dqn = BaselineAgents(env, agent_type="DQN", device=device)
    dqn.train(total_timesteps=1000)
    dqn.save("./models/dqn/dqn_model")

    # Step 5: Train PPO (Baseline)
    logger.info("--- Step 5: Training PPO ---")
    ppo = BaselineAgents(env, agent_type="PPO", device=device)
    ppo.train(total_timesteps=1000)
    ppo.save("./models/ppo/ppo_model")

    # Step 6: Train CQL (Main Model)
    logger.info("--- Step 6: Training CQL ---")
    # Note: CQL usually trains on the static dataset directly
    cql = CQLAgent(state_dim=16, action_dim=4, device=device)
    # cql.train_offline("./data/processed/icu_trajectories.parquet", n_steps=1000)
    # cql.save("./models/cql/cql_model")
    logger.info("CQL training requires data. Skipping for now.")

    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
