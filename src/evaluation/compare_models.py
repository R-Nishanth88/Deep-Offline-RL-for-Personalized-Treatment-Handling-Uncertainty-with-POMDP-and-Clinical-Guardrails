import pandas as pd
import numpy as np
import torch
from src.environment.icu_env import ICUEnv
from src.rl_agents.cql_agent import CQLAgent
from src.rl_agents.baselines import BaselineAgents
from src.evaluation.metrics import calculate_rl_metrics, calculate_clinical_metrics, calculate_safety_metrics, generate_performance_report
from src.utils.logger import setup_logger

logger = setup_logger("model_comparison")

def evaluate_agent(env, agent, n_episodes=10):
    all_rewards = []
    all_sofa = []
    unsafe_actions = 0
    total_steps = 0
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Handle different agent prediction APIs
            if hasattr(agent, 'predict'):
                action = agent.predict(obs)
                # Stable-Baselines returns (action, _states)
                if isinstance(action, tuple): action = action[0]
            else:
                action = env.action_space.sample()
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            if info.get('safety_violation'):
                unsafe_actions += 1
            total_steps += 1
            
        all_rewards.append(episode_reward)
        all_sofa.append(info.get('sofa', 0))
        
    return {
        "rl": calculate_rl_metrics(all_rewards),
        "clinical": calculate_clinical_metrics(all_sofa, [1.0]*len(all_sofa), 0.1), # Placeholders
        "safety": calculate_safety_metrics(unsafe_actions, total_steps)
    }

def run_comparison():
    env = ICUEnv()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Models
    models = {
        "DQN": BaselineAgents(env, "DQN"),
        "PPO": BaselineAgents(env, "PPO"),
        "CQL": CQLAgent(state_dim=16, action_dim=4, device=device)
    }
    
    # Load weights if they exist
    # for name, model in models.items():
    #     model.load(f"./models/{name.lower()}/{name.lower()}_model")

    results = []
    for name, agent in models.items():
        logger.info(f"Evaluating {name}...")
        metrics = evaluate_agent(env, agent)
        
        # Flatten metrics for table
        flat_metrics = {"Model": name}
        flat_metrics.update(metrics['rl'])
        flat_metrics.update(metrics['clinical'])
        flat_metrics.update(metrics['safety'])
        results.append(flat_metrics)
        
    df = pd.DataFrame(results)
    df.to_csv("./results/metrics/model_comparison.csv", index=False)
    logger.info("\nComparison Results:\n" + df.to_string())
    
    return df

if __name__ == "__main__":
    run_comparison()
