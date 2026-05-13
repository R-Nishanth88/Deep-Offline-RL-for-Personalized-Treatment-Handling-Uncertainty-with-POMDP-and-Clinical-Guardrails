import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
from src.utils.logger import setup_logger
from src.safety.safety_constraints import SafetyLayer

logger = setup_logger("icu_env")

class ICUEnv(gym.Env):
    """
    Custom Gymnasium environment for ICU treatment recommendation.
    Uses offline trajectories to simulate patient transitions.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, trajectory_path="./data/processed/icu_trajectories.parquet", render_mode=None):
        super(ICUEnv, self).__init__()
        
        # Load processed trajectories
        if os.path.exists(trajectory_path):
            self.data = pd.read_parquet(trajectory_path)
            self.patient_ids = self.data['icustay_id'].unique()
        else:
            logger.warning(f"Trajectory data not found at {trajectory_path}. Using dummy data.")
            self.data = pd.DataFrame()
            self.patient_ids = []

        # Define Observation Space: Vitals + Labs (7 vitals + 9 labs = 16)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        
        # Define Action Space: 4 discrete levels (0: None, 1: Low, 2: Med, 3: High)
        self.action_space = spaces.Discrete(4)
        
        self.safety_layer = SafetyLayer()
        self.current_trajectory = None
        self.current_step = 0
        self.render_mode = render_mode

    def _get_obs(self):
        # Extract features for the current step
        # vitals: heart_rate, sysbp, diasbp, meanbp, resprate, tempc, spo2
        # labs: creatinine, bilirubin, platelets, wbc, lactate, glucose, ph, pao2, pco2
        row = self.current_trajectory.iloc[self.current_step]
        cols = ['heart_rate', 'sysbp', 'diasbp', 'meanbp', 'resprate', 'tempc', 'spo2',
                'creatinine', 'bilirubin', 'platelets', 'wbc', 'lactate', 'glucose', 'ph', 'pao2', 'pco2']
        return row[cols].values.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if len(self.patient_ids) == 0:
            # Fallback for dummy data
            self.current_trajectory = pd.DataFrame(np.zeros((10, 20)))
        else:
            # Pick a random patient trajectory
            pid = np.random.choice(self.patient_ids)
            self.current_trajectory = self.data[self.data['icustay_id'] == pid].reset_index(drop=True)
            
        self.current_step = 0
        observation = self._get_obs()
        info = {"patient_id": self.current_trajectory['subject_id'].iloc[0] if 'subject_id' in self.current_trajectory else "N/A"}
        
        return observation, info

    def step(self, action):
        # Apply safety layer to modify action if necessary
        obs_before = self._get_obs()
        # Demo: Assume patient is already on 'Epinephrine' to test interaction with agent's 'Norepinephrine'
        current_meds = ["Epinephrine"] if self.current_step > 2 else []
        safe_action = self.safety_layer.get_safe_action(obs_before, action, current_meds=current_meds)
        
        # Advance step
        self.current_step += 1
        
        # Check if terminal
        done = self.current_step >= len(self.current_trajectory) - 1
        terminated = done
        truncated = False
        
        # In a real environment, the next state would come from a transition model.
        # Here we follow the historical trajectory (Offline RL paradigm).
        observation = self._get_obs()
        
        # Reward from the trajectory (engineered in process_data.py)
        reward = self.current_trajectory.iloc[self.current_step]['reward']
        
        # Add safety penalty if action was modified
        if safe_action != action:
            reward -= 10.0
            
        info = {
            "original_action": action,
            "executed_action": safe_action,
            "safety_violation": safe_action != action,
            "sofa": self.current_trajectory.iloc[self.current_step].get('sofa', 0)
        }
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            obs = self._get_obs()
            print(f"Step: {self.current_step} | Vitals: HR={obs[0]:.1f}, BP={obs[1]:.1f}, SpO2={obs[6]:.1f}")
