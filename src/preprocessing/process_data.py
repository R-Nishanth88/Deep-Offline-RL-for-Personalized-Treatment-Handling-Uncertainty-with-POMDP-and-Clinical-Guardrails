import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from src.utils.logger import setup_logger
from src.feature_engineering.clinical_scores import calculate_sofa_score, calculate_stability_score

logger = setup_logger("data_preprocessing")

class MIMICPreprocessor:
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path
        os.makedirs(processed_path, exist_ok=True)
        
        # Clinical feature lists
        self.vitals = ['heart_rate', 'sysbp', 'diasbp', 'meanbp', 'resprate', 'tempc', 'spo2']
        self.labs = ['creatinine', 'bilirubin', 'platelets', 'wbc', 'lactate', 'glucose', 'ph', 'pao2', 'pco2']
        self.demographics = ['age', 'gender', 'weight']
        
    def load_table(self, name):
        path = os.path.join(self.raw_path, f"{name}.csv")
        logger.info(f"Loading {name} from {path}")
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            logger.error(f"Table {name} not found!")
            # Return empty df with expected structure for robustness in dev
            return pd.DataFrame()

    def process_trajectories(self):
        """
        Main pipeline to convert MIMIC tables into hourly RL trajectories.
        """
        logger.info("Starting trajectory processing...")
        
        # 1. Load core tables
        icustays = self.load_table("ICUSTAYS")
        admissions = self.load_table("ADMISSIONS")
        patients = self.load_table("PATIENTS")
        
        if icustays.empty:
            logger.warning("No data found to process. Exiting.")
            return

        # 2. Join demographics and mortality
        df = icustays.merge(admissions, on=['subject_id', 'hadm_id'], how='inner')
        df = df.merge(patients, on='subject_id', how='inner')
        
        # 3. Create hourly windows
        trajectories = []
        
        # For each ICU stay
        for _, stay in tqdm(df.head(100).iterrows(), total=min(len(df), 100), desc="Processing stays"):
            stay_id = stay['icustay_id']
            start_time = pd.to_datetime(stay['intime'])
            end_time = pd.to_datetime(stay['outtime'])
            
            # Generate hourly index
            hours = pd.date_range(start=start_time, end=end_time, freq='h')
            
            # Placeholder for actual event loading and mapping
            # In a real scenario, we'd query CHARTEVENTS, LABEVENTS for this stay_id
            
            stay_data = pd.DataFrame(index=hours)
            stay_data['icustay_id'] = stay_id
            stay_data['subject_id'] = stay['subject_id']
            
            # Fill with random data for demonstration of structure
            stay_data['heart_rate'] = np.random.normal(80, 10, len(stay_data))
            stay_data['sysbp'] = np.random.normal(120, 15, len(stay_data))
            stay_data['spo2'] = np.random.normal(97, 2, len(stay_data))
            stay_data['map'] = np.random.normal(85, 10, len(stay_data))
            stay_data['bilirubin'] = np.random.uniform(0.5, 2.0, len(stay_data))
            stay_data['creatinine'] = np.random.uniform(0.8, 1.5, len(stay_data))
            stay_data['platelets'] = np.random.uniform(150, 450, len(stay_data))
            stay_data['pao2_fio2'] = np.random.uniform(300, 500, len(stay_data))
            stay_data['gcs'] = np.random.choice([13, 14, 15], len(stay_data))
            
            for col in self.vitals + self.labs:
                if col not in stay_data.columns:
                    stay_data[col] = np.random.normal(0, 1, len(stay_data))
            
            # 4. Handle Missing Values
            stay_data = stay_data.ffill().bfill().interpolate()
            
            # 5. Feature Engineering (SOFA, Stability)
            stay_data['sofa'] = stay_data.apply(calculate_sofa_score, axis=1)
            stay_data['stability'] = stay_data.apply(calculate_stability_score, axis=1)
            
            # 6. Action Discretization
            # Logic: No intervention (0), Low (1), Med (2), High (3)
            # Based on vasopressor dose or ventilation status
            stay_data['action'] = np.random.choice([0, 1, 2, 3], size=len(stay_data))
            
            # 7. Reward Calculation
            # Reward = +survival - penalty(instability) - penalty(unsafe)
            stay_data['reward'] = 0.0
            stay_data.loc[stay_data.index[-1], 'reward'] = 100.0 if stay['hospital_expire_flag'] == 0 else -100.0
            stay_data['reward'] -= (stay_data['sofa'] * 0.1)
            
            # 8. Format: State_t, Action_t, Reward_t, Next_State_t, Done
            stay_data['next_state'] = stay_data[self.vitals + self.labs].shift(-1).values.tolist()
            stay_data['done'] = False
            stay_data.iloc[-1, stay_data.columns.get_loc('done')] = True
            
            trajectories.append(stay_data)
            
        final_df = pd.concat(trajectories)
        output_file = os.path.join(self.processed_path, "icu_trajectories.parquet")
        final_df.to_parquet(output_file)
        logger.info(f"Saved processed trajectories to {output_file}")

if __name__ == "__main__":
    preprocessor = MIMICPreprocessor(
        raw_path="./data/raw",
        processed_path="./data/processed"
    )
    preprocessor.process_trajectories()
