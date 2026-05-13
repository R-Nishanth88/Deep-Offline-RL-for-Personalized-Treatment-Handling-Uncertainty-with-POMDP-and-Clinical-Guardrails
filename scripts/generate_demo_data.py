import pandas as pd
import numpy as np
import os

def generate_synthetic_mimic(path="./data/raw"):
    os.makedirs(path, exist_ok=True)
    n_patients = 10
    
    # 1. ICUSTAYS
    icustays = pd.DataFrame({
        'subject_id': range(n_patients),
        'hadm_id': range(100, 100+n_patients),
        'icustay_id': range(1000, 1000+n_patients),
        'intime': pd.to_datetime(['2026-05-12 12:00:00'] * n_patients),
        'outtime': pd.to_datetime(['2026-05-12 18:00:00'] * n_patients)
    })
    icustays.to_csv(os.path.join(path, "ICUSTAYS.csv"), index=False)
    
    # 2. ADMISSIONS
    admissions = pd.DataFrame({
        'subject_id': range(n_patients),
        'hadm_id': range(100, 100+n_patients),
        'hospital_expire_flag': np.random.randint(0, 2, n_patients)
    })
    admissions.to_csv(os.path.join(path, "ADMISSIONS.csv"), index=False)
    
    # 3. PATIENTS
    patients = pd.DataFrame({
        'subject_id': range(n_patients),
        'gender': np.random.choice(['M', 'F'], n_patients),
        'dob': pd.to_datetime(['1980-01-01'] * n_patients)
    })
    patients.to_csv(os.path.join(path, "PATIENTS.csv"), index=False)
    
    print(f"Synthetic MIMIC data generated at {path}")

if __name__ == "__main__":
    generate_synthetic_mimic()
