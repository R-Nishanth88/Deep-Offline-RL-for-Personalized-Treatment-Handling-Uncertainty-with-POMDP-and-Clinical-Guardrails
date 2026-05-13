import numpy as np
import pandas as pd

def calculate_sofa_score(df):
    """
    Calculate SOFA (Sequential Organ Failure Assessment) score.
    Expects columns for:
    - respiration (PaO2/FiO2)
    - coagulation (Platelets)
    - liver (Bilirubin)
    - cardiovascular (MAP, vasopressors)
    - CNS (GCS)
    - renal (Creatinine, Urine output)
    """
    sofa = 0
    
    # 1. Respiration (PaO2/FiO2)
    if 'pao2_fio2' in df:
        val = df['pao2_fio2']
        if val < 100: sofa += 4
        elif val < 200: sofa += 3
        elif val < 300: sofa += 2
        elif val < 400: sofa += 1
        
    # 2. Coagulation (Platelets x 10^3/mm3)
    if 'platelets' in df:
        val = df['platelets']
        if val < 20: sofa += 4
        elif val < 50: sofa += 3
        elif val < 100: sofa += 2
        elif val < 150: sofa += 1
        
    # 3. Liver (Bilirubin mg/dL)
    if 'bilirubin' in df:
        val = df['bilirubin']
        if val >= 12.0: sofa += 4
        elif val >= 6.0: sofa += 3
        elif val >= 2.0: sofa += 2
        elif val >= 1.2: sofa += 1
        
    # 4. Cardiovascular (MAP or vasopressors)
    # Simplified: using MAP thresholds
    if 'map' in df:
        val = df['map']
        if val < 70: sofa += 1
        # Vasopressor use would add 2, 3, or 4 points
        if df.get('vasopressor_used', 0) > 0:
            sofa += 2 # Simplified baseline for pressor use
            
    # 5. CNS (GCS)
    if 'gcs' in df:
        val = df['gcs']
        if val <= 5: sofa += 4
        elif val <= 9: sofa += 3
        elif val <= 12: sofa += 2
        elif val <= 14: sofa += 1
        
    # 6. Renal (Creatinine mg/dL or Urine output)
    if 'creatinine' in df:
        val = df['creatinine']
        if val >= 5.0: sofa += 4
        elif val >= 3.5: sofa += 3
        elif val >= 2.0: sofa += 2
        elif val >= 1.2: sofa += 1
        
    return sofa

def calculate_stability_score(df):
    """
    Custom patient stability score based on vital sign variability.
    """
    # Inverse of weighted variance of vital signs
    vital_cols = ['heart_rate', 'sysbp', 'resprate', 'spo2']
    score = 1.0
    for col in vital_cols:
        if col in df and pd.notnull(df[col]):
            # Simplified stability check: penalize extreme values
            # (Assuming normalized values or basic heuristics)
            if df[col] > 120 or df[col] < 50:
                score -= 0.2
    return max(0, score)
