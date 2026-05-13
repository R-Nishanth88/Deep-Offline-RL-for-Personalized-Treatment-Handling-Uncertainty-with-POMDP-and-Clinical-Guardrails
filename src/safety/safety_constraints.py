from src.utils.logger import setup_logger
from src.safety.drug_interactions import get_icu_drug_database

logger = setup_logger("safety_layer")

class SafetyLayer:
    """
    Safe Reinforcement Learning layer to enforce clinical constraints and drug-drug interactions.
    """
    def __init__(self, bp_threshold=70, hr_min=50, hr_max=140, spo2_threshold=90):
        self.bp_threshold = bp_threshold
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.spo2_threshold = spo2_threshold
        self.drug_db = get_icu_drug_database()
        self.total_violations = 0

    def get_safe_action(self, state, proposed_action, current_meds=None):
        """
        Validates the proposed action against clinical vitals and drug databases.
        
        Args:
            state: Patient observation vector.
            proposed_action: The action suggested by the RL agent.
            current_meds: List of drug names the patient is already receiving.
        """
        # 1. Clinical Vitals Safety Check
        safe_action = self._check_vitals_safety(state, proposed_action)
        
        # 2. Multi-Drug Interaction Check
        if current_meds and proposed_action > 0:
            # Map action levels to primary drug for demo purposes
            # In a real system, the agent might suggest specific drugs.
            proposed_drug = "Norepinephrine" 
            conflicts = self.drug_db.check_interaction(current_meds, proposed_drug)
            
            if conflicts:
                self.total_violations += 1
                highest_risk = conflicts[0] # Simplification
                logger.warning(f"🚨 CLINICAL ALERT: Interaction with {highest_risk['with']}! {highest_risk['warning']}")
                
                # If High Risk, veto the treatment to 'None' (0)
                if highest_risk['level'] == "High":
                    logger.info("Vetoing action to 0 due to High Risk interaction.")
                    return 0
                # If Medium Risk, cap the dosage to 'Low' (1)
                elif highest_risk['level'] == "Medium":
                    logger.info("Capping action to 1 due to Medium Risk interaction.")
                    return min(safe_action, 1)

        return safe_action

    def _check_vitals_safety(self, state, action):
        """
        Enforce hard clinical boundaries based on vitals.
        """
        hr = state[0]
        sysbp = state[1]
        spo2 = state[6]
        
        # Veto 1: Prevent 'No Treatment' during hypotension
        if sysbp < self.bp_threshold and action == 0:
            self.total_violations += 1
            logger.warning(f"Safety Veto: BP {sysbp:.1f} is too low. Forcing Low Dosage (1).")
            return 1
            
        # Veto 2: Prevent 'High Dosage' during extreme tachycardia
        if hr > self.hr_max and action == 3:
            self.total_violations += 1
            logger.warning(f"Safety Veto: HR {hr:.1f} is too high for High Dosage. Capping at Medium (2).")
            return 2
            
        # Veto 3: Prevent 'No Treatment' during hypoxia
        if spo2 < self.spo2_threshold and action == 0:
            self.total_violations += 1
            logger.warning(f"Safety Veto: SpO2 {spo2:.1f}% is hypoxic. Forcing Low Support (1).")
            return 1
            
        return action
