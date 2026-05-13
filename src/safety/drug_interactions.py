from src.utils.logger import setup_logger

logger = setup_logger("drug_safety")

class DrugInteractionDatabase:
    """
    A clinical database of common ICU drug interactions.
    """
    def __init__(self):
        # Format: (Drug A, Drug B): {risk_level, description}
        self.interactions = {
            ("Norepinephrine", "Epinephrine"): {
                "level": "High",
                "warning": "Synergistic vasoconstriction. Risk of severe hypertension and cardiac arrhythmia."
            },
            ("Propofol", "Dexmedetomidine"): {
                "level": "Medium",
                "warning": "Additive sedative effects. Risk of severe bradycardia and profound hypotension."
            },
            ("Vancomycin", "Piperacillin"): {
                "level": "Medium",
                "warning": "Increased risk of Acute Kidney Injury (AKI) in critical care settings."
            },
            ("Warfarin", "Aspirin"): {
                "level": "High",
                "warning": "Significant increase in major bleeding risk."
            },
            ("Fentanyl", "Midazolam"): {
                "level": "Low",
                "warning": "Enhanced respiratory depression. Requires close monitoring."
            }
        }

    def check_interaction(self, drug_list, proposed_drug):
        """
        Check if the proposed drug interacts with any drug in the current list.
        """
        conflicts = []
        for current_drug in drug_list:
            # Check both directions (A,B) and (B,A)
            key1 = (current_drug, proposed_drug)
            key2 = (proposed_drug, current_drug)
            
            if key1 in self.interactions:
                conflicts.append({**self.interactions[key1], "with": current_drug})
            elif key2 in self.interactions:
                conflicts.append({**self.interactions[key2], "with": current_drug})
                
        return conflicts

def get_icu_drug_database():
    return DrugInteractionDatabase()
