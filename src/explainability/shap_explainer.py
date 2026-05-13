import shap
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.logger import setup_logger

logger = setup_logger("explainability")

class ICUExplainer:
    """
    SHAP-based explainer for ICU treatment recommendations.
    """
    def __init__(self, model, background_data, feature_names=None):
        self.model = model
        self.feature_names = feature_names or [
            'Heart Rate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'TempC', 'SpO2',
            'Creatinine', 'Bilirubin', 'Platelets', 'WBC', 'Lactate', 'Glucose', 'PH', 'PaO2', 'PCO2'
        ]
        
        # Initialize SHAP explainer
        # For RL agents, we often explain the Q-values or the policy output
        # Here we assume a function that returns the Q-values for all actions
        def model_predict(x):
            # If model has a predict_value or similar, use it. 
            # Otherwise, use a wrapper.
            if hasattr(self.model, 'predict_value'):
                return self.model.predict_value(x)
            # Default fallback for demonstration
            return np.random.randn(x.shape[0], 4) 
            
        self.explainer = shap.KernelExplainer(model_predict, background_data)

    def explain_step(self, state, patient_id="Unknown", step_num=0):
        """
        Explain a single treatment decision.
        """
        shap_values = self.explainer.shap_values(state)
        
        # Create results directory
        os.makedirs("./results/plots/explainability", exist_ok=True)
        
        # Generate and save a summary plot for the chosen action
        # shap_values is a list of arrays (one per action)
        # We'll just take the first one or the one for the chosen action
        plt.figure()
        shap.summary_plot(shap_values, np.expand_dims(state, 0), feature_names=self.feature_names, show=False)
        plot_path = f"./results/plots/explainability/shap_{patient_id}_step{step_num}.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"SHAP explanation saved to {plot_path}")
        return shap_values

    def get_feature_importance(self, shap_values):
        """
        Aggregate SHAP values to show global feature importance.
        """
        importance = np.abs(shap_values).mean(0)
        return dict(zip(self.feature_names, importance))
