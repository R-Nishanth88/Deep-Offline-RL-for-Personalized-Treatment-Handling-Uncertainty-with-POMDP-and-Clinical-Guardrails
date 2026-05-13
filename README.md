# Safe-ICU: Personalized Clinical Decision Support using Deep Offline RL

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Safe-ICU** is a research-grade framework for optimizing treatment trajectories in Intensive Care Units using **Safe Offline Reinforcement Learning**. The system addresses partial observability in healthcare by formulating the problem as a **POMDP**, utilizing temporal encoders to capture latent health trends and **Conservative Q-Learning (CQL)** to ensure safe policy optimization from historical data.

## 🚀 Key Features
- **Temporal Latent Modeling**: Uses LSTM-VAEs to infer hidden physiological states ($z_t$).
- **Safe Offline RL**: Implements **CQL** and **QR-CQL** to prevent overestimation of risky treatments.
- **Clinical Safety Guardrails**: Multi-layered watchdog with vitals-based veto logic and drug-interaction checks.
- **Explainable AI (XAI)**: Integrated **SHAP** feature importance for clinical decision rationale.
- **Interactive Dashboard**: Streamlit-based interface for real-time risk assessment and historical patient audit.

## 🏗️ Architecture
The framework operates in five distinct stages:
1. **Data Engineering**: Processing MIMIC-III EHR records into hourly trajectories.
2. **Representation Learning**: Encoding 24-hour histories into a 16D Latent Belief State.
3. **Policy Optimization**: Training CQL agents on offline data with ensemble uncertainty.
4. **Safety Layer**: Intercepting AI actions through hard-coded clinical constraints.
5. **Human-in-the-loop**: Providing explained recommendations via an interactive UI.

## 📂 Project Structure
```text
healthcare_rl_project/
├── dashboard/              # Streamlit Application
├── data/                   # Processed MIMIC-III trajectories
├── results/                # Metrics and Performance Plots
├── src/
│   ├── rl_agents/          # CQL and QR-CQL implementations
│   ├── environment/        # Gymnasium-based ICU simulator
│   ├── temporal_model/     # LSTM-VAE Sequential Encoder
│   ├── safety/             # Veto logic and drug interaction DB
│   └── evaluation/         # Metrics and OPE (FQE) scripts
└── README.md               # This file
```

## 🛠️ Installation & Usage
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/R-Nishanth88/Safe-ICU.git
   cd Safe-ICU
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

## 📊 Results
The framework achieves an **88% alignment** with clinician policies while reducing safety violations to **0%** through its veto layer. Detailed metrics and FQE robustness benchmarks can be found in the `results/` directory.

## 📜 References
- [1] Johnson et al. (2016) - MIMIC-III Database.
- [2] Kumar et al. (2020) - Conservative Q-Learning (CQL).
- [3] Komorowski et al. (2018) - The AI Clinician (Nature Medicine).

## 👥 Team Members
- **NEHA GIRISH MANTUR** - AID23034
- **R NISHANTH DATTA** - aid23043
- **S SAi preran** - aid 23045

---
**Developed for [Project Title/Course Name]**
