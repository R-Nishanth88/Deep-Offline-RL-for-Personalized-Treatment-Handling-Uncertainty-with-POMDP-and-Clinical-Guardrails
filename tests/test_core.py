import pytest
import torch
import numpy as np
from src.utils.seed import set_seed
from src.environment.icu_env import ICUEnv
from src.safety.safety_constraints import SafetyLayer

def test_reproducibility():
    set_seed(42)
    a = np.random.rand(10)
    set_seed(42)
    b = np.random.rand(10)
    assert np.allclose(a, b)

def test_safety_layer():
    safety = SafetyLayer(bp_threshold=70)
    # Mock state where SysBP is 65 (unsafe)
    state = np.zeros(16)
    state[1] = 65 
    
    # Action 0 (no intervention) should be modified
    proposed_action = 0
    safe_action = safety.get_safe_action(state, proposed_action)
    assert safe_action != proposed_action
    assert safe_action == 1

def test_env_reset():
    # Test if environment can be instantiated and reset
    # (Using dummy data fallback)
    env = ICUEnv(trajectory_path="non_existent.parquet")
    obs, info = env.reset()
    assert obs.shape == (16,)
    assert isinstance(info, dict)
