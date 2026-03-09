import os
import numpy as np
import pickle
from alpha_adjustment import adjust_alpha

# =====================================================
# Load trained system ONCE
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "final_alpha_recommendation_system.pkl"
)

with open(MODEL_PATH, "rb") as f:
    system = pickle.load(f)

model = system["stage1_model"]
scaler = system["scaler"]


# =====================================================
# CORE RECOMMENDATION FUNCTION
# =====================================================
def recommend_alpha(
    sample_size: int,
    effect_size: float,
    risk_tolerance: str,
    test_type: str,
    num_tests: int,
    research_domain: int = 0,
    replication_success: float = 0.5
) -> float:
    """
    Returns a context-aware alpha recommendation.
    """

    # Feature construction (same as training)
    X_new = np.array([[
        sample_size,
        effect_size,
        research_domain,
        replication_success,
        sample_size * effect_size,   # power proxy
        np.log1p(sample_size),
        effect_size ** 2
    ]])

    X_new_scaled = scaler.transform(X_new)

    # Stage 1: ML prediction
    base_alpha = model.predict(X_new_scaled)[0]

    # Stage 2: Adjustment layer
    final_alpha = adjust_alpha(
        base_alpha=base_alpha,
        risk_tolerance=risk_tolerance,
        test_type=test_type,
        num_tests=num_tests
    )

    return final_alpha
