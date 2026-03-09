import numpy as np

# -------------------------------
# Risk tolerance multipliers
# -------------------------------
RISK_TOLERANCE_FACTORS = {
    "very_strict": 0.4,    # clinical trials
    "strict": 0.65,        # confirmatory
    "moderate": 1.0,       # balanced
    "lenient": 1.5,        # exploratory
    "very_lenient": 2.0    # pilot studies
}

MIN_ALPHA = 0.001
MAX_ALPHA = 0.15


def adjust_alpha(
    base_alpha: float,
    risk_tolerance: str,
    test_type: str,
    num_tests: int
):
    """
    Apply contextual adjustments to base alpha.

    Parameters
    ----------
    base_alpha : float
        Alpha predicted by ML model (Stage 1)
    risk_tolerance : str
        very_strict | strict | moderate | lenient | very_lenient
    test_type : str
        one_tailed | two_tailed
    num_tests : int
        Number of hypothesis tests

    Returns
    -------
    float
        Final adjusted alpha
    """

    # --- Risk tolerance ---
    risk_factor = RISK_TOLERANCE_FACTORS.get(risk_tolerance, 1.0)
    alpha = base_alpha * risk_factor

    # --- Test direction ---
    if test_type == "one_tailed":
        alpha *= 1.5

    # --- Multiple testing (Bonferroni) ---
    if num_tests > 1:
        alpha /= num_tests

    # --- Bound alpha ---
    alpha = np.clip(alpha, MIN_ALPHA, MAX_ALPHA)

    return round(alpha, 4)
