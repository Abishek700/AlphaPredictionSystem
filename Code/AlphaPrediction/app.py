import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from recommender import recommend_alpha

st.title("Alpha Recommendation System")

sample_size = st.slider("Sample size", 20, 1000, 100)
effect_size = st.slider("Effect size", 0.05, 1.5, 0.3)

risk_tolerance = st.selectbox(
    "Risk tolerance",
    ["very_strict", "strict", "moderate", "lenient", "very_lenient"]
)

test_type = st.selectbox(
    "Test type",
    ["one_tailed", "two_tailed"]
)

num_tests = st.number_input(
    "Number of hypothesis tests",
    min_value=1,
    value=1,
    step=1
)

if st.button("Recommend alpha"):
    alpha = recommend_alpha(
        sample_size=sample_size,
        effect_size=effect_size,
        risk_tolerance=risk_tolerance,
        test_type=test_type,
        num_tests=num_tests
    )

    st.success(f"Recommended alpha: {alpha:.4f}")
