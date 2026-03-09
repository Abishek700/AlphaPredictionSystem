from recommender import recommend_alpha

print("\n===================================")
print("  CONTEXT-AWARE ALPHA RECOMMENDER")
print("===================================\n")

sample_size = int(input("Enter sample size (required): "))
effect_size = float(input("Enter expected effect size (required): "))

print("\nRisk tolerance options:")
print("  very_strict | strict | moderate | lenient | very_lenient")
risk_tolerance = input("Choose risk tolerance [default: moderate]: ").strip().lower()
if risk_tolerance == "":
    risk_tolerance = "moderate"

test_type = input("Test type (one_tailed / two_tailed) [default: two_tailed]: ").strip().lower()
if test_type == "":
    test_type = "two_tailed"

num_tests_input = input("Number of hypothesis tests [default: 1]: ").strip()
num_tests = int(num_tests_input) if num_tests_input else 1

final_alpha = recommend_alpha(
    sample_size=sample_size,
    effect_size=effect_size,
    risk_tolerance=risk_tolerance,
    test_type=test_type,
    num_tests=num_tests
)

print("\n===================================")
print("           RESULTS")
print("===================================")
print(f"Recommended alpha: {final_alpha:.4f}")
