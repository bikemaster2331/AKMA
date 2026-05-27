"""
Quick test: Does lazy delta verification catch a lie?
Tests ground_delta() directly without going through the full pipeline.
"""

from searcher import ground_delta

# The original trusted document (from seed)
original = (
    "Python is a high-level interpreted programming language created by "
    "Guido van Rossum in 1991. It emphasizes code readability and supports "
    "multiple programming paradigms."
)

# ── TEST 1: A FALSE expansion (should be BLOCKED) ─────────────────────────────
print("=" * 60)
print("TEST 1: FALSE claim — Python named after a snake species")
print("=" * 60)

fake_refined = (
    "Python is a high-level interpreted programming language created by "
    "Guido van Rossum in 1991. It emphasizes code readability and supports "
    "multiple programming paradigms. The name Python comes from the python "
    "snake species, which is Guido's favorite animal."
)

result = ground_delta(original, fake_refined)
print(f"\nResult: {'BLOCKED' if not result['passed'] else 'PASSED'}")
print(f"Claims checked: {result['claims_checked']}")
print(f"Claims grounded: {result['claims_grounded']}")
print(f"Unverified: {result['claims_unverified']}")

# ── TEST 2: A TRUE expansion (should PASS) ────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: TRUE claim — Python named after Monty Python")
print("=" * 60)

true_refined = (
    "Python is a high-level interpreted programming language created by "
    "Guido van Rossum in 1991. It emphasizes code readability and supports "
    "multiple programming paradigms. The name Python was inspired by the "
    "British comedy series Monty Python's Flying Circus."
)

result2 = ground_delta(original, true_refined)
print(f"\nResult: {'BLOCKED' if not result2['passed'] else 'PASSED'}")
print(f"Claims checked: {result2['claims_checked']}")
print(f"Claims grounded: {result2['claims_grounded']}")
print(f"Unverified: {result2['claims_unverified']}")
