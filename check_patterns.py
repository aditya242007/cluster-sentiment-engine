"""Validate that injected patterns are recoverable from the generated dataset."""
import sys
import pandas as pd
from src.cri.classify import RuleBasedClassifier
from src.cri.generate import generate_synthetic_reviews

print("Loading reviews and running classifier...")
df = generate_synthetic_reviews(n_reviews=5000, seed=42)
classifier = RuleBasedClassifier()

food_scores, room_scores, staff_scores = [], [], []
for text in df["review_text"]:
    res = classifier.classify(text)
    food_scores.append(res["Food"]["score"])
    room_scores.append(res["Room"]["score"])
    staff_scores.append(res["Staff"]["score"])

df["Food"] = food_scores
df["Room"] = room_scores
df["Staff"] = staff_scores
df["month"] = pd.to_datetime(df["review_date"]).dt.month

failures = []

# --- Pattern 1: P1 Food declines from month 1 to month 6 ---
p1 = df[df["property_id"] == "P1"]
p1_m1 = p1[p1["month"] == 1]["Food"].mean()
p1_m6 = p1[p1["month"] == 6]["Food"].mean()
gap1 = p1_m1 - p1_m6
result1 = "PASS" if gap1 > 0.4 else "FAIL"
if result1 == "FAIL":
    failures.append("Pattern 1")
print(f"[{result1}] Pattern 1 - P1 Food Decline: month1={p1_m1:.3f}, month6={p1_m6:.3f}, gap={gap1:.3f} (need > 0.4)")

# --- Pattern 2: C2 Room winter dip ---
c2 = df[df["cluster_id"] == "C2"]
c2_winter = c2[c2["month"].isin([11, 12, 1, 2])]["Room"].mean()
c2_summer = c2[c2["month"].isin([6, 7, 8, 9])]["Room"].mean()
gap2 = c2_summer - c2_winter
result2 = "PASS" if gap2 > 0.4 else "FAIL"
if result2 == "FAIL":
    failures.append("Pattern 2")
print(f"[{result2}] Pattern 2 - C2 Room Winter Dip: winter={c2_winter:.3f}, summer={c2_summer:.3f}, gap={gap2:.3f} (need > 0.4)")

# --- Pattern 3: P4 Staff improves after month 3 ---
p4 = df[df["property_id"] == "P4"]
p4_early = p4[p4["month"] <= 3]["Staff"].mean()
p4_late = p4[p4["month"] >= 4]["Staff"].mean()
gap3 = p4_late - p4_early
result3 = "PASS" if gap3 > 0.4 else "FAIL"
if result3 == "FAIL":
    failures.append("Pattern 3")
print(f"[{result3}] Pattern 3 - P4 Staff Improvement: early={p4_early:.3f}, late={p4_late:.3f}, gap={gap3:.3f} (need > 0.4)")

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
else:
    print("ALL 3 PATTERNS VERIFIED")
