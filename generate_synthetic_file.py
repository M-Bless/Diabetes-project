import pandas as pd
import numpy as np

# Step 1: Basic setup
n_rows = 10000
np.random.seed(42)

# Step 2: Generate numeric features (realistic ranges)
historic_glucose = np.round(np.random.uniform(3.0, 20.0, n_rows), 1)
scan_glucose = np.round(historic_glucose + np.random.normal(0, 1.5, n_rows), 1)
rapid_insulin = np.random.randint(0, 21, n_rows)
long_insulin = np.random.randint(8, 21, n_rows)
carbs = np.random.randint(0, 151, n_rows)

# Step 3: Define 4 glucose-based conditions (FIXED THRESHOLDS)
# 0 = Hypoglycemia (<3.9)
# 1 = Normal (3.9–6.9)
# 2 = Hyperglycemia (7–13.9)
# 3 = DKA (≥14)
condition = []
for g in scan_glucose:
    if g < 3.9:  # FIXED: Changed from 4 to 3.9
        condition.append(0)
    elif 3.9 <= g < 7:  # FIXED: Changed from 4 to 3.9
        condition.append(1)
    elif 7 <= g < 14:
        condition.append(2)
    else:
        condition.append(3)

condition = np.array(condition)

# Step 4: Symptom generation logic per condition (IMPROVED)
def generate_symptoms(cond, glucose):
    """
    Generate symptoms based on condition AND glucose value
    Returns: [Thirst, Nausea, Weakness, Vomiting, Fatigue, Shakiness]
    """
    if cond == 0:  # Hypoglycemia (<3.9)
        return [
            0,  # Thirst (rare in hypo)
            0,  # Nausea (rare in hypo)
            np.random.choice([0, 1], p=[0.3, 0.7]),  # Weakness (common)
            0,  # Vomiting (rare in hypo)
            np.random.choice([0, 1], p=[0.2, 0.8]),  # Fatigue (very common)
            np.random.choice([0, 1], p=[0.2, 0.8])   # Shakiness (very common)
        ]
    
    elif cond == 1:  # Normal (3.9-6.9)
        # Borderline low (3.9-5.0) might have mild symptoms
        if glucose < 5.0:
            return [
                0,  # Thirst
                0,  # Nausea
                np.random.choice([0, 1], p=[0.8, 0.2]),  # Weakness (20% chance)
                0,  # Vomiting
                np.random.choice([0, 1], p=[0.8, 0.2]),  # Fatigue (20% chance)
                np.random.choice([0, 1], p=[0.6, 0.4])   # Shakiness (40% chance)
            ]
        else:
            # Mid-normal range (5.0-6.9) - mostly no symptoms but some variation
            return [
                0,  # Thirst
                0,  # Nausea
                np.random.choice([0, 1], p=[0.9, 0.1]),  # Weakness (10% chance - people get tired!)
                0,  # Vomiting
                np.random.choice([0, 1], p=[0.9, 0.1]),  # Fatigue (10% chance)
                0   # Shakiness (not at this level)
            ]
    
    elif cond == 2:  # Hyperglycemia (7-13.9)
        return [
            np.random.choice([0, 1], p=[0.3, 0.7]),  # Thirst (very common)
            np.random.choice([0, 1], p=[0.6, 0.4]),  # Nausea (moderate)
            np.random.choice([0, 1], p=[0.5, 0.5]),  # Weakness (common)
            0,  # Vomiting (rare unless severe)
            np.random.choice([0, 1], p=[0.4, 0.6]),  # Fatigue (common)
            0   # Shakiness (not typical for hyperglycemia)
        ]
    
    else:  # DKA (≥14)
        return [
            1,  # Thirst (always present)
            1,  # Nausea (always present)
            1,  # Weakness (always present)
            1,  # Vomiting (always present)
            1,  # Fatigue (always present)
            0   # Shakiness (not typical for DKA)
        ]

# Step 5: Generate symptom data (PASS BOTH condition AND glucose)
symptoms = np.array([generate_symptoms(c, g) for c, g in zip(condition, scan_glucose)])
thirst, nausea, weakness, vomiting, fatigue, shakiness = symptoms.T

# Step 6: Create DataFrame
df = pd.DataFrame({
    'Historic Glucose (mmol/L)': historic_glucose,
    'Scan Glucose (mmol/L)': scan_glucose,
    'Rapid-Acting Insulin (units)': rapid_insulin,
    'Long-Acting Insulin (units)': long_insulin,
    'Carbohydrates (grams)': carbs,
    'Thirst': thirst,
    'Nausea': nausea,
    'Weakness': weakness,
    'Vomiting': vomiting,
    'Fatigue': fatigue,
    'Shakiness': shakiness,
    'Condition': condition
})

# Step 7: Save to CSV
df.to_csv('synthetic_diabetes_data.csv', index=False)

# Step 8: Print summary
print("Synthetic dataset generated: synthetic_diabetes_data.csv")
print(f"Total rows: {len(df)}\n")

print(" Condition Distribution:")
print(df['Condition'].value_counts().sort_index())
print("\n Condition Mapping:")
print("0 = Hypoglycemia (<3.9 mmol/L)")
print("1 = Normal (3.9–6.9 mmol/L)")
print("2 = Hyperglycemia (7–13.9 mmol/L)")
print("3 = DKA (≥14 mmol/L)")

# Step 9: Verify symptom patterns
print("\n🔍 Symptom Verification:")
print("\nHypoglycemia samples (should have Shakiness, Weakness, Fatigue):")
print(df[df['Condition'] == 0][['Scan Glucose (mmol/L)', 'Shakiness', 'Weakness', 'Fatigue']].head(3))

print("\nNormal (low-end) samples (might have Shakiness):")
normal_low = df[(df['Condition'] == 1) & (df['Scan Glucose (mmol/L)'] < 5.0)]
print(normal_low[['Scan Glucose (mmol/L)', 'Shakiness']].head(3))

print("\nHyperglycemia samples (should have Thirst, Fatigue):")
print(df[df['Condition'] == 2][['Scan Glucose (mmol/L)', 'Thirst', 'Fatigue', 'Nausea']].head(3))

print("\nDKA samples (should have all symptoms except Shakiness):")
print(df[df['Condition'] == 3][['Scan Glucose (mmol/L)', 'Thirst', 'Nausea', 'Weakness', 'Vomiting', 'Fatigue']].head(3))