import pandas as pd
import numpy as np
from collections import Counter

# ============================================
# BALANCED TYPE 1 DIABETES SYNTHETIC DATA V6
# Scan Glucose now has stronger influence
# ============================================
samples_per_class = 2500
n_rows = samples_per_class * 4
np.random.seed(42)

print("Generating Balanced Type 1 Diabetes Synthetic Data V6...")
print("=" * 60)
print(f"Target: {samples_per_class} samples per condition")
print(f"Total samples: {n_rows}")

# ============================================
# STEP 1: Baseline glucose
# ============================================
baseline_glucose = np.round(np.random.uniform(4.0, 12.0, n_rows), 1)

# ============================================
# STEP 2: Long-acting insulin
# ============================================
long_insulin = np.random.randint(8, 21, n_rows)

# ============================================
# STEP 3: Carb intake
# ============================================
meal_type = np.random.choice(['none', 'snack', 'small', 'medium', 'large'], 
                             n_rows, p=[0.2, 0.2, 0.2, 0.3, 0.1])
carbs = np.zeros(n_rows)
for i, meal in enumerate(meal_type):
    if meal == 'none':
        carbs[i] = 0
    elif meal == 'snack':
        carbs[i] = np.random.randint(10, 25)
    elif meal == 'small':
        carbs[i] = np.random.randint(25, 45)
    elif meal == 'medium':
        carbs[i] = np.random.randint(45, 80)
    else:
        carbs[i] = np.random.randint(80, 130)
carbs = carbs.astype(int)

# ============================================
# STEP 4: Rapid-acting insulin
# ============================================
rapid_insulin = np.zeros(n_rows)
for i in range(n_rows):
    correction = 0
    if baseline_glucose[i] > 8.0:
        correction = (baseline_glucose[i] - 7.0) / 2.0

    meal_bolus = carbs[i] / 8.0
    rapid_insulin[i] = correction + meal_bolus
    rapid_insulin[i] += np.random.normal(0, 0.5)
    rapid_insulin[i] = np.clip(rapid_insulin[i], 0, 20)
rapid_insulin = np.round(rapid_insulin, 0).astype(int)

# ============================================
# STEP 5: Actual glucose (stronger Scan Glucose influence)
# ============================================
scan_glucose = np.zeros(n_rows)
for i in range(n_rows):
    glucose = baseline_glucose[i]
    glucose += carbs[i] * 0.03        # carbs contribute moderately
    glucose -= rapid_insulin[i] * 1.0 # insulin moderate effect
    glucose += np.random.normal(0, 0.5)  # small noise
    glucose = np.clip(glucose, 2.5, 25.0)
    scan_glucose[i] = np.round(glucose, 1)

# ============================================
# STEP 6: Historic glucose
# ============================================
historic_glucose = np.zeros(n_rows)
for i in range(n_rows):
    trend = np.random.normal(0, 1.0)
    historic_glucose[i] = np.clip(scan_glucose[i] - carbs[i]*0.02 + rapid_insulin[i]*1.5 + trend, 3.0, 20.0)
historic_glucose = np.round(historic_glucose, 1)

# ============================================
# STEP 6.5: Target-based condition generation
# ============================================
condition_indices = {0: [], 1: [], 2: [], 3: []}
for idx in range(n_rows):
    g = scan_glucose[idx]
    if g < 4.0:
        condition_indices[0].append(idx)
    elif g < 7.0:
        condition_indices[1].append(idx)
    elif g < 12.0:
        condition_indices[2].append(idx)
    else:
        condition_indices[3].append(idx)

all_available = set(range(n_rows))
selected_indices = {0: [], 1: [], 2: [], 3: []}

for cond in range(4):
    current = condition_indices[cond]
    needed = samples_per_class
    if len(current) >= needed:
        selected_indices[cond] = list(np.random.choice(current, needed, replace=False))
    else:
        selected_indices[cond] = current.copy()
        remaining_needed = needed - len(current)
        available = list(all_available - set([idx for indices in selected_indices.values() for idx in indices]))
        to_convert = np.random.choice(available, min(remaining_needed, len(available)), replace=False)
        
        for idx in to_convert:
            if cond == 0:
                rapid_insulin[idx] = np.random.randint(8, 18)
                carbs[idx] = np.random.randint(0, 30)
                scan_glucose[idx] = np.random.uniform(2.5, 3.9)
                historic_glucose[idx] = np.clip(scan_glucose[idx] + np.random.uniform(1.5, 4.0), 4.0, 8.0)
            elif cond == 1:
                scan_glucose[idx] = np.random.uniform(4.0, 6.9)
                rapid_insulin[idx] = np.random.randint(0, 8)
                carbs[idx] = np.random.randint(0, 60)
                historic_glucose[idx] = np.clip(scan_glucose[idx] + np.random.uniform(-1.5, 1.5), 3.5, 8.0)
            elif cond == 2:
                scan_glucose[idx] = np.random.uniform(7.0, 11.9)
                rapid_insulin[idx] = np.random.randint(0, 8)
                carbs[idx] = np.random.randint(30, 100)
                historic_glucose[idx] = np.clip(scan_glucose[idx] - np.random.uniform(0.5, 3.0), 5.0, 12.0)
            else:
                rapid_insulin[idx] = np.random.randint(0, 12)
                carbs[idx] = np.random.randint(40, 120)
                long_insulin[idx] = np.random.randint(0, 20)
                scan_glucose[idx] = np.random.uniform(14.0, 24.0)
                historic_glucose[idx] = np.clip(scan_glucose[idx] - np.random.uniform(3.0, 7.0), 9.0, 19.0)
            selected_indices[cond].append(idx)

# Flatten and shuffle
final_indices = []
for cond in range(4):
    final_indices.extend(selected_indices[cond])
np.random.shuffle(final_indices)

baseline_glucose = baseline_glucose[final_indices]
long_insulin = long_insulin[final_indices]
carbs = carbs[final_indices]
rapid_insulin = rapid_insulin[final_indices]
scan_glucose = scan_glucose[final_indices]
historic_glucose = historic_glucose[final_indices]
n_rows = len(final_indices)

# ============================================
# STEP 7: Symptoms (unchanged)
# ============================================
def generate_symptoms_v4(glucose, rapid_insulin, carbs):
    if glucose < 4.5:
        severity = 1.0 if glucose < 3.0 else (0.7 if glucose < 3.9 else 0.3)
        return [
            0,
            np.random.choice([0, 1], p=[0.7, 0.3]),
            np.random.choice([0, 1], p=[1-severity, severity]),
            0,
            np.random.choice([0, 1], p=[1-severity, severity]),
            np.random.choice([0, 1], p=[1-severity, severity])
        ]
    elif glucose < 7.0:
        return [0,0,np.random.choice([0,1],p=[0.95,0.05]),0,np.random.choice([0,1],p=[0.95,0.05]),0]
    elif glucose < 14.0:
        severity = min(1.0, (glucose-7.0)/7.0)
        insufficient_insulin = (carbs>50 and rapid_insulin<5)
        return [
            np.random.choice([0,1],p=[1-severity,severity]),
            np.random.choice([0,1],p=[0.7,0.3]) if insufficient_insulin else 0,
            np.random.choice([0,1],p=[0.6,0.4]),
            np.random.choice([0,1],p=[0.9,0.1]) if glucose>12 else 0,
            np.random.choice([0,1],p=[1-severity,severity]),
            0
        ]
    else:
        return [np.random.choice([0,1],p=[0.05,0.95]),
                np.random.choice([0,1],p=[0.1,0.9]),
                np.random.choice([0,1],p=[0.05,0.95]),
                np.random.choice([0,1],p=[0.2,0.8]),
                np.random.choice([0,1],p=[0.05,0.95]),
                0]

symptoms = np.array([generate_symptoms_v4(g, ri, c) for g, ri, c in zip(scan_glucose, rapid_insulin, carbs)])
thirst, nausea, weakness, vomiting, fatigue, shakiness = symptoms.T

# ============================================
# STEP 8: Assign conditions (UPDATED CLINICAL VERSION)
# ============================================
def assign_condition_v5(glucose, thirst, nausea, vomiting, weakness, fatigue, shakiness):
    """
    Clinical fix: 
    - Glucose < 4.0 mmol/L → hypoglycemia (0), independent of symptoms
    - 4.0–7.0 → normal (1)
    - 7.0–12.0 → hyperglycemia (2)
    - ≥12.0 → severe hyperglycemia (3)
    """
    if glucose < 4.0:
        return 0
    elif glucose < 7.0:
        base_condition = 1
    elif glucose < 12.0:
        base_condition = 2
    else:
        base_condition = 3

    # Optional severe hyper symptom adjustment
    if 11.0 <= glucose < 13.0:
        severe_symptoms = thirst + nausea + weakness + vomiting + fatigue
        if severe_symptoms >= 4 or (severe_symptoms >= 2 and glucose >= 12.0):
            return 3

    return base_condition

condition = np.array([assign_condition_v5(g, t, n, v, w, f, s) 
                      for g,t,n,v,w,f,s in zip(scan_glucose, thirst, nausea, vomiting, weakness, fatigue, shakiness)])

# ============================================
# STEP 9: Create DataFrame
# ============================================
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

# Balance classes
counts = Counter(condition)
min_count = min(counts.values())
balanced_indices = []
for cond in range(4):
    cond_indices = np.where(condition == cond)[0]
    selected = np.random.choice(cond_indices, min_count, replace=False)
    balanced_indices.extend(selected)
df = df.iloc[balanced_indices].reset_index(drop=True)

df.to_csv('synthetic_T1_diabetes_data.csv', index=False)
print("\nFinal class distribution:")
print(df['Condition'].value_counts().sort_index())
print(f"\nTotal samples: {len(df)}")
print("\nDataset saved to 'synthetic_T1_diabetes_data_v6.csv'")
print("=" * 60)