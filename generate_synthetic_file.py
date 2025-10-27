import pandas as pd
import numpy as np

# ============================================
# IMPROVED TYPE 1 DIABETES SYNTHETIC DATA V3
# Key Changes: 
# 1. Make insulin & carbs CAUSALLY affect glucose
# 2. Ensure balanced class distribution
# 3. ADD NOISE TO THRESHOLDS (prevent overfitting)
# 4. Symptoms influence borderline cases
# ============================================

n_rows = 10000
np.random.seed(42)

print(" Generating Type 1 Diabetes Synthetic Data V3...")
print("=" * 60)

# ============================================
# STEP 1: Start with baseline glucose
# ============================================
baseline_glucose = np.round(np.random.uniform(4.0, 12.0, n_rows), 1)

# ============================================
# STEP 2: Generate insulin doses
# ============================================
long_insulin = np.random.randint(8, 21, n_rows)

# ============================================
# STEP 3: Generate carb intake
# ============================================
meal_type = np.random.choice(['none', 'snack', 'small', 'medium', 'large'], 
                             n_rows, 
                             p=[0.2, 0.2, 0.2, 0.3, 0.1])

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
# STEP 4: Calculate rapid insulin
# ============================================
rapid_insulin = np.zeros(n_rows)

for i in range(n_rows):
    correction = 0
    if baseline_glucose[i] > 8.0:
        correction = (baseline_glucose[i] - 7.0) / 2.5
    
    meal_bolus = carbs[i] / 10.0
    rapid_insulin[i] = correction + meal_bolus
    rapid_insulin[i] += np.random.normal(0, 1.0)
    rapid_insulin[i] = np.clip(rapid_insulin[i], 0, 20)

rapid_insulin = np.round(rapid_insulin, 0).astype(int)

# ============================================
# STEP 5: Calculate ACTUAL glucose
# ============================================
scan_glucose = np.zeros(n_rows)

for i in range(n_rows):
    glucose = baseline_glucose[i]
    
    carb_effect = carbs[i] * 0.03
    glucose += carb_effect
    
    insulin_effect = rapid_insulin[i] * 2.5
    glucose -= insulin_effect
    
    if long_insulin[i] < 12:
        glucose += np.random.uniform(0, 2)
    
    glucose += np.random.normal(0, 1.0)
    glucose = np.clip(glucose, 2.5, 25.0)
    
    scan_glucose[i] = np.round(glucose, 1)

# ============================================
# STEP 6: Generate historic glucose
# ============================================
historic_glucose = np.zeros(n_rows)

for i in range(n_rows):
    trend = np.random.normal(0, 1.5)
    historic_glucose[i] = np.clip(scan_glucose[i] - carbs[i] * 0.02 + rapid_insulin[i] * 1.5 + trend, 
                                   3.0, 20.0)

historic_glucose = np.round(historic_glucose, 1)

# ============================================
# STEP 6.5: CREATE DKA SCENARIOS
# ============================================
print("\n Creating realistic DKA scenarios...")

target_dka_count = int(n_rows * 0.09)
print(f"Target DKA cases: {target_dka_count}")

dka_indices = np.random.choice(n_rows, target_dka_count, replace=False)

for idx in dka_indices:
    scenario = np.random.random()
    
    if scenario < 0.4:
        rapid_insulin[idx] = 0
        long_insulin[idx] = np.random.randint(0, 10)
        
        if carbs[idx] > 50:
            scan_glucose[idx] = np.random.uniform(16.0, 24.0)
        elif carbs[idx] > 20:
            scan_glucose[idx] = np.random.uniform(14.0, 20.0)
        else:
            scan_glucose[idx] = np.random.uniform(14.0, 18.0)
    
    elif scenario < 0.7:
        long_insulin[idx] = np.random.randint(0, 8)
        rapid_insulin[idx] = np.random.randint(0, 4)
        
        if carbs[idx] < 30:
            carbs[idx] = np.random.randint(60, 120)
        
        scan_glucose[idx] = np.random.uniform(14.5, 22.0)
    
    elif scenario < 0.85:
        scan_glucose[idx] = np.random.uniform(15.0, 25.0)
        rapid_insulin[idx] = np.random.randint(5, 15)
        long_insulin[idx] = np.random.randint(10, 20)
        
        if carbs[idx] < 40:
            carbs[idx] = np.random.randint(50, 100)
    
    else:
        rapid_insulin[idx] = np.random.randint(6, 15)
        long_insulin[idx] = np.random.randint(12, 20)
        scan_glucose[idx] = np.random.uniform(14.5, 23.0)
        
        if carbs[idx] < 30:
            carbs[idx] = np.random.randint(40, 90)
    
    historic_glucose[idx] = np.clip(
        scan_glucose[idx] - np.random.uniform(3.0, 7.0),
        9.0, 19.0
    )

print(f" Created {target_dka_count} DKA scenarios")

# ============================================
# STEP 6.6: BALANCE HYPOGLYCEMIA CASES
# ============================================
print("\n Ensuring sufficient hypoglycemia cases...")

target_hypo_count = int(n_rows * 0.13)
current_hypo = (scan_glucose < 3.9).sum()
needed_hypo = target_hypo_count - current_hypo

if needed_hypo > 0:
    candidates = np.where(
        (scan_glucose >= 3.9) & 
        (scan_glucose < 5.5) & 
        (~np.isin(np.arange(n_rows), dka_indices))
    )[0]
    
    if len(candidates) >= needed_hypo:
        hypo_indices = np.random.choice(candidates, needed_hypo, replace=False)
        
        for idx in hypo_indices:
            scenario = np.random.random()
            
            if scenario < 0.5:
                rapid_insulin[idx] = np.random.randint(8, 18)
                carbs[idx] = np.random.randint(0, 30)
                scan_glucose[idx] = np.random.uniform(2.5, 3.8)
            else:
                rapid_insulin[idx] = np.random.randint(4, 12)
                carbs[idx] = 0
                scan_glucose[idx] = np.random.uniform(2.7, 3.8)
            
            historic_glucose[idx] = np.clip(
                scan_glucose[idx] + np.random.uniform(1.5, 4.0),
                4.0, 8.0
            )
        
        print(f" Created {needed_hypo} additional hypoglycemia cases")
else:
    print(f" Already have sufficient hypoglycemia cases")

# ============================================
# STEP 7: GENERATE SYMPTOMS FIRST (BEFORE CONDITIONS!)
# ============================================
print("\n Generating symptoms based on glucose patterns...")

def generate_symptoms_v3(glucose, rapid_insulin, carbs):
    """
    Generate symptoms based on glucose RANGES (not perfect conditions)
    This allows symptoms to inform condition assignment
    """
    
    if glucose < 4.5:  # Low/borderline low
        severity = 1.0 if glucose < 3.0 else (0.7 if glucose < 3.9 else 0.3)
        return [
            0,  # Thirst
            np.random.choice([0, 1], p=[0.7, 0.3]),  # Nausea
            np.random.choice([0, 1], p=[1-severity, severity]),  # Weakness
            0,  # Vomiting
            np.random.choice([0, 1], p=[1-severity, severity]),  # Fatigue
            np.random.choice([0, 1], p=[1-severity, severity])   # Shakiness
        ]
    
    elif glucose < 7.0:  # Normal range
        return [0, 0, 
                np.random.choice([0, 1], p=[0.95, 0.05]),
                0, 
                np.random.choice([0, 1], p=[0.95, 0.05]),
                0]
    
    elif glucose < 14.0:  # Hyperglycemia range
        severity = min(1.0, (glucose - 7.0) / 7.0)
        insufficient_insulin = (carbs > 50 and rapid_insulin < 5)
        
        return [
            np.random.choice([0, 1], p=[1-severity, severity]),  # Thirst
            np.random.choice([0, 1], p=[0.7, 0.3]) if insufficient_insulin else 0,
            np.random.choice([0, 1], p=[0.6, 0.4]),
            np.random.choice([0, 1], p=[0.9, 0.1]) if glucose > 12 else 0,  # Vomiting
            np.random.choice([0, 1], p=[1-severity, severity]),
            0
        ]
    
    else:  # DKA range (≥14)
        # DKA: Almost always all severe symptoms
        return [
            np.random.choice([0, 1], p=[0.05, 0.95]),  # Thirst (95% have it)
            np.random.choice([0, 1], p=[0.1, 0.9]),    # Nausea (90% have it)
            np.random.choice([0, 1], p=[0.05, 0.95]),  # Weakness (95% have it)
            np.random.choice([0, 1], p=[0.2, 0.8]),    # Vomiting (80% have it)
            np.random.choice([0, 1], p=[0.05, 0.95]),  # Fatigue (95% have it)
            0  # Shakiness (not typical in DKA)
        ]

# Generate symptoms
symptoms = np.array([
    generate_symptoms_v3(g, ri, carbs_val) 
    for g, ri, carbs_val in zip(scan_glucose, rapid_insulin, carbs)
])

thirst, nausea, weakness, vomiting, fatigue, shakiness = symptoms.T

# ============================================
# STEP 8: ASSIGN CONDITIONS WITH NOISE AND SYMPTOM INFLUENCE
# ============================================
print("\n Assigning conditions with realistic noise...")

condition = np.zeros(n_rows, dtype=int)

for i in range(n_rows):
    g = scan_glucose[i]
    
    # Add individual variation to thresholds (±0.3-0.5 mmol/L)
    # This simulates:
    # - Measurement error
    # - Individual metabolic differences
    # - Time-of-day effects
    threshold_noise = np.random.normal(0, 0.4)
    
    # Base thresholds with noise
    hypo_threshold = 3.9 + threshold_noise
    normal_upper = 7.0 + threshold_noise
    hyper_upper = 14.0 + threshold_noise
    
    # Initial assignment based on glucose
    if g < hypo_threshold:
        condition[i] = 0  # Hypoglycemia
    elif g < normal_upper:
        condition[i] = 1  # Normal
    elif g < hyper_upper:
        condition[i] = 2  # Hyperglycemia
    else:
        condition[i] = 3  # DKA
    
    # =========================================
    # SYMPTOM-BASED ADJUSTMENTS (CRITICAL!)
    # =========================================
    
    # If in borderline zones, symptoms can influence classification
    
    # Borderline Hypo/Normal (3.7-4.2 mmol/L)
    if 3.7 <= g <= 4.2:
        if weakness[i] == 1 and (shakiness[i] == 1 or fatigue[i] == 1):
            condition[i] = 0  # Symptoms suggest hypoglycemia
        else:
            condition[i] = 1  # No strong symptoms, consider normal
    
    # Borderline Normal/Hyper (6.5-7.5 mmol/L)
    if 6.5 <= g <= 7.5:
        if thirst[i] == 1 and fatigue[i] == 1:
            condition[i] = 2  # Symptoms suggest hyperglycemia
        else:
            condition[i] = 1  # Minimal symptoms, stay normal
    
    # Borderline Hyper/DKA (13.0-14.5 mmol/L)
    # This is CRITICAL - allows DKA diagnosis slightly below 14.0
    if 13.0 <= g <= 14.5:
        severe_symptoms = (thirst[i] + nausea[i] + weakness[i] + 
                          vomiting[i] + fatigue[i])
        
        if severe_symptoms >= 4:  # 4-5 symptoms present
            condition[i] = 3  # Upgrade to DKA based on clinical presentation
        elif severe_symptoms >= 2 and g >= 13.5:
            condition[i] = 3  # Moderate symptoms + high glucose = DKA
        else:
            condition[i] = 2  # Not enough symptoms, stay hyperglycemia
    
    # High glucose with severe symptoms = DKA (even if <14)
    # This handles cases like illness where symptoms appear early
    if 12.0 <= g < 14.0:
        if (thirst[i] == 1 and nausea[i] == 1 and 
            vomiting[i] == 1 and weakness[i] == 1):
            condition[i] = 3  # Clinical DKA despite glucose <14

print("Conditions assigned with noise and symptom influence")

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

# ============================================
# STEP 10: Save and verify
# ============================================
df.to_csv('synthetic_diabetes_data_v2.csv', index=False)

print("\n" + "=" * 60)
print(" REALISTIC synthetic dataset generated!")
print("   File: synthetic_diabetes_data_v2.csv")
print(f"   Total rows: {len(df)}")
print("=" * 60)

print("\n FINAL CLASS DISTRIBUTION:")
condition_counts = df['Condition'].value_counts().sort_index()
print("\nAbsolute counts:")
print(condition_counts)

print("\nPercentages:")
for cond, count in condition_counts.items():
    pct = count / len(df) * 100
    condition_name = ['Hypoglycemia', 'Normal', 'Hyperglycemia', 'DKA'][cond]
    print(f"  Class {cond} ({condition_name}): {count:4d} samples ({pct:5.1f}%)")

min_samples = condition_counts.min()
print(f"\n Minimum samples per class: {min_samples}")
if min_samples >= 2:
    print("   ✓ Safe for stratified train-test split!")

# ============================================
# VERIFY: Check glucose ranges by condition
# ============================================
print("\n Glucose Range Verification (should have OVERLAP!):")
for cond in range(4):
    condition_name = ['Hypoglycemia', 'Normal', 'Hyperglycemia', 'DKA'][cond]
    glucose_values = df[df['Condition'] == cond]['Scan Glucose (mmol/L)']
    print(f"\n{condition_name} (Class {cond}):")
    print(f"  Glucose range: {glucose_values.min():.1f} - {glucose_values.max():.1f} mmol/L")
    print(f"  Mean: {glucose_values.mean():.1f}, Median: {glucose_values.median():.1f}")

# ============================================
# VERIFY: Symptom influence on borderline cases
# ============================================
print("\n Borderline Case Analysis:")

# Cases where symptoms influenced diagnosis
borderline_dka = df[(df['Scan Glucose (mmol/L)'] >= 12.0) & 
                     (df['Scan Glucose (mmol/L)'] < 14.0) & 
                     (df['Condition'] == 3)]
print(f"\nDKA cases with glucose 12-14 (symptom-based): {len(borderline_dka)}")
if len(borderline_dka) > 0:
    print(f"  Average symptoms: {borderline_dka[['Thirst', 'Nausea', 'Weakness', 'Vomiting', 'Fatigue']].sum(axis=1).mean():.1f}/5")

borderline_hypo = df[(df['Scan Glucose (mmol/L)'] >= 3.9) & 
                      (df['Scan Glucose (mmol/L)'] <= 4.2) & 
                      (df['Condition'] == 0)]
print(f"\nHypo cases with glucose 3.9-4.2 (symptom-based): {len(borderline_hypo)}")

print("\n Data generation complete!")
print("=" * 60)
print("\n KEY IMPROVEMENTS:")
print("   1. ✓ Noisy thresholds (±0.4 mmol/L variation)")
print("   2. ✓ Symptoms influence borderline diagnoses")
print("   3. ✓ DKA possible with glucose 12-14 + severe symptoms")
print("   4. ✓ Realistic overlap between classes")
print("=" * 60)