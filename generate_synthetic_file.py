import pandas as pd
import numpy as np

# ============================================
# UPDATED TYPE 1 DIABETES SYNTHETIC DATA V4
# Glucose ranges aligned with clinical T1D targets
# ============================================

n_rows = 10000
np.random.seed(42)

print("Generating Type 1 Diabetes Synthetic Data V4...")
print("=" * 60)

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
        correction = (baseline_glucose[i] - 7.0) / 2.5
    
    meal_bolus = carbs[i] / 10.0
    rapid_insulin[i] = correction + meal_bolus
    rapid_insulin[i] += np.random.normal(0, 1.0)
    rapid_insulin[i] = np.clip(rapid_insulin[i], 0, 20)
rapid_insulin = np.round(rapid_insulin, 0).astype(int)

# ============================================
# STEP 5: Actual glucose
# ============================================
scan_glucose = np.zeros(n_rows)
for i in range(n_rows):
    glucose = baseline_glucose[i]
    glucose += carbs[i] * 0.03
    glucose -= rapid_insulin[i] * 2.5
    if long_insulin[i] < 12:
        glucose += np.random.uniform(0, 2)
    glucose += np.random.normal(0, 1.0)
    glucose = np.clip(glucose, 2.5, 25.0)
    scan_glucose[i] = np.round(glucose, 1)

# ============================================
# STEP 6: Historic glucose
# ============================================
historic_glucose = np.zeros(n_rows)
for i in range(n_rows):
    trend = np.random.normal(0, 1.5)
    historic_glucose[i] = np.clip(scan_glucose[i] - carbs[i]*0.02 + rapid_insulin[i]*1.5 + trend, 3.0, 20.0)
historic_glucose = np.round(historic_glucose, 1)

# ============================================
# STEP 6.5: DKA scenarios
# ============================================
target_dka_count = int(n_rows * 0.09)
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
    historic_glucose[idx] = np.clip(scan_glucose[idx] - np.random.uniform(3.0, 7.0), 9.0, 19.0)

# ============================================
# STEP 6.6: Hypoglycemia balance
# ============================================
target_hypo_count = int(n_rows * 0.13)
current_hypo = (scan_glucose < 4.0).sum()  # clinical hypoglycemia
needed_hypo = target_hypo_count - current_hypo

if needed_hypo > 0:
    candidates = np.where(
        (scan_glucose >= 4.0) & 
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
                scan_glucose[idx] = np.random.uniform(2.5, 3.9)
            else:
                rapid_insulin[idx] = np.random.randint(4, 12)
                carbs[idx] = 0
                scan_glucose[idx] = np.random.uniform(2.7, 3.9)
            historic_glucose[idx] = np.clip(scan_glucose[idx] + np.random.uniform(1.5, 4.0), 4.0, 8.0)

# ============================================
# STEP 7: Symptoms
# ============================================
def generate_symptoms_v4(glucose, rapid_insulin, carbs):
    if glucose < 4.5:  # low/borderline
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
# STEP 8: Assign conditions using fixed thresholds
# ============================================
def assign_condition_v4(glucose, thirst, nausea, vomiting, weakness, fatigue, shakiness):
    hypo_threshold = 4.0
    normal_upper = 7.0
    hyper_upper = 12.0
    # Base
    if glucose < hypo_threshold:
        base_condition = 0
    elif glucose < normal_upper:
        base_condition = 1
    elif glucose < hyper_upper:
        base_condition = 2
    else:
        base_condition = 3
    # Borderline adjustments
    if 3.7 <= glucose <= 4.2:
        if weakness==1 and (shakiness==1 or fatigue==1):
            return 0
        else:
            return 1
    if 6.5 <= glucose <= 7.5:
        if thirst==1 and fatigue==1:
            return 2
        else:
            return 1
    if 11.0 <= glucose <= 12.5:
        severe_symptoms = thirst+nausea+weakness+vomiting+fatigue
        if severe_symptoms>=4 or (severe_symptoms>=2 and glucose>=12.0):
            return 3
        else:
            return 2
    if 10.5 <= glucose < 12.0:
        if thirst==1 and nausea==1 and vomiting==1 and weakness==1:
            return 3
    return base_condition

condition = np.array([assign_condition_v4(g, t, n, v, w, f, s) 
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

df.to_csv('synthetic_diabetes_data.csv', index=False)
print("Synthetic dataset generated with updated glucose ranges.")
