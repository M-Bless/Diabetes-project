import pandas as pd
import numpy as np

#  Step 1: Basic setup
n_rows = 10000
np.random.seed(42)

#  Step 2: Generate numeric features (realistic ranges)
historic_glucose = np.round(np.random.uniform(3.0, 20.0, n_rows), 1)
scan_glucose = np.round(historic_glucose + np.random.normal(0, 1.5, n_rows), 1)
rapid_insulin = np.random.randint(0, 21, n_rows)
long_insulin = np.random.randint(8, 21, n_rows)
carbs = np.random.randint(0, 151, n_rows)

# Step 3: Define 4 glucose-based conditions
# 0 = Hypoglycemia (<4)
# 1 = Normal (4–6.9)
# 2 = Hyperglycemia (7–13.9)
# 3 = DKA (≥14)
condition = []
for g in scan_glucose:
    if g < 4:
        condition.append(0)
    elif 4 <= g < 7:
        condition.append(1)
    elif 7 <= g < 14:
        condition.append(2)
    else:
        condition.append(3)
condition = np.array(condition)

#  Step 4: Symptom generation logic per condition
def generate_symptoms(cond):
    if cond == 0:  # Hypoglycemia
        return [
            0,  # Thirst
            0,  # Nausea
            np.random.choice([0, 1], p=[0.3, 0.7]),  # Weakness
            0,  # Vomiting
            np.random.choice([0, 1], p=[0.2, 0.8]),  # Fatigue
            np.random.choice([0, 1], p=[0.3, 0.7])   # Shakiness
        ]
    elif cond == 1:  # Normal
        return [0, 0, 0, 0, 0, 0]
    elif cond == 2:  # Hyperglycemia
        return [
            np.random.choice([0, 1], p=[0.3, 0.7]),  # Thirst
            np.random.choice([0, 1], p=[0.7, 0.3]),  # Nausea
            np.random.choice([0, 1], p=[0.4, 0.6]),  # Weakness
            0,  # Vomiting
            np.random.choice([0, 1], p=[0.5, 0.5]),  # Fatigue
            0   # Shakiness
        ]
    else:  # DKA
        return [1, 1, 1, 1, 1, 0]

# Step 5: Generate symptom data
symptoms = np.array([generate_symptoms(c) for c in condition])
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

#  Step 7: Save to CSV
df.to_csv('synthetic_diabetes_data.csv', index=False)

#  Step 8: Print summary
print("Synthetic dataset generated: synthetic_diabetes_data.csv")
print(df['Condition'].value_counts().sort_index())
print("\nCondition mapping:")
print("0 = Hypoglycemia (<4 mmol/L)")
print("1 = Normal (4–6.9 mmol/L)")
print("2 = Hyperglycemia (7–13.9 mmol/L)")
print("3 = DKA (≥14 mmol/L)")
