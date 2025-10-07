import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def generate_normal_cases(n=100):
    """Generate normal glucose control cases (70-140 mg/dL)"""
    data = []
    for i in range(n):
        glucose = np.random.uniform(70, 140)
        
        data.append({
            'patient_id': f'NORM_{i+1:03d}',
            'glucose_mg_dl': round(glucose, 1),
            'carbs_last_meal_grams': round(np.random.uniform(40, 80), 0),  # Balanced carbs
            'recent_insulin': np.random.choice(['Yes', 'No'], p=[0.6, 0.4]),
            'symptoms': np.random.choice(['None', 'Mild_fatigue'], p=[0.85, 0.15]),
            'condition': 'Normal',
            'severity': 'None'
        })
    return data

def generate_hypoglycemia_cases(n=100):
    """Generate hypoglycemia cases (blood glucose < 70 mg/dL)"""
    data = []
    for i in range(n):
        glucose = np.random.uniform(20, 69)
        
        # Hypoglycemia often caused by: too much insulin + low carbs
        recent_insulin = np.random.choice(['Yes', 'No'], p=[0.75, 0.25])  # Usually recent insulin
        carbs = round(np.random.uniform(0, 40), 0)  # Low or no carbs
        
        # Symptoms specific to hypoglycemia
        if glucose < 54:  # Severe
            severity = 'Severe'
            symptoms = np.random.choice([
                'confusion,shakiness,sweating',
                'severe_confusion,trembling,profuse_sweating',
                'disorientation,weakness,cold_sweats',
                'confusion,anxiety,tremors'
            ])
        else:  # Mild
            severity = 'Mild'
            symptoms = np.random.choice([
                'shakiness,sweating,hunger',
                'trembling,anxiety,dizziness',
                'nervousness,sweating,weakness',
                'irritability,shakiness,mild_sweating'
            ])
        
        data.append({
            'patient_id': f'HYPO_{i+1:03d}',
            'glucose_mg_dl': round(glucose, 1),
            'carbs_last_meal_grams': carbs,
            'recent_insulin': recent_insulin,
            'symptoms': symptoms,
            'condition': 'Hypoglycemia',
            'severity': severity
        })
    return data

def generate_hyperglycemia_cases(n=100):
    """Generate hyperglycemia cases (blood glucose > 180 mg/dL, no DKA)"""
    data = []
    for i in range(n):
        glucose = np.random.uniform(200, 400)
        
        # Hyperglycemia often caused by: no/insufficient insulin + high carbs
        recent_insulin = np.random.choice(['Yes', 'No'], p=[0.25, 0.75])  # Usually no recent insulin
        carbs = round(np.random.uniform(80, 150), 0)  # High carb intake
        
        # Symptoms specific to hyperglycemia
        if glucose > 300:  # Severe
            severity = 'Severe'
            symptoms = np.random.choice([
                'severe_thirst,frequent_urination,fatigue',
                'extreme_fatigue,polyuria,blurred_vision',
                'excessive_thirst,weakness,frequent_urination',
                'fatigue,dry_mouth,polyuria'
            ])
        else:  # Moderate
            severity = 'Moderate'
            symptoms = np.random.choice([
                'increased_thirst,frequent_urination',
                'fatigue,blurred_vision,headache',
                'dry_mouth,frequent_urination,weakness',
                'thirst,polyuria,mild_fatigue'
            ])
        
        data.append({
            'patient_id': f'HYPER_{i+1:03d}',
            'glucose_mg_dl': round(glucose, 1),
            'carbs_last_meal_grams': carbs,
            'recent_insulin': recent_insulin,
            'symptoms': symptoms,
            'condition': 'Hyperglycemia',
            'severity': severity
        })
    return data

def generate_dka_cases(n=100):
    """Generate DKA cases (glucose > 250, pH < 7.3, ketones present)"""
    data = []
    for i in range(n):
        glucose = np.random.uniform(250, 600)
        
        # DKA caused by: severe insulin deficiency
        recent_insulin = np.random.choice(['Yes', 'No'], p=[0.15, 0.85])  # Rarely recent insulin
        carbs = round(np.random.uniform(30, 120), 0)  # Variable
        
        # DKA severity
        if glucose > 500:
            severity = 'Severe'
            symptoms = np.random.choice([
                'vomiting,severe_dehydration,rapid_breathing,confusion',
                'severe_abdominal_pain,vomiting,Kussmaul_breathing,altered_consciousness',
                'nausea,vomiting,confusion,labored_breathing',
                'altered_mental_status,vomiting,rapid_breathing,fruity_breath'
            ])
        elif glucose > 400:
            severity = 'Moderate'
            symptoms = np.random.choice([
                'nausea,vomiting,rapid_breathing,abdominal_pain',
                'vomiting,weakness,Kussmaul_breathing,polyuria',
                'abdominal_pain,vomiting,rapid_breathing,thirst',
                'nausea,fatigue,rapid_breathing,vomiting'
            ])
        else:
            severity = 'Mild'
            symptoms = np.random.choice([
                'nausea,vomiting,abdominal_discomfort',
                'mild_nausea,increased_urination,weakness',
                'thirst,nausea,increased_breathing,fatigue',
                'abdominal_discomfort,nausea,weakness'
            ])
        
        data.append({
            'patient_id': f'DKA_{i+1:03d}',
            'glucose_mg_dl': round(glucose, 1),
            'carbs_last_meal_grams': carbs,
            'recent_insulin': recent_insulin,
            'symptoms': symptoms,
            'condition': 'DKA',
            'severity': severity
        })
    return data

# Generate datasets
print("="*70)
print("GENERATING SIMPLIFIED DIABETES DATASET")
print("="*70)

normal_data = generate_normal_cases(100)
hypo_data = generate_hypoglycemia_cases(100)
hyper_data = generate_hyperglycemia_cases(100)
dka_data = generate_dka_cases(100)

# Combine all data
all_data = normal_data + hypo_data + hyper_data + dka_data

# Create DataFrame
df = pd.DataFrame(all_data)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
filename = 'diabetes_dataset.csv'
df.to_csv(filename, index=False)

print(f"\n✓ Dataset created successfully!")
print(f"✓ Saved as: {filename}")
print(f"\nTotal records: {len(df)}")

print("\n" + "="*70)
print("FEATURES INCLUDED (4 KEY FEATURES)")
print("="*70)
print("\n1. glucose_mg_dl (numeric)")
print("   - Continuous variable")
print("   - Range: 20-600 mg/dL")
print("   - Can be used directly in ML models")

print("\n2. carbs_last_meal_grams (numeric)")
print("   - Continuous variable")
print("   - Range: 0-150 grams")
print("   - Can be used directly in ML models")

print("\n3. recent_insulin (categorical - binary)")
print("   - Binary variable: 'Yes' or 'No'")
print("   - Encode as: Yes=1, No=0")
print("   - Example: df['recent_insulin_encoded'] = df['recent_insulin'].map({'Yes': 1, 'No': 0})")

print("\n4. symptoms (categorical)")
print("   - Multiple categories")
print("   - Needs encoding: One-hot encoding or Label encoding")
print("   - Example One-hot: pd.get_dummies(df['symptoms'])")
print("   - Example Label: LabelEncoder().fit_transform(df['symptoms'])")

print("\n" + "="*70)
print("CONDITION DISTRIBUTION")
print("="*70)
print(df['condition'].value_counts())

print("\n" + "="*70)
print("SEVERITY DISTRIBUTION")
print("="*70)
print(df.groupby(['condition', 'severity']).size())

print("\n" + "="*70)
print("RECENT INSULIN BY CONDITION")
print("="*70)
print(pd.crosstab(df['condition'], df['recent_insulin'], normalize='index') * 100)

print("\n" + "="*70)
print("SAMPLE DATA (First 10 rows)")
print("="*70)
print(df.head(10))

print("\n" + "="*70)
print("GLUCOSE STATISTICS BY CONDITION")
print("="*70)
print(df.groupby('condition')['glucose_mg_dl'].describe())

print("\n" + "="*70)
print("CARBS STATISTICS BY CONDITION")
print("="*70)
print(df.groupby('condition')['carbs_last_meal_grams'].describe())

print("\n" + "="*70)
print("ENCODING EXAMPLE CODE")
print("="*70)
print("""
# Binary encoding for recent_insulin
df['recent_insulin_encoded'] = df['recent_insulin'].map({'Yes': 1, 'No': 0})

# One-hot encoding for symptoms
symptoms_encoded = pd.get_dummies(df['symptoms'], prefix='symptom')
df_encoded = pd.concat([df, symptoms_encoded], axis=1)

# Or Label encoding for symptoms
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['symptoms_encoded'] = le.fit_transform(df['symptoms'])

# Final feature set for ML
X = df[['glucose_mg_dl', 'carbs_last_meal_grams', 'recent_insulin_encoded', 'symptoms_encoded']]
y = df['condition']
""")

print("\n" + "="*70)
print("✓ Dataset ready for machine learning!")
print("✓ Clean, focused features with clear relationships")
print("="*70)