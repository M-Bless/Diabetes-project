import pandas as pd
import numpy as np

np.random.seed(42)

def generate_cases(n=100, condition='Normal'):
    """Generate realistic synthetic medical data"""
    data = []
    
    for _ in range(n):
        
        if condition == 'Hypoglycemia':
            glucose = np.random.uniform(20, 69)
            carbs = np.random.randint(0, 40)  # LOW carbs - didn't eat enough
            insulin_recent = np.random.choice(['Yes', 'No'], p=[0.75, 0.25])  # Usually took insulin
            symptoms = np.random.choice([
                'shakiness,sweating,confusion',
                'severe_confusion,trembling,profuse_sweating',
                'disorientation,weakness,cold_sweats',
                'nervousness,sweating,weakness',
                'trembling,anxiety,dizziness',
                'confusion,anxiety,tremors'
            ])
            
        elif condition == 'Hyperglycemia':
            glucose = np.random.uniform(200, 400)
            carbs = np.random.randint(80, 150)  # HIGH carbs - ate too much
            insulin_recent = np.random.choice(['Yes', 'No'], p=[0.35, 0.65])  # Usually didn't take insulin
            symptoms = np.random.choice([
                'extreme_fatigue,polyuria,blurred_vision',
                'excessive_thirst,weakness,frequent_urination',
                'severe_thirst,frequent_urination,fatigue',
                'thirst,polyuria,mild_fatigue',
                'dry_mouth,frequent_urination,weakness',
                'fatigue,blurred_vision,headache'
            ])
            
        elif condition == 'DKA':
            glucose = np.random.uniform(250, 600)
            carbs = np.random.randint(30, 120)  # MODERATE carbs
            insulin_recent = np.random.choice(['Yes', 'No'], p=[0.2, 0.8])  # Rarely took insulin
            symptoms = np.random.choice([
                'altered_mental_status,vomiting,rapid_breathing,fruity_breath',
                'severe_abdominal_pain,vomiting,Kussmaul_breathing,altered_consciousness',
                'vomiting,weakness,Kussmaul_breathing,polyuria',
                'nausea,vomiting,rapid_breathing,abdominal_pain',
                'abdominal_pain,vomiting,rapid_breathing,thirst',
                'thirst,nausea,increased_breathing,fatigue',
                'mild_nausea,increased_urination,weakness'
            ])
            
        else:  # Normal
            glucose = np.random.uniform(70, 140)
            carbs = np.random.randint(40, 80)  # MODERATE carbs
            insulin_recent = np.random.choice(['Yes', 'No'], p=[0.5, 0.5])  # 50/50
            symptoms = np.random.choice(['None', 'None', 'None', 'Mild_fatigue'])
        
        data.append({
            'glucose_mg_dl': round(glucose, 1),
            'carbs_last_meal_grams': carbs,
            'recent_insulin': insulin_recent,
            'symptoms': symptoms,
            'condition': condition
        })
        
    return data

# Generate 500 samples per condition (2000 total for training)
normal_data = generate_cases(500, 'Normal')
hypo_data = generate_cases(500, 'Hypoglycemia')
hyper_data = generate_cases(500, 'Hyperglycemia')
dka_data = generate_cases(500, 'DKA')

# Combine and shuffle
all_data = normal_data + hypo_data + hyper_data + dka_data
df = pd.DataFrame(all_data).sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
filename = 'synthetic_diabetes_data.csv'
df.to_csv(filename, index=False)

# Print summary
print(f"✓ Saved: {filename}")
print(f"✓ Total: {len(df)} records\n")
print("Condition counts:")
print(df['condition'].value_counts())
print("\nGlucose by condition:")
print(df.groupby('condition')['glucose_mg_dl'].mean().round(1))
print("\nCarbs by condition (KEY FEATURE):")
print(df.groupby('condition')['carbs_last_meal_grams'].mean().round(1))
print("\nInsulin 'Yes' percentage by condition (KEY FEATURE):")
for cond in ['Hypoglycemia', 'Hyperglycemia', 'DKA', 'Normal']:
    pct = (df[df['condition'] == cond]['recent_insulin'] == 'Yes').mean()
    print(f"  {cond}: {pct:.0%}")