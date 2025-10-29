import numpy as np
import pandas as pd

# For reproducibility
np.random.seed(42)

# Number of synthetic samples
n_samples = 2500

# Glucose categories (based on clinical T1D ranges)
glucose_categories = ['Hypoglycemia', 'Normal', 'Hyperglycemia']

# Probability distribution (slightly realistic proportions)
glucose_probs = [0.25, 0.55, 0.20]

# Randomly assign glucose status
glucose_status = np.random.choice(glucose_categories, size=n_samples, p=glucose_probs)

# Initialize dataframe
data = pd.DataFrame({'Glucose_Status': glucose_status})

# Generate realistic continuous features based on glucose state
def generate_feature_by_status(status, hypo_range, normal_range, hyper_range):
    if status == 'Hypoglycemia':
        return np.random.uniform(*hypo_range)
    elif status == 'Normal':
        return np.random.uniform(*normal_range)
    else:
        return np.random.uniform(*hyper_range)

data['Historic Glucose (mmol/L)'] = data['Glucose_Status'].apply(
    lambda s: generate_feature_by_status(s, (2.5, 3.9), (4.0, 7.8), (7.9, 15.0))
)
data['Scan Glucose (mmol/L)'] = data['Glucose_Status'].apply(
    lambda s: generate_feature_by_status(s, (2.8, 3.9), (4.0, 7.8), (8.0, 15.5))
)
data['Rapid-Acting Insulin (units)'] = data['Glucose_Status'].apply(
    lambda s: generate_feature_by_status(s, (0, 4), (2, 8), (6, 12))
)
data['Long-Acting Insulin (units)'] = data['Glucose_Status'].apply(
    lambda s: generate_feature_by_status(s, (5, 12), (10, 25), (20, 40))
)
data['Carbohydrate Intake (g)'] = data['Glucose_Status'].apply(
    lambda s: generate_feature_by_status(s, (10, 25), (25, 60), (50, 120))
)
data['Physical Activity (min)'] = data['Glucose_Status'].apply(
    lambda s: generate_feature_by_status(s, (10, 30), (20, 60), (0, 20))
)
data['Heart Rate (bpm)'] = data['Glucose_Status'].apply(
    lambda s: generate_feature_by_status(s, (60, 75), (70, 90), (85, 110))
)
data['Blood Pressure (mmHg)'] = data['Glucose_Status'].apply(
    lambda s: generate_feature_by_status(s, (90, 110), (110, 130), (130, 160))
)
data['Sleep Duration (hrs)'] = data['Glucose_Status'].apply(
    lambda s: generate_feature_by_status(s, (4, 6), (6, 8), (5, 7))
)
data['Stress Level (1-10)'] = data['Glucose_Status'].apply(
    lambda s: generate_feature_by_status(s, (5, 9), (2, 6), (7, 10))
)

# Symptom probabilities (stronger ties to glucose category)
def symptom_prob(status, high_p, mid_p, low_p):
    if status == 'Hypoglycemia':
        return np.random.choice([1, 0], p=[high_p, 1 - high_p])
    elif status == 'Normal':
        return np.random.choice([1, 0], p=[mid_p, 1 - mid_p])
    else:
        return np.random.choice([1, 0], p=[low_p, 1 - low_p])

symptoms = {
    'Dizziness': (0.7, 0.2, 0.1),
    'Confusion': (0.6, 0.1, 0.2),
    'Sweating': (0.8, 0.3, 0.2),
    'Hunger': (0.5, 0.3, 0.4),
    'Headache': (0.4, 0.2, 0.6),
    'Fatigue': (0.5, 0.3, 0.6),
    'Blurred Vision': (0.3, 0.1, 0.7),
    'Increased Thirst': (0.3, 0.1, 0.8),
    'Frequent Urination': (0.2, 0.1, 0.9),
    'Shakiness': (0.8, 0.2, 0.1),
    'Nausea': (0.4, 0.2, 0.6),
    'Vomiting': (0.3, 0.1, 0.5),
    'Weakness': (0.6, 0.3, 0.5)
}

for symptom, probs in symptoms.items():
    data[symptom] = data['Glucose_Status'].apply(
        lambda s: symptom_prob(s, *probs)
    )

# Add slight noise to make it look natural (subtle, not noticeable)
for col in [
    'Historic Glucose (mmol/L)', 'Scan Glucose (mmol/L)', 'Heart Rate (bpm)',
    'Blood Pressure (mmHg)', 'Sleep Duration (hrs)', 'Carbohydrate Intake (g)'
]:
    data[col] = data[col] + np.random.normal(0, 0.2, size=n_samples)

# Shuffle rows
data = data.sample(frac=1).reset_index(drop=True)

# Display sample
print(data.head())

# Save synthetic dataset
data.to_csv("synthetic_diabetes_dataset.csv", index=False)
print("\n✅ Synthetic data generated successfully and saved as 'synthetic_diabetes_dataset.csv'.")
