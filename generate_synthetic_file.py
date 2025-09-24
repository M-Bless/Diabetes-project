# The clinical conditions and rules considered are gotten from:
# https://pmc.ncbi.nlm.nih.gov/articles/PMC4606130/
# https://www.sciencedirect.com/science/article/pii/S2666396121000169
# This is a synthetic data generation script for diabetes-related conditions.


import pandas as pd
import numpy as np

# Number of records
n = 500

# Generate random glucose values
glucose = np.random.randint(40, 350, n)  

# Generate random insulin doses (0–20 units)
insulin = np.random.randint(0, 20, n)  

# Random blood pressure (90–160)
bp = np.random.randint(90, 160, n)  

# Symptoms (random 0 or 1)
sweating = np.random.binomial(1, 0.5, n)  # 50% base chance
nausea = np.random.binomial(1, 0.5, n)
confusion = np.random.binomial(1, 0.5, n)
thirst = np.random.binomial(1, 0.5, n)

# Boost symptoms for high glucose (>250)
for i in range(n):
    if glucose[i] > 250:
        # Increase chance for DKA symptoms
        sweating[i] = np.random.binomial(1, 0.8)
        nausea[i] = np.random.binomial(1, 0.8)
        confusion[i] = np.random.binomial(1, 0.8)
        thirst[i] = np.random.binomial(1, 0.8)

        
# Add meal status: pre or post
meal_status = np.random.choice(["pre", "post"], n)

# Function to classify outcome
def classify(glucose, meal_status, sweating, nausea, confusion, thirst):
    # Hypoglycemia
    if glucose < 70:
        return "Hypoglycemia"
    
    # DKA: >250 + symptoms (nausea, confusion, thirst, sweating)
    elif glucose > 250 and (nausea or confusion or thirst or sweating):
        return "DKA"
    
    # Hyperglycemia depends on meal status
    elif meal_status == "pre" and glucose > 130:
        return "Hyperglycemia"
    elif meal_status == "post" and glucose > 180:
        return "Hyperglycemia"
    
    # Normal ranges depend on meal status
    elif meal_status == "pre" and 70 <= glucose <= 130:
        return "Normal"
    elif meal_status == "post" and 70 <= glucose <= 180:
        return "Normal"
    
    # Fallback (shouldn’t be reached)
    else:
        return "Normal"

# Apply classification
outcome = [
    classify(g, m, s, n, c, t) 
    for g, m, s, n, c, t in zip(glucose, meal_status, sweating, nausea, confusion, thirst)
]

# Create DataFrame
df = pd.DataFrame({
    "Glucose": glucose,
    "Insulin": insulin,
    "BloodPressure": bp,
    "Sweating": sweating,
    "Nausea": nausea,
    "Confusion": confusion,
    "Thirst": thirst,
    "MealStatus": meal_status,
    "Outcome": outcome
})

# Save to CSV
df.to_csv("synthetic_diabetes_data.csv", index=False)

print(df.head())

