================================================================================
  README — Type 1 Diabetes Condition Classification (final year project)
================================================================================

OVERVIEW
--------
This project builds and evaluates machine learning models that classify
blood glucose readings into one of four clinical conditions relevant to
Type 1 Diabetes (T1D):

  0 — Hypoglycemia    (glucose < 4.0 mmol/L)
  1 — Normal          (glucose 4.0 – 7.0 mmol/L)
  2 — Hyperglycemia   (glucose 7.0 – 12.0 mmol/L)
  3 — DKA             (glucose ≥ 12.0 mmol/L)

The models are trained on a synthetic T1D dataset and then evaluated
against real patient data to test generalisation.


--------------------------------------------------------------------------------
DATASETS
--------------------------------------------------------------------------------

synthetic_T1_diabetes_data.csv
  - Synthetic dataset used for model training and testing.
  - Contains labelled examples covering all four conditions.
  - Split 80/20 (train/test) with stratification.

real_data.csv  (tab-separated)
  - Real patient data used for final model evaluation.
  - Contains five raw columns:
      * Historic Glucose (mmol/L)
      * Scan Glucose (mmol/L)
      * Rapid-Acting Insulin (units)
      * Long-Acting Insulin (units)
      * Carbohydrates (grams)
  - Conditions are not pre-labelled; they are assigned by the notebook
    using clinical thresholds and symptom logic (see Feature Engineering).


--------------------------------------------------------------------------------
FEATURES
--------------------------------------------------------------------------------

Input features used by all trained models:

  1. Historic Glucose (mmol/L)
  2. Scan Glucose (mmol/L)
  3. Rapid-Acting Insulin (units)
  4. Long-Acting Insulin (units)
  5. Carbohydrates (grams)
  6. Thirst         (binary symptom flag, engineered)
  7. Nausea         (binary symptom flag, engineered)
  8. Weakness       (binary symptom flag, engineered)
  9. Vomiting       (binary symptom flag, engineered)
 10. Fatigue        (binary symptom flag, engineered)
 11. Shakiness      (binary symptom flag, engineered)


--------------------------------------------------------------------------------
NOTEBOOK STRUCTURE
--------------------------------------------------------------------------------

1. Imports & Setup
   - pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, joblib

2. Load Synthetic Data
   - Load synthetic_T1_diabetes_data.csv
   - Inspect class distribution

3. Train/Test Split
   - 80% training / 20% test, stratified by condition label

4. Load & Clean Real Data
   - Read real_data.csv (tab-delimited)
   - Handle missing values:
       * Historic Glucose zeros replaced with column median
       * Scan Glucose zeros/NaNs filled from Historic Glucose, then median
       * Insulin and carb NaNs filled with 0 (no dose / no meal recorded)

5. Feature Engineering — Symptom Assignment
   - Six binary symptom columns are created from Scan Glucose values:
       Hypo  (<4.0)        → Shakiness, Weakness, Fatigue
       Normal (4.0–7.0)    → mild Shakiness or Fatigue at boundaries
       Hyper (7.0–12.0)    → Thirst, Fatigue; Nausea ≥10.0; Weakness ≥11.0
       DKA   (≥12.0)       → Thirst, Nausea, Vomiting, Fatigue, Weakness

6. Condition Assignment (Real Data)
   - A rule-based function (assign_condition) applies the same glucose
     thresholds as the synthetic generator, with soft symptom influence
     at borderline values (e.g. 4.0–5.0, 6.5–7.5, 11.5–12.5).

7. Class Balancing
   - Real data is downsampled to the size of the smallest class so all
     four conditions contribute equally to evaluation metrics.

8. Model Training & Evaluation
   Four classifiers are trained on the synthetic data and scored on both
   the held-out synthetic test set and the balanced real data:

     a) Random Forest       (n_estimators=100)
     b) XGBoost             (n_estimators=100, max_depth=6, lr=0.1)
     c) SVM                 (RBF kernel, C=1.0, gamma='scale')
     d) Logistic Regression (multinomial, lbfgs solver)

   Each model produces:
     - Accuracy on synthetic test set
     - Accuracy on real data
     - Classification report (precision, recall, F1 per class)
     - Confusion matrix heatmap

9. Feature Analysis
   - Feature Importance (Random Forest built-in importances)
   - Ablation Study — one feature removed at a time for both Random
     Forest and XGBoost; impact on test and real-data accuracy recorded
   - Feature Group Analysis — accuracy when using subsets of features
     (Glucose Only, Insulin Only, Symptoms Only, etc.)
   - Correlation Matrix heatmap across all features

10. Model Comparison
    - Side-by-side accuracy table: synthetic test vs real data
    - Generalisation gap per model (lower = better transfer)
    - Best model identified: XGBoost

11. Save Best Model
    - XGBoost model serialised to: diabetes_xgboost_model.joblib
    - Metadata (hyperparameters, feature names, performance metrics,
      class thresholds) saved to: xgboost_metadata.pkl
    - Saved model reloaded and accuracy verified


--------------------------------------------------------------------------------
OUTPUT FILES
--------------------------------------------------------------------------------

  diabetes_xgboost_model.joblib   — Trained XGBoost classifier
  xgboost_metadata.pkl            — Model metadata dictionary


--------------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------------

  pandas
  numpy
  scikit-learn
  xgboost
  matplotlib
  seaborn
  joblib

Install with:
  pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib


--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------

1. Place both data files in the same directory as the notebook:
     synthetic_T1_diabetes_data.csv
     real_data.csv

2. Open the notebook:
     jupyter notebook diabetes_analysis_train.ipynb

3. Run all cells in order (Kernel → Restart & Run All).

4. After completion, diabetes_xgboost_model.joblib and
   xgboost_metadata.pkl will be saved in the working directory.


--------------------------------------------------------------------------------
CLINICAL GLUCOSE THRESHOLDS USED
--------------------------------------------------------------------------------

  Condition       Glucose Range (mmol/L)
  ----------      ----------------------
  Hypoglycemia    < 4.0
  Normal          4.0 – 7.0
  Hyperglycemia   7.0 – 12.0
  DKA             ≥ 12.0

  Note: These thresholds reflect the ranges used in the synthetic data
  generator. The saved model metadata records slightly wider DKA/Hyper
  boundaries (up to 14.0 mmol/L) for reference purposes.


--------------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------------

- The model is trained entirely on synthetic data. Real-world clinical
  use would require validation on a much larger, properly labelled
  patient dataset.
- Symptom flags in the real data are engineered from glucose values,
  not collected directly from patients.
- XGBoost was selected as the best model due to its higher real-data
  accuracy and lower generalisation gap compared to the other classifiers.

================================================================================
