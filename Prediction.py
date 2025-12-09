#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Final cleaned version (no conceptual changes)
"""

# ======================
# STEP 0 – IMPORTS
# ======================
import os
import re
import numpy as np
import pandas as pd

from IPython.display import display

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import joblib

DATA_DIR = "MIMIC"

# ======================
# STEP 1 – CORE TABLES
# ======================
patients      = pd.read_csv(os.path.join(DATA_DIR, "PATIENTS.csv"))
admissions    = pd.read_csv(os.path.join(DATA_DIR, "ADMISSIONS.csv"))
icustays      = pd.read_csv(os.path.join(DATA_DIR, "ICUSTAYS.csv"))
drgcodes      = pd.read_csv(os.path.join(DATA_DIR, "DRGCODES.csv"))
diagnoses_icd = pd.read_csv(os.path.join(DATA_DIR, "DIAGNOSES_ICD.csv"))
d_icd_diag    = pd.read_csv(os.path.join(DATA_DIR, "D_ICD_DIAGNOSES.csv"))
chartevents   = pd.read_csv(os.path.join(DATA_DIR, "CHARTEVENTS.csv"))
labevents     = pd.read_csv(os.path.join(DATA_DIR, "LABEVENTS.csv"))

# ===================================================================
# STEP 2 – ICU COHORT (ADULT, FIRST ICU, DOB FIX, DEMOGRAPHICS)
# ===================================================================

# 1) Ensure datetime types
admissions["admittime"] = pd.to_datetime(admissions["admittime"])
patients["dob"]         = pd.to_datetime(patients["dob"])
icustays["intime"]      = pd.to_datetime(icustays["intime"])
icustays["outtime"]     = pd.to_datetime(icustays["outtime"])

# 2) Merge DOB into admissions
adm = admissions.merge(
    patients[["subject_id", "dob"]],
    on="subject_id",
    how="left"
)

print("Columns in merged admissions (adm):")
print(adm.columns)

# 3) DOB correction logic (MIMIC-style date shifting)
yrs = adm["dob"].dt.year

# RULE 1: If DOB year < 1900 → add +200
mask_200 = yrs < 1900
adm.loc[mask_200, "dob"] = adm.loc[mask_200, "dob"].apply(
    lambda d: d.replace(year=d.year + 200) if pd.notnull(d) else d
)

# RULE 2: If DOB year < 2000 → add +100
yrs2 = adm["dob"].dt.year
mask_100 = yrs2 < 2000
adm.loc[mask_100, "dob"] = adm.loc[mask_100, "dob"].apply(
    lambda d: d.replace(year=d.year + 100) if pd.notnull(d) else d
)

# Final: recompute age
adm["age"] = (adm["admittime"] - adm["dob"]).dt.days / 365.25
adm.loc[(adm["age"] < 0) | (adm["age"] > 120), "age"] = np.nan

print("DOB correction done.")
print("Valid ages:", adm["age"].between(0, 120).sum())
print("Invalid ages:", adm["age"].isna().sum())

# 4) Adults only
adm = adm[adm["age"] >= 18].copy()

# 5) First ICU stay per hospital admission
icustays_clean = (
    icustays
    .sort_values(["subject_id", "hadm_id", "intime"])
    .drop_duplicates(["subject_id", "hadm_id"], keep="first")
)

# ===================================================================
# STEP 3 – DEMOGRAPHICS & ADMISSION FEATURES
# ===================================================================
# 6) Merge ICU stays with corrected admissions
cohort = icustays_clean.merge(
    adm,
    on=["subject_id", "hadm_id"],
    how="inner"
)

# 7) ICU LOS in hours
cohort["icu_los_hours"] = (
    (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600.0
)

# Optional: remove ultra-short stays (< 6 hours)
cohort = cohort[cohort["icu_los_hours"] >= 6].copy()

# 8) Add gender + expire_flag (from PATIENTS)
cohort = cohort.merge(
    patients[["subject_id", "gender", "expire_flag"]],
    on="subject_id",
    how="left"
)

# 9) Keep a clean demographic/admission view
cohort_demo = cohort[[
    "subject_id", "hadm_id", "icustay_id",
    "age", "gender", "expire_flag",
    "admission_type", "admission_location",
    "discharge_location", "insurance",
    "language", "marital_status", "ethnicity",
    "first_careunit",
    "intime", "outtime"
]].copy()

# ===================================================================
# STEP 4 – DISEASE LABEL FROM DRGCODES (WITH NEURO_RENAL MERGED)
# ===================================================================
drg_main = (
    drgcodes
    .sort_values(["hadm_id", "drg_severity"], ascending=[True, False])
    .drop_duplicates(["hadm_id"], keep="first")
)[["hadm_id", "drg_code", "description"]]

cohort_label = cohort_demo.merge(drg_main, on="hadm_id", how="left")

def map_drg_to_group(desc):
    desc = str(desc).lower()

    # Respiratory
    if any(k in desc for k in [
        "respiratory", "lung", "pneumonia", "copd", "asthma",
        "pulmonary", "ventilator", "resp failure"
    ]):
        return "RESP"

    # Cardiac
    if any(k in desc for k in [
        "cardiac", "heart", "myocardial infarction", "mi",
        "ischemia", "angina", "heart failure", "chf", "arrhythmia"
    ]):
        return "CARDIO"

    # NEURO + RENAL merged
    if any(k in desc for k in [
        "neuro", "stroke", "cerebral", "intracranial",
        "encephalopathy", "seizure", "subdural", "brain"
    ]):
        return "NEURO_RENAL"

    if any(k in desc for k in [
        "renal", "kidney", "nephro", "dialysis"
    ]):
        return "NEURO_RENAL"

    # GI / hepatic
    if any(k in desc for k in [
        "gastro", "gi ", "intestinal", "bowel", "abdomen", "abdominal",
        "hepatic", "liver", "cholecyst", "pancreat", "biliary"
    ]):
        return "GI_HEP"

    # Sepsis / infection
    if any(k in desc for k in [
        "sepsis", "septic", "bacteremia", "septicemia", "septic shock"
    ]):
        return "SEPSIS"

    return "OTHER"

cohort_label["disease_group"] = cohort_label["description"].apply(map_drg_to_group)
print(cohort_label["disease_group"].value_counts())

# -------------------------------------------------------------------
# STEP 4b – NEURO vs RENAL SUBGROUP (USED BY 2nd-STAGE MODEL)
# -------------------------------------------------------------------
def map_drg_to_subgroup(desc):
    desc = str(desc).lower()

    if any(k in desc for k in ["renal", "kidney", "nephro", "dialysis"]):
        return "RENAL"

    if any(k in desc for k in [
        "neuro", "stroke", "cerebral", "intracranial",
        "encephalopathy", "seizure", "subdural", "brain"
    ]):
        return "NEURO"

    return None

cohort_label["disease_subgroup"] = cohort_label["description"].apply(map_drg_to_subgroup)

print("Top-level groups:")
print(cohort_label["disease_group"].value_counts())
print("\nSubgroups (NEURO/RENAL):")
print(cohort_label["disease_subgroup"].value_counts(dropna=False))

# ===================================================================
# SAVE RAW MERGED TABLE
# ===================================================================
raw_final = cohort_label.copy()
raw_final.to_csv("Final.csv", index=False)
print("Saved Final.csv with shape:", raw_final.shape)

# ===================================================================
# STEP 5 – DIAGNOSIS TEXT → TF-IDF + SVD FEATURES
# ===================================================================
diag_long = diagnoses_icd.merge(
    d_icd_diag[["icd9_code", "long_title"]],
    on="icd9_code",
    how="left"
)

diag_text = (
    diag_long
    .groupby("hadm_id")["long_title"]
    .apply(lambda x: " ; ".join(str(t) for t in x))
    .reset_index()
    .rename(columns={"long_title": "diag_text"})
)

cohort_label = cohort_label.merge(diag_text, on="hadm_id", how="left")

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

cohort_label["diag_text_clean"] = cohort_label["diag_text"].fillna("").apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000)
svd   = TruncatedSVD(n_components=50, random_state=42)
norm  = Normalizer()

tfidf_matrix = tfidf.fit_transform(cohort_label["diag_text_clean"])
diag_svd = svd.fit_transform(tfidf_matrix)
diag_svd = norm.fit_transform(diag_svd)

for i in range(diag_svd.shape[1]):
    cohort_label[f"diag_svd_{i}"] = diag_svd[:, i]

# ===================================================================
# STEP 6 – FIRST-24h VITALS (CHARTEVENTS)
# ===================================================================
chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])
ce = chartevents.merge(
    cohort_label[["icustay_id", "intime"]],
    on="icustay_id",
    how="inner"
)

ce["hours_from_icu"] = (ce["charttime"] - ce["intime"]).dt.total_seconds() / 3600.0
ce24 = ce[(ce["hours_from_icu"] >= 0) & (ce["hours_from_icu"] <= 24)].copy()

vital_ids = {
    "HR":   [211, 220045],
    "SBP":  [51, 442, 455, 220050, 220179],
    "DBP":  [8368, 8440, 8502, 220051, 220180],
    "MAP":  [456, 52, 220052, 220181],
    "RESP": [618, 220210],
    "TEMP": [676, 678, 223761, 223762],
    "SPO2": [646, 220277, 228232],
}

vital_agg_list = []
for name, ids in vital_ids.items():
    sub = ce24[ce24["itemid"].isin(ids)].copy()
    sub = sub[(sub["valuenum"].notna()) & (sub["valuenum"] > 0)]
    agg = (
        sub.groupby("icustay_id")["valuenum"]
        .agg(["mean", "min", "max"])
        .rename(columns={
            "mean": f"{name}_mean_24h",
            "min":  f"{name}_min_24h",
            "max":  f"{name}_max_24h",
        })
    )
    vital_agg_list.append(agg)

vitals_24h = pd.concat(vital_agg_list, axis=1)

# ===================================================================
# STEP 7 – FIRST-24h LABS (LABEVENTS)
# ===================================================================
labevents["charttime"] = pd.to_datetime(labevents["charttime"])

lab = labevents.merge(
    cohort_label[["hadm_id", "intime"]],
    on="hadm_id",
    how="inner"
)

lab["hours_from_icu"] = (lab["charttime"] - lab["intime"]).dt.total_seconds() / 3600.0
lab24 = lab[(lab["hours_from_icu"] >= 0) & (lab["hours_from_icu"] <= 24)].copy()

lab_ids = {
    "WBC":      [51300],
    "HGB":      [51222],
    "PLATELET": [51265],
    "CREAT":    [50912],
    "BUN":      [51006],
    "NA":       [50983],
    "K":        [50971],
    "LACTATE":  [50813],
}

lab_agg_list = []
for name, ids in lab_ids.items():
    sub = lab24[lab24["itemid"].isin(ids)].copy()
    sub = sub[(sub["valuenum"].notna()) & (sub["valuenum"] > 0)]
    agg = (
        sub.groupby("hadm_id")["valuenum"]
        .agg(["mean", "min", "max"])
        .rename(columns={
            "mean": f"{name}_mean_24h",
            "min":  f"{name}_min_24h",
            "max":  f"{name}_max_24h",
        })
    )
    lab_agg_list.append(agg)

labs_24h = pd.concat(lab_agg_list, axis=1)

# ===================================================================
# STEP 8 – MERGE EVERYTHING INTO final_df
# ===================================================================
final_df = (
    cohort_label
    .set_index("icustay_id")
    .join(vitals_24h, how="left")
    .reset_index()
)

final_df = (
    final_df
    .set_index("hadm_id")
    .join(labs_24h, how="left")
    .reset_index()
)

id_cols = ["subject_id", "hadm_id", "icustay_id"]
text_cols = ["diag_text", "diag_text_clean", "description"]

for c in text_cols:
    if c in final_df.columns:
        final_df = final_df.drop(columns=c)

# Drop datetime columns
dt_cols = final_df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
print("Dropping datetime columns:", list(dt_cols))
final_df = final_df.drop(columns=dt_cols)

target_col = "disease_group"

categorical_cols = final_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in id_cols + [target_col]]

numeric_cols = final_df.select_dtypes(include=["number"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in id_cols + [target_col]]

print("Categorical cols:", categorical_cols)
print("Numeric cols:", len(numeric_cols), "numeric features")

X = final_df[categorical_cols + numeric_cols].copy()
y = final_df[target_col].copy()

print("Any datetime left in X?", X.dtypes[X.dtypes == "datetime64[ns]"])

# ===================================================================
# STEP 9 – PREPROCESSING TRANSFORMER (NO MODEL YET)
# ===================================================================
numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

print(final_df["disease_group"].value_counts())
print(final_df["disease_group"].value_counts(normalize=True))

# ===================================================================
# STEP 10 – SUBJECT-WISE SPLIT + MANUAL CLASS WEIGHTS
# (on leak-free features, with encoded label)
# ===================================================================

# 10.0 – Ensure label is numeric
if "disease_group_encoded" not in final_df.columns:
    label_encoder = LabelEncoder()
    final_df["disease_group_encoded"] = label_encoder.fit_transform(final_df["disease_group"])
else:
    label_encoder = LabelEncoder()
    final_df["disease_group_encoded"] = label_encoder.fit_transform(final_df["disease_group"])

target_col = "disease_group_encoded"

# 10.1 – Drop any datetime & all-NaN columns (safety pass)
dt_cols = final_df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
final_df = final_df.drop(columns=dt_cols)

all_nan_cols = final_df.columns[final_df.isna().all()].tolist()
final_df = final_df.drop(columns=all_nan_cols)

print("Dropped datetime columns:", list(dt_cols))
print("Dropped all-NaN columns:", all_nan_cols)

# 10.2 – Subject-wise split
id_cols = ["subject_id", "hadm_id", "icustay_id"]
unique_subjects = final_df["subject_id"].unique()

train_subj, test_subj = train_test_split(
    unique_subjects,
    test_size=0.2,
    random_state=42
)

train_mask = final_df["subject_id"].isin(train_subj)
test_mask  = final_df["subject_id"].isin(test_subj)

df_train = final_df.loc[train_mask].reset_index(drop=True)
df_test  = final_df.loc[test_mask].reset_index(drop=True)

print("Train rows:", df_train.shape[0], " Test rows:", df_test.shape[0])

# 10.3 – Build X, y and column lists
categorical_cols = df_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in id_cols + [target_col]]

numeric_cols = df_train.select_dtypes(include=["number"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in id_cols + [target_col]]

print("Categorical columns:", categorical_cols)
print("Numeric columns:", len(numeric_cols), "numeric features")

X_train = df_train[categorical_cols + numeric_cols].copy()
y_train = df_train[target_col].copy()

X_test  = df_test[categorical_cols + numeric_cols].copy()
y_test  = df_test[target_col].copy()

# 10.4 – Preprocessing for modeling
numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# 10.5 – Manual class weights
class_counts = y_train.value_counts().to_dict()
print("Class counts in y_train:", class_counts)

max_count = max(class_counts.values())
encoded_class_weight = {
    cls: max_count / cnt
    for cls, cnt in class_counts.items()
}
print("Manual class weights (encoded):", encoded_class_weight)

# 10.6 – Define models
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight=encoded_class_weight
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight=encoded_class_weight
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42
    )
}

# 10.7 – Train & evaluate
results = []
best_model_name = None
best_model_pipe = None
best_macro_f1 = -1.0

for name, clf in models.items():
    print("\n" + "=" * 60)
    print(f"Training model (subject-wise, manual weights): {name}")
    print("=" * 60)

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf),
    ])

    if name == "XGBoost":
        sample_weight = np.array([encoded_class_weight[int(c)] for c in y_train])
        pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"{name} Accuracy:   {acc:.4f}")
    print(f"{name} Macro F1:   {macro_f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    results.append((name, acc, macro_f1))

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_model_name = name
        best_model_pipe = pipe

print("\nSummary of subject-wise, class-weighted models:")
for name, acc, macro_f1 in results:
    print(f"{name:25s}  Acc: {acc:.4f}  Macro F1: {macro_f1:.4f}")

print("\n✅ Best model (subject-wise + manual class weights):",
      best_model_name, "with Macro F1 =", best_macro_f1)

# 10.8 – Plot comparison graphs
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "MacroF1"])
print("\nResults table:\n", results_df)

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["MacroF1"])
plt.title("Model Macro F1 Comparison")
plt.ylabel("Macro F1 Score")
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

x = range(len(results_df))
plt.figure(figsize=(10, 6))
plt.bar([p - 0.2 for p in x], results_df["Accuracy"], width=0.4, label="Accuracy")
plt.bar([p + 0.2 for p in x], results_df["MacroF1"], width=0.4, label="Macro F1")
plt.xticks(x, results_df["Model"], rotation=45)
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# ===================================================================
# STEP 11 – SAVE THE BEST MODEL
# ===================================================================
final_model = best_model_pipe
final_model_name = best_model_name

print("Saving final disease model:", final_model_name)

joblib.dump(final_model, "final_disease_model.joblib")
joblib.dump(label_encoder, "disease_label_encoder.joblib")

meta = {
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "id_cols": id_cols,
    "target_col": target_col,
    "final_model_name": final_model_name
}
joblib.dump(meta, "final_disease_model_meta.joblib")

print("Saved: final_disease_model.joblib, disease_label_encoder.joblib, final_disease_model_meta.joblib")

# ===================================================================
# STEP 12 – SECOND-STAGE MODEL: NEURO vs RENAL INSIDE NEURO_RENAL
# ===================================================================
nr_mask = final_df["disease_subgroup"].isin(["NEURO", "RENAL"])
nr_df = final_df.loc[nr_mask].copy()

print("NEURO/RENAL rows:", nr_df.shape[0])
print(nr_df["disease_subgroup"].value_counts())

if nr_df.shape[0] >= 4:
    nr_label_encoder = LabelEncoder()
    nr_df["nr_encoded"] = nr_label_encoder.fit_transform(nr_df["disease_subgroup"])

    X_nr = nr_df[categorical_cols + numeric_cols]
    y_nr = nr_df["nr_encoded"]

    X_nr_train, X_nr_test, y_nr_train, y_nr_test = train_test_split(
        X_nr, y_nr,
        test_size=0.3,
        random_state=42,
        stratify=y_nr
    )

    nr_model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )),
    ])

    print("Training NEURO vs RENAL model...")
    nr_model.fit(X_nr_train, y_nr_train)

    y_nr_pred = nr_model.predict(X_nr_test)
    nr_acc = accuracy_score(y_nr_test, y_nr_pred)
    nr_f1 = f1_score(y_nr_test, y_nr_pred, average="macro")

    print(f"\nNEURO/RENAL – Accuracy: {nr_acc:.4f}")
    print(f"NEURO/RENAL – Macro F1: {nr_f1:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_nr_test, y_nr_pred, target_names=nr_label_encoder.classes_))

    joblib.dump(nr_model, "neuro_renal_model.joblib")
    joblib.dump(nr_label_encoder, "neuro_renal_label_encoder.joblib")
    print("\nSaved neuro_renal_model.joblib and neuro_renal_label_encoder.joblib")
else:
    print("Not enough NEURO/RENAL rows to train a reliable second-stage model.")

# ===================================================================
# STEP 13 – HIERARCHICAL PREDICTION HELPER
# ===================================================================
top_model = joblib.load("final_disease_model.joblib")
top_label_encoder = joblib.load("disease_label_encoder.joblib")
nr_model = joblib.load("neuro_renal_model.joblib")
nr_label_encoder = joblib.load("neuro_renal_label_encoder.joblib")
meta = joblib.load("final_disease_model_meta.joblib")

categorical_cols = meta["categorical_cols"]
numeric_cols = meta["numeric_cols"]

def predict_disease_hierarchical(row_df, prob_threshold: float = 0.40):
    """
    row_df: single-row DataFrame with same feature columns (categorical+numeric).

    Logic:
      1. Use the top-level model to get class probabilities.
      2. If max probability < prob_threshold -> label as NORMAL.
      3. Else:
           - use the top predicted disease_group
           - if it's NEURO_RENAL, run the second-stage model
    Returns:
        top_group: main disease_group (CARDIO, RESP, GI_HEP, OTHER, NEURO_RENAL, or NORMAL)
        refined_group: if NEURO_RENAL and confident, one of {NEURO, RENAL};
                       if low confidence, "NORMAL";
                       otherwise same as top_group.
    """
    # Ensure correct column order
    row_df = row_df[categorical_cols + numeric_cols]

    # --- 1) Probabilities from top-level model ---
    proba = top_model.predict_proba(row_df)[0]   # shape: (num_classes,)
    max_prob = float(proba.max())
    best_idx = int(proba.argmax())

    raw_top_group = top_label_encoder.inverse_transform([best_idx])[0]

    # Default values
    top_group = raw_top_group
    refined_group = raw_top_group

    # --- 2) LOW CONFIDENCE → NORMAL ---
    if max_prob < prob_threshold:
        top_group = "NORMAL"
        refined_group = "NORMAL"
        return top_group, refined_group

    # --- 3) If confident NEURO_RENAL → refine to NEURO vs RENAL ---
    if raw_top_group == "NEURO_RENAL" and nr_model is not None and nr_label_encoder is not None:
        nr_pred_encoded = nr_model.predict(row_df)[0]
        refined_group = nr_label_encoder.inverse_transform([nr_pred_encoded])[0]

    return top_group, refined_group


# ===================================================================
# MEDICINE RECOMMENDER
# ===================================================================

# STEP 13.1 – JOIN PRESCRIPTIONS WITH COHORT & FILTER TO FIRST 24H
prescriptions_path = os.path.join(DATA_DIR, "PRESCRIPTIONS.csv")
prescriptions = pd.read_csv(prescriptions_path)

print("PRESCRIPTIONS shape:", prescriptions.shape)
print("Columns:", prescriptions.columns.tolist()[:20])

for col in ["startdate", "enddate"]:
    if col in prescriptions.columns:
        prescriptions[col] = pd.to_datetime(prescriptions[col], errors="coerce")

cols_needed = ["hadm_id", "icustay_id", "intime", "disease_group"]
print("Check cohort_label columns:", cohort_label.columns.tolist())

cohort_for_drugs = cohort_label[cols_needed].drop_duplicates("hadm_id")

pres_cohort = prescriptions.merge(
    cohort_for_drugs,
    on="hadm_id",
    how="inner"
)

print("Joined PRESCRIPTIONS with cohort:", pres_cohort.shape)

pres_cohort["hours_from_icu"] = (
    pres_cohort["startdate"] - pres_cohort["intime"]
).dt.total_seconds() / 3600.0

early_drugs = pres_cohort[
    (pres_cohort["hours_from_icu"] >= 0) &
    (pres_cohort["hours_from_icu"] <= 24)
].copy()

print("early_drugs shape (0–24h):", early_drugs.shape)
print("Example rows:")
display(early_drugs.head())

# STEP 13.2 – BUILD DISEASE_GROUP → TOP DRUGS TABLE
drug_col = None
for candidate in ["drug_name_generic", "drug"]:
    if candidate in early_drugs.columns:
        drug_col = candidate
        break

if drug_col is None:
    raise ValueError("No suitable drug column found in PRESCRIPTIONS (expected 'drug_name_generic' or 'drug').")

print("Using drug column:", drug_col)

early_drugs = early_drugs[early_drugs[drug_col].notna()].copy()
early_drugs[drug_col] = (
    early_drugs[drug_col]
    .astype(str)
    .str.strip()
    .str.lower()
)

drug_stats = (
    early_drugs
    .groupby(["disease_group", drug_col])
    .size()
    .reset_index(name="count")
)

drug_stats["group_total"] = drug_stats.groupby("disease_group")["count"].transform("sum")
drug_stats["freq_within_group"] = drug_stats["count"] / drug_stats["group_total"]

print("Drug stats sample:")
display(drug_stats.head())

TOP_N = 10

top_drugs_per_group = (
    drug_stats
    .sort_values(["disease_group", "freq_within_group"], ascending=[True, False])
    .groupby("disease_group")
    .head(TOP_N)
    .reset_index(drop=True)
)

print("Top drugs per group (head):")
display(top_drugs_per_group.head(20))

# STEP 13.3 – BUILD LOOKUP DICT & SAVE
drug_map = {
    dg: [
        {
            "drug": row[drug_col],
            "count": int(row["count"]),
            "freq": float(row["freq_within_group"])
        }
        for _, row in grp.sort_values("freq_within_group", ascending=False).iterrows()
    ]
    for dg, grp in top_drugs_per_group.groupby("disease_group")
}

print("Example CARDIO entry:")
print(drug_map.get("CARDIO", [])[:5])

joblib.dump(drug_map, "disease_group_top_drugs.joblib")
print("Saved disease_group_top_drugs.joblib")

# STEP 13.4 – INFERENCE: PATIENT → DISEASE → MEDS
disease_model = joblib.load("final_disease_model.joblib")
disease_label_encoder = joblib.load("disease_label_encoder.joblib")
meta = joblib.load("final_disease_model_meta.joblib")
drug_map = joblib.load("disease_group_top_drugs.joblib")

categorical_cols = meta["categorical_cols"]
numeric_cols = meta["numeric_cols"]
id_cols = meta["id_cols"]
target_col = meta["target_col"]
final_model_name = meta.get("final_model_name", "Unknown")

print("Loaded model:", final_model_name)
print("Using categorical_cols:", len(categorical_cols), " numeric_cols:", len(numeric_cols))

def recommend_meds_for_patient(
    patient_row: pd.DataFrame,
    top_n_classes: int = 2,
    top_n_drugs: int = 5,
    prob_threshold: float = 0.40,
):
    """
    patient_row: DataFrame with a single row, containing the same
                 categorical + numeric columns used in training (no IDs, no target).

    Behavior:
      - Compute class probabilities from the saved disease_model.
      - If max probability < prob_threshold → treat as NORMAL and do not recommend meds.
      - Else:
          * use top_n_classes disease groups (as before)
          * get top_n_drugs per group from drug_map

    Returns:
        dict with:
          - "disease_predictions": list of dicts:
                { "disease_group": <str>,
                  "prob": <float>,
                  "raw_top_group": <str> (only for NORMAL case) }
          - "recommendations": dict[disease_group] -> list of {drug, freq, count}
    """
    # Ensure correct column subset and order
    patient_X = patient_row[categorical_cols + numeric_cols].copy()

    # Probabilities over encoded disease labels
    proba = disease_model.predict_proba(patient_X)[0]   # shape: (num_classes,)
    max_prob = float(proba.max())
    best_idx = int(proba.argmax())

    # This is the model's true top class, regardless of our NORMAL override
    raw_top_group = disease_label_encoder.inverse_transform([best_idx])[0]

    # ---------------------------------
    # CASE 1: LOW CONFIDENCE → NORMAL
    # ---------------------------------
    if max_prob < prob_threshold:
        disease_predictions = [
            {
                "disease_group": "NORMAL",
                "prob": max_prob,
                "raw_top_group": raw_top_group,  # what the model would have picked
            }
        ]

        # No meds for NORMAL
        recommendations = {
            "NORMAL": []
        }

        return {
            "disease_predictions": disease_predictions,
            "recommendations": recommendations
        }

    # ---------------------------------
    # CASE 2: CONFIDENT IN SOME DISEASE GROUP
    # (use top_n_classes as before)
    # ---------------------------------
    class_indices = np.argsort(proba)[::-1][:top_n_classes]
    disease_groups = disease_label_encoder.inverse_transform(class_indices)
    disease_probs = proba[class_indices]

    disease_predictions = [
        {"disease_group": dg, "prob": float(p)}
        for dg, p in zip(disease_groups, disease_probs)
    ]

    # Fetch top meds for each predicted disease_group
    recommendations = {}
    for dg in disease_groups:
        meds_for_group = drug_map.get(dg, [])
        recommendations[dg] = meds_for_group[:top_n_drugs]

    return {
        "disease_predictions": disease_predictions,
        "recommendations": recommendations
    }


# STEP 13.5 – DEMO ON ONE TEST PATIENT
example_row = df_test.sample(1, random_state=123)
example_features = example_row.drop(columns=id_cols + [target_col])

result = recommend_meds_for_patient(example_features, top_n_classes=2, top_n_drugs=5)

true_encoded = int(example_row[target_col].iloc[0])
true_label = disease_label_encoder.inverse_transform([true_encoded])[0]
print("True disease_group:", true_label)

print("\nPredicted disease groups (with probabilities):")
for d in result["disease_predictions"]:
    print(f"  {d['disease_group']}: {d['prob']:.3f}")

print("\nRecommended medicines:")
for dg, meds in result["recommendations"].items():
    print(f"\nFor disease_group = {dg}:")
    if not meds:
        print("  No historical meds found.")
    else:
        for m in meds:
            print(f"  {m['drug']}  (freq={m['freq']:.2f}, count={m['count']})")
