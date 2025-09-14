import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler

# ✅ Define PostgreSQL connection parameters
db_host = "localhost"  
db_port = "5432"  
db_name = "freddie_mac_2020"  
db_user = "postgres"
db_password = "2781"

# ✅ Create a database connection
engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# 🔹 Query for mortgage loan data
query = """
SELECT loan_sequence_number, credit_score, original_interest_rate, current_interest_rate, 
       zero_balance_code, loan_purpose, original_ltv, original_dti, original_loan_term,
       estimated_loan_to_value, delinquent_accrued_interest, loan_age, 
       current_loan_delinquency_status
FROM merged_loan_data
"""
df = pd.read_sql(query, engine)

# 🔹 Handle "RA" in delinquency status (REO Acquisition)
df["current_loan_delinquency_status"] = df["current_loan_delinquency_status"].replace("RA", 999)

# 🔹 Convert delinquency status to numeric
df["current_loan_delinquency_status"] = pd.to_numeric(df["current_loan_delinquency_status"], errors="coerce")

# 🔹 Create delinquency flag (1 = delinquent, 0 = current)
df["delinquent_flag"] = (df["current_loan_delinquency_status"] > 0).astype(int)

# 🔹 **Categorizing Credit Score, LTV, and DTI Before Visualizing**
bins = [0, 599, 649, 699, 749, 799, 850]
labels = ["Poor", "Fair", "Average", "Good", "Very Good", "Excellent"]
df["credit_score_category"] = pd.cut(df["credit_score"], bins=bins, labels=labels, right=True)

ltv_bins = [0, 80, 90, 100]
ltv_labels = ["Low Risk (<80%)", "Moderate Risk (80-90%)", "High Risk (>90%)"]
df["ltv_category"] = pd.cut(df["original_ltv"], bins=ltv_bins, labels=ltv_labels, right=False)

dti_bins = [0, 20, 35, 100]
dti_labels = ["Low DTI (<20%)", "Moderate DTI (20-35%)", "High DTI (>35%)"]
df["dti_category"] = pd.cut(df["original_dti"], bins=dti_bins, labels=dti_labels, right=False)

# ✅ Check distributions
print("\n🔹 Credit Score Categories Distribution:\n", df["credit_score_category"].value_counts())

# 🔹 **Feature Selection for Logistic Regression on Delinquency**
features = ["credit_score", "original_ltv", "original_dti", "original_loan_term"]
target = "delinquent_flag"

df_delinquency = df[features + [target]].dropna()

# ✅ Check Class Distribution
print("\n🔹 Delinquency Class Distribution:\n", df_delinquency[target].value_counts())

# ✅ Handle Class Imbalance using **Undersampling**
X = df_delinquency[features]
y = df_delinquency[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

undersample = RandomUnderSampler(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = undersample.fit_resample(X_train, y_train)

# ✅ Train Logistic Regression with **Standardization**
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced")
model.fit(X_train_scaled, y_train_resampled)

# ✅ Predictions
y_pred = model.predict(X_test_scaled)

# ✅ Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔹 **Feature Importance Analysis**
feature_importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\n🔹 Feature Importance (Logistic Regression):")
print(feature_importance)

# 🔹 **Visualize Feature Importance**
plt.figure(figsize=(8,5))
sns.barplot(x="Coefficient", y="Feature", data=feature_importance)
plt.title("Logistic Regression Coefficients for Loan Delinquency")
plt.show()

# ✅ **Visualizations for Delinquency**
plt.figure(figsize=(10, 5))
sns.barplot(x="credit_score_category", y="delinquent_flag", data=df)
plt.title("Delinquency Rate by Credit Score Category")
plt.ylabel("Delinquency Rate")
plt.xlabel("Credit Score Category")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="ltv_category", y="delinquent_flag", data=df)
plt.title("Delinquency Rate by LTV Category")
plt.ylabel("Delinquency Rate")
plt.xlabel("LTV Category")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="dti_category", y="delinquent_flag", data=df)
plt.title("Delinquency Rate by DTI Category")
plt.ylabel("Delinquency Rate")
plt.xlabel("DTI Category")
plt.show()

# ✅ Train Logistic Regression with **Standardization**
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced", solver="liblinear")
model.fit(X_train_scaled, y_train_resampled)

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# ✅ Handle Class Imbalance using **SMOTE** (Instead of Undersampling)
smote = SMOTE(sampling_strategy=0.4, random_state=42)  # Increase minority class by 10%
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ✅ Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# ✅ Predictions
y_pred_rf = rf_model.predict(X_test)

# ✅ Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))




# 🔹 Step 4: Categorize Credit Score
bins = [0, 599, 649, 699, 749, 799, 850]
labels = ["Poor", "Fair", "Average", "Good", "Very Good", "Excellent"]
df["credit_score_category"] = pd.cut(df["credit_score"], bins=bins, labels=labels, right=True)

# 🔹 Step 5: Categorize Loan-to-Value (LTV)
ltv_bins = [0, 80, 90, 100]
ltv_labels = ["Low Risk (<80%)", "Moderate Risk (80-90%)", "High Risk (>90%)"]
df["ltv_category"] = pd.cut(df["original_ltv"], bins=ltv_bins, labels=ltv_labels, right=False)

# 🔹 Step 6: Categorize Debt-to-Income Ratio (DTI)
dti_bins = [0, 20, 35, 100]
dti_labels = ["Low DTI (<20%)", "Moderate DTI (20-35%)", "High DTI (>35%)"]
df["dti_category"] = pd.cut(df["original_dti"], bins=dti_bins, labels=dti_labels, right=False)

# ✅ Check distributions
print("\n🔹 Credit Score Categories Distribution:\n", df["credit_score_category"].value_counts())

# 🔹 Step 7: Map Loan Purpose
loan_purpose_map = {
    "P": "Purchase",
    "C": "Refinance - Cash Out",
    "N": "Refinance - No Cash Out",
    "R": "Refinance - Not Specified",
    "9": "Not Available"
}
df["loan_purpose_category"] = df["loan_purpose"].map(loan_purpose_map)

# 🔹 Step 8: Refinancing Analysis (Using Loan Purpose)
df_refinance = df[df["loan_purpose"].isin(["P","C", "N", "R", "9"])]

# ✅ Refinancing Class Distribution
print("\n🔹 Refinancing Loan Purpose Distribution:\n", df_refinance["loan_purpose_category"].value_counts())

# 🔹 Prepare Logistic Regression Dataset
features = ["credit_score", "original_interest_rate"]
target = "loan_purpose_category"

df_refinance = df_refinance[features + [target]].dropna()
df_refinance["refinanced"] = df_refinance[target].apply(lambda x: 1 if "Refinance" in x else 0)
df_refinance.drop(columns=[target], inplace=True)

# ✅ Check Class Distribution
print("\n🔹 Refinanced Value Counts:\n", df_refinance["refinanced"].value_counts())

# ✅ Fix Class Imbalance with **Undersampling**
X = df_refinance[features]
y = df_refinance["refinanced"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

undersample = RandomUnderSampler(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = undersample.fit_resample(X_train, y_train)

# ✅ Train Logistic Regression with Class Weights
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced")
model.fit(X_train_scaled, y_train_resampled)

# ✅ Predictions
y_pred = model.predict(X_test_scaled)

# ✅ Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔹 Fix Seaborn Plot Warnings (Explicitly Assign Hue and Legend)
plt.figure(figsize=(10, 5))
sns.boxplot(x="credit_score_category", y="original_interest_rate", data=df, hue="credit_score_category", palette= "deep", legend=False)
plt.title("Mortgage Interest Rates Across Credit Score Bands")
plt.xlabel("Credit Score Category")
plt.ylabel("Original Interest Rate")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="dti_category", y="delinquent_flag", data=df, hue="dti_category", palette= "deep", legend=False)
plt.title("Delinquency Rate by DTI Category")
plt.ylabel("Delinquency Rate")
plt.xlabel("DTI Category")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="ltv_category", y="delinquent_flag", data=df, hue="ltv_category",  palette= "deep", legend=False)
plt.title("Delinquency Rate by LTV Category")
plt.ylabel("Delinquency Rate")
plt.xlabel("LTV Category")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="credit_score_category", y="delinquent_flag", data=df, hue="credit_score_category", palette= "deep", legend=False)
plt.title("Delinquency Rate by Credit Score Category")
plt.ylabel("Delinquency Rate")
plt.xlabel("Credit Score Category")
plt.show()



