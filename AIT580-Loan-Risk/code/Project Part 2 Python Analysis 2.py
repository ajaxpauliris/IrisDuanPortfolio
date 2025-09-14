import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler

# PostgreSQL Connection Parameters
db_host = "localhost"
db_port = "5432"
db_name = "freddie_mac_2020"
db_user = "postgres"
db_password = "2781"

# Create a database connection to postgresql
engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# Query for mortgage loan data
query = """
SELECT loan_sequence_number, credit_score, original_interest_rate, current_interest_rate, 
       loan_purpose, original_ltv, original_dti, original_loan_term,
       estimated_loan_to_value, loan_age, 
       current_loan_delinquency_status
FROM merged_loan_data
"""
df = pd.read_sql(query, engine)




#  Handle "RA" in delinquency status (RA= REO Acquisition)
df["current_loan_delinquency_status"] = df["current_loan_delinquency_status"].replace("RA", 999)
df["current_loan_delinquency_status"] = pd.to_numeric(df["current_loan_delinquency_status"], errors="coerce")

# Create delinquency flag to denote delinquency 
df["delinquent_flag"] = (df["current_loan_delinquency_status"] > 0).astype(int)

# Create a Correlation Matrix
correlation_matrix = df.select_dtypes(include=[np.number]).corr()

# Display the Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Variable Correlation Matrix")
plt.show()

# Use credit score, LTV, DTI, and loan term as our deliquency predictors to answer question 1 
predictors = ["credit_score", "original_ltv", "original_dti", "original_loan_term"]
outcome = "delinquent_flag"

df_delinquency = df[predictors + [outcome]].dropna()

# Generate summary stats 
print(df[["credit_score", "original_interest_rate", "original_ltv", "original_dti", "delinquent_flag"]].describe())

# There is a big imbalance in the number of delinquent and non delinquent loans. We can address this by undersampling. 
X = df_delinquency[predictors]
y = df_delinquency[outcome]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

undersample = RandomUnderSampler(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = undersample.fit_resample(X_train, y_train)

# Train Logistic Regression with standardization. We need to standardize because of the big differences in numerical ranges for credit score, LTV, etc. 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced")
model.fit(X_train_scaled, y_train_resampled)

# Create a prediction model
y_pred = model.predict(X_test_scaled)

# Evaluate the model 
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Delinquency Predictor Strength Analysis
predictor_strength = pd.DataFrame({
    "Predictors": predictors,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("Delinquency Predictor Strength (Logistic Regression):")
print(predictor_strength)

# Visualize delinquency predictor strength analysis 
plt.figure(figsize=(8,5))
sns.barplot(x="Coefficient", y="Predictors", data=predictor_strength, palette="deep")
plt.title("Logistic Regression Coefficients for Loan Delinquency")
plt.show()

# Categorize Credit Score, LTV, and DTI and place them into bins for easier analysis to answer question 1 and 3 
bins = [0, 599, 649, 699, 749, 799, 850]
labels = ["Poor", "Fair", "Average", "Good", "Very Good", "Excellent"]
df["credit_score_category"] = pd.cut(df["credit_score"], bins=bins, labels=labels, right=True)

ltv_bins = [0, 80, 90, 100]
ltv_labels = ["Low Risk (<80%)", "Moderate Risk (80-90%)", "High Risk (>90%)"]
df["ltv_category"] = pd.cut(df["original_ltv"], bins=ltv_bins, labels=ltv_labels, right=False)

dti_bins = [0, 20, 35, 100]
dti_labels = ["Low DTI (<20%)", "Moderate DTI (20-35%)", "High DTI (>35%)"]
df["dti_category"] = pd.cut(df["original_dti"], bins=dti_bins, labels=dti_labels, right=False)


#Show variable distribution 
df[["credit_score", "original_ltv", "original_dti"]].hist(figsize=(10,5), bins=30)
plt.suptitle("Distributions of Key Loan Variables")
plt.show()


# Create a visualization to show deliquency rates by credit score. Since this is categorical a bar plot works best
plt.figure(figsize=(10, 5))
sns.barplot(x="credit_score_category", y="delinquent_flag", hue="credit_score_category", data=df, palette="deep", dodge=False)
plt.title("Delinquency Rate by Credit Score Category")
plt.ylabel("Delinquency Rate")
plt.xlabel("Credit Score Category")
plt.show()

# Create a visualization to show deliquency rates by LTV Category. Since this is categorical a bar plot works best
plt.figure(figsize=(10, 5))
sns.barplot(x="ltv_category", y="delinquent_flag", hue="ltv_category", data=df, palette="deep", dodge=False)
plt.title("Delinquency Rate by LTV Category")
plt.ylabel("Delinquency Rate")
plt.xlabel("LTV Category")
plt.show()

# Create a visualization to show deliquency rates by DTI Category. Since this is categorical a bar plot works best
plt.figure(figsize=(10, 5))
sns.barplot(x="dti_category", y="delinquent_flag", hue="dti_category", data=df, palette="deep", dodge=False)
plt.title("Delinquency Rate by DTI Category")
plt.ylabel("Delinquency Rate")
plt.xlabel("DTI Category")
plt.show()




#------------- Question 2 --------------
# Refinancing Analysis

# Create a Refinancing Flag based on loan_purpose
df["refinanced"] = df["loan_purpose"].apply(lambda x: 1 if x in ["C", "N", "R"] else 0)

#Limit credit score to 850 max
df_refinance = df[df["credit_score"] <= 850]

# Calculate the predictor strength based on Summary Statistics
refinance_summary = df.groupby("refinanced")[["credit_score", "original_interest_rate", "current_interest_rate", "original_ltv"]].mean()
print("\n Refinancing Summary Statistics:\n", refinance_summary)

# Calculate Correlations with Refinancing
refinance_correlation = df[["credit_score", "original_interest_rate", "current_interest_rate", "original_ltv", "refinanced"]].corr()["refinanced"].sort_values(ascending=False)
print("\n Predictor Strength (Correlation with Refinancing):\n", refinance_correlation)


# Visualize the predictor strength for refinancing
plt.figure(figsize=(8,5))
sns.barplot(x=refinance_correlation.index, y=refinance_correlation.values, palette="deep")
plt.title("Predictor Strength for Refinancing (Correlation-Based)")
plt.xlabel("Feature")
plt.ylabel("Correlation with Refinancing")
plt.xticks(rotation=45)
plt.show()

# Visualize Refinancing Trends for interest ratess
plt.figure(figsize=(10,5))
sns.boxplot(x="refinanced", y="original_interest_rate", data=df, palette="deep")
plt.title("Original Interest Rates for Refinanced vs. Non-Refinanced Loans")
plt.xlabel("Refinanced (1 = Yes, 0 = No)")
plt.ylabel("Original Interest Rate")
plt.show()

# Visualize Refinancing Trends for credit score
plt.figure(figsize=(10,5))
sns.boxplot(x="refinanced", y="credit_score", data=df_refinance, palette="deep")
plt.title("Credit Scores for Refinanced vs. Non-Refinanced Loans")
plt.xlabel("Refinanced (1 = Yes, 0 = No)")
plt.ylabel("Credit Score")
plt.show()

# Visualize Refinancing Trends for LTV
plt.figure(figsize=(10,5))
sns.boxplot(x="refinanced", y="original_ltv", data=df, palette="deep")
plt.title("LTV for Refinanced vs. Non-Refinanced Loans")
plt.xlabel("Refinanced (1 = Yes, 0 = No)")
plt.ylabel("Original LTV")
plt.show()

#------------- Question 3 --------------
# Relationship between mortgage rates and credit scores


plt.figure(figsize=(10, 5))
sns.boxplot(x="credit_score_category", y="original_interest_rate", data=df_refinance, hue="credit_score_category", palette="deep", legend=False)
plt.title("Mortgage Interest Rates Across Credit Score Bands")
plt.xlabel("Credit Score Category")
plt.ylabel("Original Interest Rate")
plt.show()

