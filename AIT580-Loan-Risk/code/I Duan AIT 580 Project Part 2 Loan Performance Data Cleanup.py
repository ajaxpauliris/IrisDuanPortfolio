# Clean up Q1 and Q2 2020 performance data. Q3 and 4 are too large so they need to be cleaned up in chunks. 

import pandas as pd

# Define column names based on metadata
performance_columns = [
    "loan_sequence_number", "monthly_reporting_period", "current_actual_upb", "current_loan_delinquency_status",
    "loan_age", "remaining_months_to_legal_maturity", "defect_settlement_date", "modification_flag",
    "zero_balance_code", "zero_balance_effective_date", "current_interest_rate", "current_non_interest_bearing_upb",
    "due_date_last_paid_installment", "mi_recoveries", "net_sale_proceeds", "non_mi_recoveries", "total_expenses",
    "legal_costs", "maintenance_preservation_costs", "taxes_and_insurance", "miscellaneous_expenses",
    "actual_loss_calculation", "cumulative_modification_cost", "step_modification_flag", "payment_deferral",
    "estimated_loan_to_value", "zero_balance_removal_upb", "delinquent_accrued_interest",
    "delinquency_due_to_disaster", "borrower_assistance_status_code", "current_month_modification_cost",
    "interest_bearing_upb"
]

# Define correct data types based on metadata
performance_dtype_mapping = {
    "loan_sequence_number": str,
    "monthly_reporting_period": str,  # Convert to DATE format later
    "current_actual_upb": "float64",
    "current_loan_delinquency_status": str,
    "loan_age": "Int64",
    "remaining_months_to_legal_maturity": "Int64",
    "defect_settlement_date": str,  # Convert to DATE later
    "modification_flag": str,
    "zero_balance_code": "Int64",
    "zero_balance_effective_date": str,  # Convert to DATE later
    "current_interest_rate": "float64",
    "current_non_interest_bearing_upb": "float64",  
    "due_date_last_paid_installment": str,  # Convert to DATE later
    "mi_recoveries": "float64",
    "net_sale_proceeds": str,  # Read as string first, convert later
    "non_mi_recoveries": "float64",
    "total_expenses": "float64",
    "legal_costs": "float64",
    "maintenance_preservation_costs": "float64",
    "taxes_and_insurance": "float64",
    "miscellaneous_expenses": "float64",
    "actual_loss_calculation": "float64",
    "cumulative_modification_cost": "float64",
    "step_modification_flag": str,
    "payment_deferral": str,
    "estimated_loan_to_value": "float64",  
    "zero_balance_removal_upb": "float64",
    "delinquent_accrued_interest": "float64",
    "delinquency_due_to_disaster": str,
    "borrower_assistance_status_code": str,
    "current_month_modification_cost": "float64",
    "interest_bearing_upb": "float64"
}

# Function to clean and transform loan performance data
def process_performance_txt_file(file_path, quarter_year):
    # Load TXT file with pipe delimiter and no headers
    df = pd.read_csv(file_path, delimiter="|", names=performance_columns, dtype=performance_dtype_mapping)

    # Convert date fields to proper DATE format (YYYYMM to YYYY-MM-DD)
    date_columns = ["monthly_reporting_period", "defect_settlement_date", "zero_balance_effective_date", "due_date_last_paid_installment"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%Y%m', errors='coerce')

    # Convert net_sale_proceeds to float, replacing "U" with NaN
    df["net_sale_proceeds"] = df["net_sale_proceeds"].replace("U", pd.NA).astype(float)

   # Round numeric fields to match PostgreSQL precision
    round_cols_2 = [
    "current_actual_upb", "current_non_interest_bearing_upb", "mi_recoveries", "net_sale_proceeds",
    "non_mi_recoveries", "total_expenses", "legal_costs", "maintenance_preservation_costs",
    "taxes_and_insurance", "miscellaneous_expenses", "actual_loss_calculation",
    "cumulative_modification_cost", "estimated_loan_to_value", "zero_balance_removal_upb",
    "delinquent_accrued_interest", "current_month_modification_cost", "interest_bearing_upb"
    ]
   
    df[round_cols_2] = df[round_cols_2].round(2)

# Round "current_interest_rate" to 3 decimal places
    df["current_interest_rate"] = df["current_interest_rate"].round(3)


    # Add quarter_year column
    df["quarter_year"] = quarter_year

    return df

# Process Q1 performance data file
file_path_performance = r"C:\Users\irisd\OneDrive - George Mason University - O365 Production\AIT 580\historical_data_2020\historical_data_2020Q1\historical_data_time_2020Q1.txt"
q1_performance_df = process_performance_txt_file(file_path_performance, "2020Q1")

# Save cleaned performance data
q1_performance_df.to_csv(r"C:\Users\irisd\cleaned_historical_performance_2020Q1.csv", index=False)

print("✅ Loan Performance Data cleaned and saved successfully!")

#----Clean up q2 loan performance data

# Define column names based on metadata
performance_columns = [
    "loan_sequence_number", "monthly_reporting_period", "current_actual_upb", "current_loan_delinquency_status",
    "loan_age", "remaining_months_to_legal_maturity", "defect_settlement_date", "modification_flag",
    "zero_balance_code", "zero_balance_effective_date", "current_interest_rate", "current_non_interest_bearing_upb",
    "due_date_last_paid_installment", "mi_recoveries", "net_sale_proceeds", "non_mi_recoveries", "total_expenses",
    "legal_costs", "maintenance_preservation_costs", "taxes_and_insurance", "miscellaneous_expenses",
    "actual_loss_calculation", "cumulative_modification_cost", "step_modification_flag", "payment_deferral",
    "estimated_loan_to_value", "zero_balance_removal_upb", "delinquent_accrued_interest",
    "delinquency_due_to_disaster", "borrower_assistance_status_code", "current_month_modification_cost",
    "interest_bearing_upb"
]

# Define correct data types based on metadata
performance_dtype_mapping = {
    "loan_sequence_number": str,
    "monthly_reporting_period": str,  # Convert to DATE format later
    "current_actual_upb": "float64",
    "current_loan_delinquency_status": str,
    "loan_age": "Int64",
    "remaining_months_to_legal_maturity": "Int64",
    "defect_settlement_date": str,  # Convert to DATE later
    "modification_flag": str,
    "zero_balance_code": "Int64",
    "zero_balance_effective_date": str,  # Convert to DATE later
    "current_interest_rate": "float64",
    "current_non_interest_bearing_upb": "float64",  
    "due_date_last_paid_installment": str,  # Convert to DATE later
    "mi_recoveries": "float64",
    "net_sale_proceeds": str,  # Read as string first, convert later
    "non_mi_recoveries": "float64",
    "total_expenses": "float64",
    "legal_costs": "float64",
    "maintenance_preservation_costs": "float64",
    "taxes_and_insurance": "float64",
    "miscellaneous_expenses": "float64",
    "actual_loss_calculation": "float64",
    "cumulative_modification_cost": "float64",
    "step_modification_flag": str,
    "payment_deferral": str,
    "estimated_loan_to_value": "float64",  
    "zero_balance_removal_upb": "float64",
    "delinquent_accrued_interest": "float64",
    "delinquency_due_to_disaster": str,
    "borrower_assistance_status_code": str,
    "current_month_modification_cost": "float64",
    "interest_bearing_upb": "float64"
}

# Function to clean and transform loan performance data
def process_performance_txt_file(file_path, quarter_year):
    # Load TXT file with pipe delimiter and no headers
    df = pd.read_csv(file_path, delimiter="|", names=performance_columns, dtype=performance_dtype_mapping)

    # Convert date fields to proper DATE format (YYYYMM to YYYY-MM-DD)
    date_columns = ["monthly_reporting_period", "defect_settlement_date", "zero_balance_effective_date", "due_date_last_paid_installment"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%Y%m', errors='coerce')

    # Convert net_sale_proceeds to float, replacing "U" with NaN
    df["net_sale_proceeds"] = df["net_sale_proceeds"].replace("U", pd.NA).astype(float)

   # Round numeric fields to match PostgreSQL precision
    round_cols_2 = [
    "current_actual_upb", "current_non_interest_bearing_upb", "mi_recoveries", "net_sale_proceeds",
    "non_mi_recoveries", "total_expenses", "legal_costs", "maintenance_preservation_costs",
    "taxes_and_insurance", "miscellaneous_expenses", "actual_loss_calculation",
    "cumulative_modification_cost", "estimated_loan_to_value", "zero_balance_removal_upb",
    "delinquent_accrued_interest", "current_month_modification_cost", "interest_bearing_upb"
    ]
   
    df[round_cols_2] = df[round_cols_2].round(2)

# Round "current_interest_rate" to 3 decimal places
    df["current_interest_rate"] = df["current_interest_rate"].round(3)


    # Add quarter_year column
    df["quarter_year"] = quarter_year

    return df

# Process Q2 performance data file
file_path_performance = r"C:\Users\irisd\OneDrive - George Mason University - O365 Production\AIT 580\historical_data_2020\historical_data_2020Q2\historical_data_time_2020Q2.txt"
q1_performance_df = process_performance_txt_file(file_path_performance, "2020Q2")

# Save cleaned performance data
q1_performance_df.to_csv(r"C:\Users\irisd\cleaned_historical_performance_2020Q2.csv", index=False)

print("✅ Loan Performance Data cleaned and saved successfully!")



#----Clean up q3 loan performance data in chunks since the file is too large
import pandas as pd

# File path for Q3 Performance Data
file_path_q3 = "C:/Users/irisd/OneDrive - George Mason University - O365 Production/AIT 580/historical_data_2020/historical_data_2020Q3/historical_data_time_2020Q3.txt"
output_q3_cleaned = "C:/Users/irisd/cleaned_historical_performance_2020Q3.csv"


# Define column names based on metadata
performance_columns = [
    "loan_sequence_number", "monthly_reporting_period", "current_actual_upb", "current_loan_delinquency_status",
    "loan_age", "remaining_months_to_legal_maturity", "defect_settlement_date", "modification_flag",
    "zero_balance_code", "zero_balance_effective_date", "current_interest_rate", "current_non_interest_bearing_upb",
    "due_date_last_paid_installment", "mi_recoveries", "net_sale_proceeds", "non_mi_recoveries", "total_expenses",
    "legal_costs", "maintenance_preservation_costs", "taxes_and_insurance", "miscellaneous_expenses",
    "actual_loss_calculation", "cumulative_modification_cost", "step_modification_flag", "payment_deferral",
    "estimated_loan_to_value", "zero_balance_removal_upb", "delinquent_accrued_interest",
    "delinquency_due_to_disaster", "borrower_assistance_status_code", "current_month_modification_cost",
    "interest_bearing_upb"
]

# Define correct data types based on metadata
performance_dtype_mapping = {
    "loan_sequence_number": str,
    "monthly_reporting_period": str,  # Convert to DATE format later
    "current_actual_upb": "float64",
    "current_loan_delinquency_status": str,
    "loan_age": "Int64",
    "remaining_months_to_legal_maturity": "Int64",
    "defect_settlement_date": str,  # Convert to DATE later
    "modification_flag": str,
    "zero_balance_code": "Int64",
    "zero_balance_effective_date": str,  # Convert to DATE later
    "current_interest_rate": "float64",
    "current_non_interest_bearing_upb": "float64",  
    "due_date_last_paid_installment": str,  # Convert to DATE later
    "mi_recoveries": "float64",
    "net_sale_proceeds": str,  # Read as string first, convert later
    "non_mi_recoveries": "float64",
    "total_expenses": "float64",
    "legal_costs": "float64",
    "maintenance_preservation_costs": "float64",
    "taxes_and_insurance": "float64",
    "miscellaneous_expenses": "float64",
    "actual_loss_calculation": "float64",
    "cumulative_modification_cost": "float64",
    "step_modification_flag": str,
    "payment_deferral": str,
    "estimated_loan_to_value": "float64",  
    "zero_balance_removal_upb": "float64",
    "delinquent_accrued_interest": "float64",
    "delinquency_due_to_disaster": str,
    "borrower_assistance_status_code": str,
    "current_month_modification_cost": "float64",
    "interest_bearing_upb": "float64"
}

# Read and process the file in chunks
chunksize = 500000  # Adjust as needed

# Open output file to write cleaned data
with open(output_q3_cleaned, "w") as output_file:
    # Process the file in chunks
    for chunk in pd.read_csv(file_path_q3, delimiter="|", names=performance_columns, dtype=performance_dtype_mapping, chunksize=chunksize):
        
        # Convert date fields
        chunk["monthly_reporting_period"] = pd.to_datetime(chunk["monthly_reporting_period"], format='%Y%m', errors='coerce')
        chunk["defect_settlement_date"] = pd.to_datetime(chunk["defect_settlement_date"], format='%Y%m', errors='coerce')
        chunk["zero_balance_effective_date"] = pd.to_datetime(chunk["zero_balance_effective_date"], format='%Y%m', errors='coerce')
        chunk["due_date_last_paid_installment"] = pd.to_datetime(chunk["due_date_last_paid_installment"], format='%Y%m', errors='coerce')

        # Convert "U" in net sale proceeds to NaN
        chunk["net_sale_proceeds"] = pd.to_numeric(chunk["net_sale_proceeds"], errors='coerce')

        # Round float values for PostgreSQL
        float_cols = [
            "current_actual_upb", "current_interest_rate", "current_non_interest_bearing_upb", "mi_recoveries", "net_sale_proceeds",
            "non_mi_recoveries", "total_expenses", "legal_costs", "maintenance_preservation_costs", "taxes_and_insurance",
            "miscellaneous_expenses", "actual_loss_calculation", "cumulative_modification_cost", "estimated_loan_to_value",
            "zero_balance_removal_upb", "delinquent_accrued_interest", "current_month_modification_cost", "interest_bearing_upb"
        ]
        chunk[float_cols] = chunk[float_cols].round(3)

        # Add quarter-year column
        chunk["quarter_year"] = "2020Q3"

        # Save cleaned chunk to file (append mode)
        chunk.to_csv(output_q3_cleaned, index=False, mode="a", header=False)

print("✅ Q3 Performance Data processed and saved successfully!")

#----Clean up q4 loan performance data in chunks since the file is too large
import pandas as pd

# File path for Q4 Performance Data
file_path_q4 = "C:/Users/irisd/OneDrive - George Mason University - O365 Production/AIT 580/historical_data_2020/historical_data_2020Q4/historical_data_time_2020Q4.txt"
output_q4_cleaned = "C:/Users/irisd/cleaned_historical_performance_2020Q4.csv"


# Define column names based on metadata
performance_columns = [
    "loan_sequence_number", "monthly_reporting_period", "current_actual_upb", "current_loan_delinquency_status",
    "loan_age", "remaining_months_to_legal_maturity", "defect_settlement_date", "modification_flag",
    "zero_balance_code", "zero_balance_effective_date", "current_interest_rate", "current_non_interest_bearing_upb",
    "due_date_last_paid_installment", "mi_recoveries", "net_sale_proceeds", "non_mi_recoveries", "total_expenses",
    "legal_costs", "maintenance_preservation_costs", "taxes_and_insurance", "miscellaneous_expenses",
    "actual_loss_calculation", "cumulative_modification_cost", "step_modification_flag", "payment_deferral",
    "estimated_loan_to_value", "zero_balance_removal_upb", "delinquent_accrued_interest",
    "delinquency_due_to_disaster", "borrower_assistance_status_code", "current_month_modification_cost",
    "interest_bearing_upb"
]

# Define correct data types based on metadata
performance_dtype_mapping = {
    "loan_sequence_number": str,
    "monthly_reporting_period": str,  # Convert to DATE format later
    "current_actual_upb": "float64",
    "current_loan_delinquency_status": str,
    "loan_age": "Int64",
    "remaining_months_to_legal_maturity": "Int64",
    "defect_settlement_date": str,  # Convert to DATE later
    "modification_flag": str,
    "zero_balance_code": "Int64",
    "zero_balance_effective_date": str,  # Convert to DATE later
    "current_interest_rate": "float64",
    "current_non_interest_bearing_upb": "float64",  
    "due_date_last_paid_installment": str,  # Convert to DATE later
    "mi_recoveries": "float64",
    "net_sale_proceeds": str,  # Read as string first, convert later
    "non_mi_recoveries": "float64",
    "total_expenses": "float64",
    "legal_costs": "float64",
    "maintenance_preservation_costs": "float64",
    "taxes_and_insurance": "float64",
    "miscellaneous_expenses": "float64",
    "actual_loss_calculation": "float64",
    "cumulative_modification_cost": "float64",
    "step_modification_flag": str,
    "payment_deferral": str,
    "estimated_loan_to_value": "float64",  
    "zero_balance_removal_upb": "float64",
    "delinquent_accrued_interest": "float64",
    "delinquency_due_to_disaster": str,
    "borrower_assistance_status_code": str,
    "current_month_modification_cost": "float64",
    "interest_bearing_upb": "float64"
}

# Read and process the file in chunks
chunksize = 500000  # Adjust as needed

# Open output file to write cleaned data
with open(output_q4_cleaned, "w") as output_file:
    # Process the file in chunks
    for chunk in pd.read_csv(file_path_q4, delimiter="|", names=performance_columns, dtype=performance_dtype_mapping, chunksize=chunksize):
        
        # Convert date fields
        chunk["monthly_reporting_period"] = pd.to_datetime(chunk["monthly_reporting_period"], format='%Y%m', errors='coerce')
        chunk["defect_settlement_date"] = pd.to_datetime(chunk["defect_settlement_date"], format='%Y%m', errors='coerce')
        chunk["zero_balance_effective_date"] = pd.to_datetime(chunk["zero_balance_effective_date"], format='%Y%m', errors='coerce')
        chunk["due_date_last_paid_installment"] = pd.to_datetime(chunk["due_date_last_paid_installment"], format='%Y%m', errors='coerce')

        # Convert "U" in net sale proceeds to NaN
        chunk["net_sale_proceeds"] = pd.to_numeric(chunk["net_sale_proceeds"], errors='coerce')

        # Round float values for PostgreSQL
        float_cols = [
            "current_actual_upb", "current_interest_rate", "current_non_interest_bearing_upb", "mi_recoveries", "net_sale_proceeds",
            "non_mi_recoveries", "total_expenses", "legal_costs", "maintenance_preservation_costs", "taxes_and_insurance",
            "miscellaneous_expenses", "actual_loss_calculation", "cumulative_modification_cost", "estimated_loan_to_value",
            "zero_balance_removal_upb", "delinquent_accrued_interest", "current_month_modification_cost", "interest_bearing_upb"
        ]
        chunk[float_cols] = chunk[float_cols].round(3)

        # Add quarter-year column
        chunk["quarter_year"] = "2020Q4"

        # Save cleaned chunk to file (append mode)
        chunk.to_csv(output_q4_cleaned, index=False, mode="a", header=False)

print("✅ Q4 Performance Data processed and saved successfully!")