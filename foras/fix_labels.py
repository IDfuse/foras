# Script to fix the labels in the motherfile for the three datasets that I delivered.
# It will check if the OpenAlex identifier is in one of my datasets and adjust the
# corresponding column.

from pathlib import Path

import pandas as pd

data_dir = Path("data")
motherfile = pd.read_excel(data_dir / "motherfile_with_ranking.xlsx")
df_logistic = pd.read_csv(
    data_dir / "delivered_datasets" / "Data_3c_active_learning_total1000.csv"
)
df_criteria = pd.read_csv(
    data_dir / "delivered_datasets" / "Data_3a_inlusion_criteria.csv"
)
df_included = pd.read_csv(
    data_dir / "delivered_datasets" / "Data_3b_includedrecords_top88.csv"
)

motherfile["in_logistic"] = motherfile.openalex_id.isin(df_logistic.id.values)
motherfile["in_criteria"] = motherfile.openalex_id.isin(df_criteria.id.values)
motherfile["in_included"] = motherfile.openalex_id.isin(df_included.id.values)

df_logistic["in_motherfile"] = df_logistic.id.isin(motherfile.openalex_id.values)
df_criteria["in_motherfile"] = df_criteria.id.isin(motherfile.openalex_id.values)
df_included["in_motherfile"] = df_included.id.isin(motherfile.openalex_id.values)

output_dir = data_dir / "dataset_linking"
motherfile.to_excel(output_dir / "motherfile.xlsx")
df_logistic.to_excel(output_dir / "Data_3c_active_learning_total1000.xlsx")
df_criteria["abstract"] = df_criteria.abstract.str.replace("\x02", "-")
df_criteria.to_excel(output_dir / "Data_3a_inlusion_criteria.xlsx")
df_included.to_excel(output_dir / "Data_3b_includedrecords_top88.xlsx")


print(f"Logistic length: {len(df_logistic)}")
print(f"Motherfile has {motherfile.in_logistic.sum()} in logistic")
print(f"Logistic has {df_logistic.in_motherfile.sum()} in motherfile")
# Logistic length: 1000
# Motherfile has 983 in logistic
# Logistic has 974 in motherfile
print(f"criteria length: {len(df_criteria)}")
print(f"Motherfile has {motherfile.in_criteria.sum()} in criteria")
print(f"criteria has {df_criteria.in_motherfile.sum()} in motherfile")
# criteria length: 1000
# Motherfile has 992 in criteria
# criteria has 985 in motherfile
print(f"included length: {len(df_included)}")
print(f"Motherfile has {motherfile.in_included.sum()} in included")
print(f"included has {df_included.in_motherfile.sum()} in motherfile")
# included length: 1009
# Motherfile has 997 in included
# included has 994 in motherfile