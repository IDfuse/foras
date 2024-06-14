# Script to add the data of the corrected dataset 3b to the motherfile.

import os
from pathlib import Path

import pandas as pd
import pyalex
from dotenv import load_dotenv

load_dotenv()
pyalex.config.email = os.environ.get("OPENALEX_EMAIL")
data_dir = Path("data")
OPENALEX_PREFIX = "https://openalex.org/"
OPENALEX_MAX_OR_LENGTH = 50


def enrich_data(identifiers: list[str], columns: list[str]) -> pd.DataFrame:
    """Enrich OpenAlex identifiers with data from OpenAlex.

    Parameters
    ----------
    identifiers : list[str]
        List of OpenAlex identifiers.
    columns : list[str]
        List of columns to select. These should be valid OpenAlex works fields. The only
        exception is 'abstract' which is allowed. The value from
        'abstract_inverted_index' will be automatically converted to an abstract.

    Returns
    -------
    pd.DataFrame
        Dataframe containing one row for each identifier with the given columns.
    """
    request_columns = [
        col if col != "abstract" else "abstract_inverted_index" for col in columns
    ]
    dataframes = []
    for i in range(0, len(identifiers), OPENALEX_MAX_OR_LENGTH):
        page = identifiers[i : i + OPENALEX_MAX_OR_LENGTH]
        data = (
            pyalex.Works()
            .filter(openalex="|".join(page))
            .select(request_columns)
            .get(per_page=OPENALEX_MAX_OR_LENGTH)
        )
        data_df = pd.DataFrame(data)
        if "abstract" in columns:
            data_df.drop("abstract_inverted_index", axis=1, inplace=True)
            data_df["abstract"] = [work["abstract"] for work in data]
        dataframes.append(data_df)
        if len(data_df) != 50:
            print(i)
            print(page)
    return pd.concat(dataframes).reset_index(drop=True)


# Dataset B: The top N when querying using included records, such that the total is 1000
included_records_df = pd.read_parquet(data_dir / "included_records_response.parquet")

# If we pick N=50, we get 999 record.
# I sort by rank so that the first records are the most likely to be relevant.
included_records_df = (
    included_records_df[included_records_df["rank"].lt(50)]
    .drop_duplicates("id")
    .sort_values(by="rank")
    .reset_index(drop=True)
)
included_records_df["id"] = OPENALEX_PREFIX + included_records_df.id
included_records_data = enrich_data(
    included_records_df.id.tolist(), columns=["id", "doi", "title", "abstract"]
)
included_records_data = (
    included_records_data.merge(
        included_records_df[["id", "rank"]], on="id", how="left"
    )
    .sort_values(by="rank")
    .reset_index(drop=True)
)
included_records_data.to_excel(
    data_dir / "included_records_dataset_1000.xlsx", index=False
)

# Add the data to the motherfile.
motherfile = pd.read_excel(data_dir / "Motherfile_130624.xlsx")
motherfile = motherfile.rename(
    {"Data_3b_includedrecords_top88-corrected": "Data_3b_old"}, axis=1
)

max_mid = motherfile["MID"].str.slice(1).astype(int).max()
# 15004

included_records_data["Data_3b"] = 1
included_records_data["MID_3b"] = [
    f"M{mid}" for mid in range(max_mid + 1, max_mid + len(included_records_data) + 1)
]
included_records_data = included_records_data.rename(
    {
        "id": "openalex_id",
        "doi": "doi_3b",
        "title": "title_3b",
        "abstract": "abstract_3b",
    },
    axis=1,
).drop("rank", axis=1)
motherfile = motherfile.merge(included_records_data, how="outer", on="openalex_id")
motherfile["Data_3b"] = motherfile["Data_3b"].fillna(0)
# Use DOI, title, abstract from motherfile where available, otherwise from 3B.
motherfile["doi"] = motherfile["doi"].where(
    motherfile["doi"].notna(), motherfile["doi_3b"]
)
motherfile["title"] = motherfile["title"].where(
    motherfile["title"].notna(), motherfile["title_3b"]
)
motherfile["abstract"] = motherfile["abstract"].where(
    motherfile["abstract"].notna(), motherfile["abstract_3b"]
)
motherfile["MID"] = motherfile["MID"].where(
    motherfile["MID"].notna(), motherfile["MID_3b"]
)
for column in [
    "Data_0_Synergy",
    "Data_1_old_replication",
    "Data_2_old_comprehensive",
    "Data_4a_snowballing",
    "Data_3a_inlusion_criteria-corrected",
    "Data_3b_old",
    "Data_3c_active_learning_total1000-corrected",
    "Data_3b",
]:
    motherfile[column] = motherfile[column].fillna(0)

motherfile = motherfile.drop(["MID_3b", "doi_3b", "title_3b", "abstract_3b"], axis=1)
motherfile.to_excel(data_dir / "Motherfile_140624.xlsx", index=False)