import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import pyalex
from process_data import SentenceTransformerWithPrefix
from asreview.models.balance import DoubleBalance
from asreview.models.classifiers import LogisticClassifier

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
        page = identifiers[i:i+OPENALEX_MAX_OR_LENGTH]
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

# Dataset A: The 1000 records closest to the inclusion criteria.
inclusion_criteria_df = pd.read_parquet(
    data_dir / "inclusion_criteria_response.parquet"
)
inclusion_criteria_df = (
    inclusion_criteria_df[inclusion_criteria_df["rank"].lt(1000)]
    .sort_values(by="rank")
    .drop("embedding", axis=1)
)
inclusion_criteria_df["id"] = OPENALEX_PREFIX + inclusion_criteria_df.id
inclusion_criteria_data = enrich_data(
    inclusion_criteria_df.id.tolist(),
    columns=["id", "doi", "title", "abstract"]
)
inclusion_criteria_data = (
    inclusion_criteria_data.merge(
        inclusion_criteria_df[["id", "rank"]], on="id", how="left"
    )
    .sort_values(by="rank")
    .reset_index(drop=True)
)
inclusion_criteria_data.to_csv(
    data_dir / "inclusion_criteria_dataset.csv",
    index=False
)


# Dataset B: The top N when querying using included records, such that the total is 7000
included_records_df = pd.read_parquet(data_dir / "included_records_response.parquet")

# As we saw in the exploratory analysis, N=427 is what we should pick.
# I sort by rank so that the first records are the most likely to be relevant.
included_records_df = (
    included_records_df[included_records_df["rank"].lt(427)]
    .drop_duplicates("id")
    .sort_values(by="rank")
    .reset_index(drop=True)
)
included_records_df["id"] = OPENALEX_PREFIX + included_records_df.id
included_records_data = enrich_data(
    included_records_df.id.tolist(),
    columns=["id", "doi", "title", "abstract"]
)
included_records_data = (
    included_records_data.merge(
        included_records_df[["id", "rank"]], on="id", how="left"
    )
    .sort_values(by="rank")
    .reset_index(drop=True)
)
included_records_data.to_csv(
    data_dir / "included_records_dataset.csv",
    index=False
)


# Dataset C: The top 10*N and then bring it down to 7000 using logistic regression.
included_records_df = pd.read_parquet(data_dir / "included_records_response.parquet")
included_records_df = (
    included_records_df[included_records_df["rank"].lt(4270)]
    .drop_duplicates("id")
    .sort_values(by="rank")
    .reset_index(drop=True)
)
feature_matrix = np.vstack(included_records_df["embedding"])

# Get the training feature matrix using the original dataset.
original_dataset = pd.read_csv(data_dir / "synergy" / "van_de_Schoot_2018.csv")
# Remove the extra excluded dois from the included records.
extra_excluded_dois = [
    "10.1037/a0020809",
    "10.1097/BCR.0b013e3181cb8ee6",
    "10.1007/s00520-015-2960-x",
    "10.1016/j.pain.2010.02.013",
]
extra_excluded_row_numbers = []
for doi in extra_excluded_dois:
    extra_excluded_row_numbers.append(original_dataset.loc[
        original_dataset.doi.fillna("").str.lower().str.contains(doi.lower())
    , "doi"].index.values[0])
original_dataset.loc[extra_excluded_row_numbers, "label_included"] = 0

# Vectorize the original dataset.
MODEL_NAME = "intfloat/multilingual-e5-small"
PREFIX = "query: "
DEVICE = "cuda"
MODEL = SentenceTransformerWithPrefix(MODEL_NAME, device=DEVICE)
MODEL.set_prefix(PREFIX)

texts = (
    original_dataset.title.fillna("").astype(str) 
    + " " + original_dataset.abstract.fillna("").astype(str)
).to_list()
training_feature_matrix = MODEL.encode(texts, show_progress_bar=True)

# Train a model.
seed = 42
balance_model = DoubleBalance(random_state=seed)
classifier = LogisticClassifier(random_state=seed)

X_train, y_train = balance_model.sample(
    X=training_feature_matrix,
    y=original_dataset.label_included.to_numpy(),
    train_idx=np.arange(len(training_feature_matrix))
)
classifier.fit(X_train, y_train)

# Make relevance predictions
relevance_scores = classifier.predict_proba(feature_matrix)[:,1]
included_records_df["relevance_score"] = relevance_scores

# Get the top 7000 unique papers.
included_records_df = (
    included_records_df
    .drop_duplicates("id")
    .sort_values(by="relevance_score", ascending=False)
    .iloc[:7000]
)

# Enrich with data from OpenAlex
included_records_df["id"] = OPENALEX_PREFIX + included_records_df.id
included_records_data = enrich_data(
    included_records_df.id.tolist(),
    columns=["id", "doi", "title", "abstract"]
)
included_records_data = (
    included_records_data.merge(
        included_records_df[["id", "relevance_score"]], on="id", how="left"
    )
    .sort_values(by="relevance_score", ascending=False)
    .reset_index(drop=True)
)
included_records_data.to_csv(
    data_dir / "included_records_logistic_dataset.csv",
    index=False
)
# Note that this dataset has 6997 records instead of 7000. The reason is that OpenAlex
# has merged 3 records since October. For example https://openalex.org/W4287871760 and
# https://openalex.org/W3046970688 point to the same work now.
