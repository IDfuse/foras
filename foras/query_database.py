from pathlib import Path

import os
import pandas as pd
import requests
from process_data import SentenceTransformerWithPrefix

MODEL_NAME = "intfloat/multilingual-e5-small"
PREFIX = "query: "
DEVICE = "cpu"
MODEL = SentenceTransformerWithPrefix(MODEL_NAME, device=DEVICE)
MODEL.set_prefix(PREFIX)
N_HITS = 5000


def query_database(
    url: str,
    n_hits: int,
    vector: list[float],
) -> list[dict]:
    """Query the database using a vector.

    Parameters
    ----------
    url : str
        URL of the search endpoint of the Vespa database.
    n_hits : int
        Number of hits to return. Maximum is 10000.
    vector : list[float]
        Text embedding

    Returns
    -------
    list[dict]
        List of dictionaries with the keys 'id', 'embedding' and 'rank', with the
        identifier, text embedding vector and rank of each record in the query response.
    """
    params = {
        "hits": n_hits,
        "timeout": 5000,
        "yql": (
            f"select * from works where "
            f"{{targetHits: {n_hits}}}nearestNeighbor(embedding,q)"
        ),
        "input.query(q)": str(vector),
    }
    res = requests.get(url=url, params=params)
    res.raise_for_status()
    res = res.json()["root"]["children"]
    return [
        {
            "id": record["fields"]["id"],
            "embedding": record["fields"]["embedding"]["values"],
            "rank": idx,
        }
        for idx, record in enumerate(res)
    ]


# Get the included records of the original review.
data_dir = Path(os.environ["DATA_DIR"])
df = pd.read_csv(Path(data_dir, "synergy", "van_de_Schoot_2018.csv"))
extra_excluded_dois = [
    "10.1037/a0020809",
    "10.1097/BCR.0b013e3181cb8ee6",
    "10.1007/s00520-015-2960-x",
    "10.1016/j.pain.2010.02.013",
]
df["extra_excluded"] = False
for doi in extra_excluded_dois:
    df.loc[
        df.doi.fillna("").str.lower().str.contains(doi.lower()), "extra_excluded"
    ] = True

df = df[df.label_included.astype(bool) & ~df.extra_excluded]
df.info()

# Vectorize the texts of the included records.
texts = (
    df.title.fillna("").astype(str) + " " + df.abstract.fillna("").astype(str)
).to_list()
included_vectors = MODEL.encode(texts)

# Query the database with each vector.
url = f"http://{os.environ['VESPA_IP']}:{os.environ['PORT']}/search/"
result_dataframes = []
for identifier, vector in zip(df.id.to_list(), included_vectors):
    print(f"Querying database for record {identifier}")
    res = query_database(url, N_HITS, vector.tolist())
    res_df = pd.DataFrame(
        {
            "query_id": identifier,
            "rank": [record["rank"] for record in res],
            "id": [record["id"] for record in res],
            "embedding": [record["embedding"] for record in res],
        }
    )
    result_dataframes.append(res_df)

result = pd.concat(result_dataframes, ignore_index=True)
print("Saving results")
result.to_parquet(Path(data_dir, "included_records_response.parquet"), index=False)


# Search using inclusion criteria
inclusion_criteria = """
Longitudinal/Prospective study with at least three-time point assessments.
PTSS are measured with a validated continuous scale (either self-report questionnaire or clinical interview) measuring PTSD symptoms according to the diagnostic criteria specificized in DSM-IV or DSM-5(TR).  Some eligible PTSD scales: Clinician Administered PTSD Scale (CAPS), PTSD Checklist (PCL), Impact of Event Scale (IES), Harvard Trauma Questionnaire (HTQ), Posttraumatic Diagnostic Scale (PDS), UCLA PTSD
PTSD assessment conducted after a traumatic event as defined by DSM-IV or DSM-V.
Trajectory analysis of PTSD using one of the following methods: Latent Growth Mixture Modeling (LGMM), Latent Curve Mixture Analysis (LCMA), or other hierarchical cluster analysis.
"""

vector = MODEL.encode([inclusion_criteria])[0]
print("Querying using inclusion criteria")
res = query_database(url, N_HITS, vector.tolist())
assert len(res) == N_HITS
res_df = pd.DataFrame(res)
res_df.to_parquet(Path(data_dir, "inclusion_criteria_response.parquet"), index=False)
