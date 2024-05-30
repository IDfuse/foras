import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from process_data import SentenceTransformerWithPrefix
from sklearn.metrics.pairwise import cosine_similarity


def recall_data(grouped_response, openalex_id: str) -> list[dict]:
    """Get the recall data corresponding to an OpenAlex identifier.

    Parameters
    ----------
    openalex_id : str
        OpenAlex identifier in short form.

    Returns
    -------
    list[dict]
        List of dictionaries {"rank": int, "query_id": openalex_id of query record}.
    """
    if openalex_id not in grouped_response.groups.keys():
        return [{"rank": None, "query_id": None}]
    ranks = grouped_response.get_group(openalex_id)["rank"].to_list()
    query_ids = grouped_response.get_group(openalex_id)["query_id"].to_list()
    return [{"rank": pair[0], "query_id": pair[1]} for pair in zip(ranks, query_ids)]


if __name__ == "__main__":
    data_dir = Path("data")
    inclusions_response = pd.read_parquet(
        data_dir / "included_records_response.parquet"
    )
    criteria_response = pd.read_parquet(
        data_dir / "inclusion_criteria_response.parquet"
    )
    labeled_data = pd.read_excel(
        data_dir / "Motherfile_280524_relevant_but_not_found_via_openAlex.xlsx"
    )
    labeled_data["short_openalex_id"] = labeled_data["openalex_id"].str.removeprefix(
        "https://openalex.org/"
    )
    labeled_data["text"] = (
        labeled_data.title.fillna("").astype(str)
        + " "
        + labeled_data.abstract.fillna("").astype(str)
    )
    grouped_response = inclusions_response.groupby("id")
    labeled_data["inclusions_recall_data"] = labeled_data.short_openalex_id.apply(
        lambda openalex_id: recall_data(grouped_response, openalex_id)
    )
    labeled_data["inclusions_min_rank_data"] = labeled_data[
        "inclusions_recall_data"
    ].apply(lambda x: min(x, key=lambda y: y["rank"]))
    labeled_data["inclusions_min_rank"] = (
        labeled_data["inclusions_min_rank_data"]
        .apply(lambda x: x["rank"])
        .astype("Int64")
    )
    labeled_data["inclusions_min_rank_id"] = labeled_data[
        "inclusions_min_rank_data"
    ].apply(lambda x: x["query_id"])
    labeled_data["inclusions_recall_data"] = labeled_data[
        "inclusions_recall_data"
    ].apply(lambda x: None if x[0]["rank"] is None else x)

    embeddings_fp = data_dir / "labeled_data_embeddings.parquet"
    if embeddings_fp.exists():
        embeddings_df = pd.read_parquet(embeddings_fp)
    else:
        model_name = "intfloat/multilingual-e5-small"
        prefix = "query: "
        device = "cuda"
        model = SentenceTransformerWithPrefix(model_name, device=device)
        model.set_prefix(prefix)
        model.max_seq_length = 512
        encoding_batch_size = 16

        embeddings = model.encode(
            labeled_data["text"], show_progress_bar=True, batch_size=encoding_batch_size
        )

        labeled_data["embedding"] = embeddings.tolist()
        embeddings_df = labeled_data[["openalex_id", "embedding"]]
        embeddings_df.to_parquet(embeddings_fp)

    inclusion_embeddings_fp = data_dir / "inclusion_embeddings.parquet"
    if inclusion_embeddings_fp.exists():
        inclusion_embeddings_df = pd.read_parquet(inclusion_embeddings_fp)
    else:
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
                df.doi.fillna("").str.lower().str.contains(doi.lower()),
                "extra_excluded",
            ] = True

        df = df[df.label_included.astype(bool) & ~df.extra_excluded]
        df.info()

        # Vectorize the texts of the included records.
        texts = (
            df.title.fillna("").astype(str) + " " + df.abstract.fillna("").astype(str)
        ).to_list()
        included_vectors = model.encode(texts)
        df["embedding"] = included_vectors.tolist()
        inclusion_embeddings_df = df[["id", "embedding"]]
        inclusion_embeddings_df.to_parquet(inclusion_embeddings_fp)

    inclusion_embeddings = np.array(inclusion_embeddings_df["embedding"].to_list())
    inclusion_ids = inclusion_embeddings_df["id"].to_list()
    embeddings = np.array(embeddings_df["embedding"].to_list())
    cossim_matrix = cosine_similarity(embeddings, inclusion_embeddings)
    labeled_data["inclusions_distance"] = cossim_matrix.max(axis=1)
    labeled_data["inclusions_nearest"] = cossim_matrix.argmax(axis=1)
    labeled_data["inclusions_nearest"] = labeled_data["inclusions_nearest"].apply(
        lambda n: inclusion_ids[n]
    )

    criteria_embedding_fp = data_dir / "criteria_embedding.json"
    if criteria_embedding_fp.exists():
        with open(criteria_embedding_fp) as f:
            criteria_embedding = json.load(f)
    else:
        inclusion_criteria = """
        Longitudinal/Prospective study with at least three-time point assessments.
        PTSS are measured with a validated continuous scale (either self-report questionnaire or clinical interview) measuring PTSD symptoms according to the diagnostic criteria specificized in DSM-IV or DSM-5(TR).  Some eligible PTSD scales: Clinician Administered PTSD Scale (CAPS), PTSD Checklist (PCL), Impact of Event Scale (IES), Harvard Trauma Questionnaire (HTQ), Posttraumatic Diagnostic Scale (PDS), UCLA PTSD
        PTSD assessment conducted after a traumatic event as defined by DSM-IV or DSM-V.
        Trajectory analysis of PTSD using one of the following methods: Latent Growth Mixture Modeling (LGMM), Latent Curve Mixture Analysis (LCMA), or other hierarchical cluster analysis.
        """  # noqa: E501
        criteria_embedding = model.encode([inclusion_criteria])[0]
        with open(criteria_embedding_fp, "w") as f:
            json.dump(criteria_embedding.tolist(), f)
    labeled_data["criteria_distance"] = cosine_similarity(
        embeddings, np.array(criteria_embedding).reshape(1, -1)
    )
    rank_mapping = pd.Series(
        criteria_response["rank"].to_list(), index=criteria_response["id"]
    )
    labeled_data["criteria_rank"] = (
        labeled_data["short_openalex_id"]
        .apply(lambda x: rank_mapping.get(x))
        .astype("Int64")
    )

    logistic_reranking_df = pd.read_csv(
        data_dir / "included_records_logistic_dataset.csv"
    )
    logistic_reranking_df["logistic_rank"] = range(len(logistic_reranking_df))
    labeled_data = pd.merge(
        labeled_data,
        logistic_reranking_df[["id", "relevance_score", "logistic_rank"]],
        left_on="openalex_id",
        right_on="id",
        how="left",
    )
    labeled_data = labeled_data.drop("id", axis=1)
    labeled_data = labeled_data.rename({"relevance_score": "logistic_score"}, axis=1)
    labeled_data["logistic_rank"] = labeled_data["logistic_rank"].astype("Int64")

    labeled_data.drop(
        ["short_openalex_id", "text", "inclusions_min_rank_data"],
        axis=1,
        inplace=True,
    )

    labeled_data.to_excel(data_dir / "motherfile_with_ranking.xlsx")

# inclusions = labeled_data[labeled_data["FT_inclusion_Bruno"] == 1]
# recall_at_rank = []
# n_found = 0
# for i in range(5000):
#     n_found += inclusions["min_rank"].eq(i).sum()
#     recall_at_rank.append(n_found)


# fig, ax = plt.subplots()
# ax.plot(range(5000), recall_at_rank)
# ax.set_title("Recall at n")
# ax.set_xlabel("n")
# ax.set_ylabel("Recall at n")
# ax.hlines(y=len(inclusions), xmin=0, xmax=4999, color="r")
# yticks = [tick for tick in ax.get_yticks() if tick < len(inclusions)] + [
#     len(inclusions)
# ]
# ax.set_yticks(yticks)
# fig.savefig("recall_at_n.png")
