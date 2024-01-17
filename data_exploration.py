import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns


OPENALEX_PREFIX = "https://openalex.org/"

data_dir = Path("data")
original_df = pd.read_csv(data_dir / "synergy" / "van_de_Schoot_2018.csv")
original_df["id"] = original_df.id.str.removeprefix(OPENALEX_PREFIX)
included_df = pd.read_parquet(data_dir / "included_records_response.parquet")
included_df["query_id"] = included_df.query_id.str.removeprefix(OPENALEX_PREFIX)
criteria_df = pd.read_parquet(data_dir / "inclusion_criteria_response.parquet")
citations_df = pd.read_csv(data_dir / "citations.csv")
citations_df["id"] = citations_df.id.str.removeprefix(OPENALEX_PREFIX)


## Number of unique records:

included_df.id.nunique()
# 57232
included_df.id.value_counts()
# id
# W2885152410    34
# W3030052793    34
# W2769415291    34
# W2938412282    34
# W4386552853    33
#                ..
# W3154600247     1
# W4297457475     1
# W3006670027     1
# W2883302710     1
# W4292356720     1
included_df.id.value_counts().value_counts()[::-1]
# count
# 34        4
# 33        8
# 31       24
# 29       26
# 32       29
# 30       36
# 28       38
# 25       51
# 27       56
# 26       58
# 24       66
# 23       73
# 22       86
# 21       92
# 20      116
# 19      142
# 18      151
# 17      184
# 16      210
# 15      267
# 14      313
# 13      332
# 12      409
# 11      512
# 10      610
# 9       765
# 8       856
# 7      1117
# 6      1431
# 5      1786
# 4      2505
# 3      4085
# 2      7944
# 1     32850

unique_records_by_rank = []
for max_rank in range(5000):
    unique_records_by_rank.append(
        included_df[included_df["rank"].le(max_rank)].id.nunique()
    )

dif_values = [0] + [
    unique_records_by_rank[i+1] - unique_records_by_rank[i]
    for i in range(len(unique_records_by_rank) - 1)
]
pd.Series(dif_values, name="n_unique_values_at_rank").plot(
    kind="line",
    title="Unique records by rank",
    xlabel="Rank",
    ylabel="New records at rank",
    figure=plt.figure()
).get_figure().savefig("n_unique_at_rank.png")

id_sets = {}
for query_id in included_df.query_id.unique():
    id_sets[query_id] = set(included_df[included_df.query_id.eq(query_id)].id.values)

overlap_df = pd.DataFrame(
    [[len(id_sets[q1].intersection(id_sets[q2])) for q2 in id_sets] for q1 in id_sets]
)
fig = sns.clustermap(overlap_df)
fig.ax_row_dendrogram.set_visible(False)
fig.ax_row_dendrogram.set_xlim([0,0])
fig.ax_col_dendrogram.set_visible(False)
fig.savefig("overlap_clustered.png")


# First above 7000: 427
included_df[included_df["rank"].lt(427)].id.nunique()
# 7000

included_df[included_df["rank"].lt(4270)].id.nunique()
# 50055


# Overlap between datasets
included_df[included_df.id.isin(original_df.id)].id.nunique()
# 315
included_df[included_df.id.isin(criteria_df.id)].id.nunique()
# 4173
included_df[included_df.id.isin(citations_df.id)].id.nunique()
# 2871
criteria_df[criteria_df.id.isin(original_df.id)].id.nunique()
# 97
criteria_df[criteria_df.id.isin(citations_df.id)].id.nunique()
# 533

primary_citations_df = citations_df[citations_df.level.eq("primary")]
included_df[included_df.id.isin(primary_citations_df.id)].id.nunique()
# 281
