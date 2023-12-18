# %%
import os
from pathlib import Path

import pandas as pd
import pyalex
from dotenv import load_dotenv
from pyalex import Works

load_dotenv()

pyalex.config.email = os.environ.get("OPENALEX_EMAIL")


# %%
def get_citing_works(identifiers: list[str]) -> pd.DataFrame:
    """Get all works citing a work with the OpenAlex identifier from the list.

    Parameters
    ----------
    identifiers : list[str]
        List of OpenAlex identifiers.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns ["id", "doi", "title", "abstract", "referenced_works"].
        The "id" columns contains the OpenAlex identifier of the citing work.
        The "cites" column contains the identifier in "identifiers" of the work
        that it cites.
    """
    used_columns = [
        "id",
        "doi",
        "title",
        "abstract_inverted_index",
        "referenced_works",
        "publication_date",
    ]
    citing_works = []
    for idx, openalex_id in enumerate(identifiers):
        print(f"{idx}. Getting cited works for {openalex_id}")
        works_citing_id = Works().filter(cites=openalex_id).select(used_columns).get()
        citing_works += [
            {
                key: work[key]
                for key in [
                    col if col != "abstract_inverted_index" else "abstract"
                    for col in used_columns
                ]
            }
            for work in works_citing_id
        ]
    df = pd.DataFrame(citing_works)
    df.drop_duplicates("id", inplace=True, ignore_index=True)
    return df

# %%
data_dir = Path(os.environ["DATA_DIR"])
df = pd.read_csv(data_dir / "synergy" / "van_de_Schoot_2018.csv")

# %%
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

included = df[df.label_included.astype(bool) & ~df.extra_excluded].copy()
included.info()


# %%
primary_cites_df = get_citing_works(included.id)
secondary_cites_df = get_citing_works(primary_cites_df.id)
primary_cites_df["level"] = "primary"
secondary_cites_df["level"] = "secondary"
cites_df = pd.concat([primary_cites_df, secondary_cites_df])
cites_df.drop_duplicates("id", keep="first", inplace=True, ignore_index=True)
in_original = cites_df.id.apply(lambda x: x in df.id.to_list())
cites_df = cites_df[~in_original]
cites_df.to_csv(data_dir / "citations.csv")
# %%
