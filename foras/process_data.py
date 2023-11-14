import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
OA_PREFIX = "https://openalex.org/"


class EmptyDataFrameError(Exception):
    pass


def process_works(
    fp: Path,
    save_dir: Path,
    embedding_save_dir: Path,
    model: SentenceTransformer,
    chunksize: int,
) -> None:
    df_list = []
    embedding_list = []
    for idx, df in enumerate(
        pd.read_json(
            fp,
            lines=True,
            chunksize=chunksize,
            # Specify dtype, otherwise is all abstracts are missing it becomes float.
            dtype={
                "abstract_inverted_index": dict,
            },
        )
    ):
        logging.info(f"Processing chunk {idx}")
        try:
            processed_df, embeddings = process_chunk(df=df, model=model)
        except EmptyDataFrameError:
            continue
        df_list.append(processed_df)
        embedding_list.append(embeddings)

    df = pd.concat(df_list)
    if df.empty:
        raise EmptyDataFrameError
    embeddings = np.vstack(embedding_list)

    # Replace all missing values by None instead of other Pandas missing formats.
    df.replace(float("nan"), None, inplace=True)

    updated_date = fp.parent.stem.split("=")[-1]
    part_nr = fp.stem.split("_")[-1]
    text_save_fp = Path(save_dir, f"text_data_{updated_date}_{part_nr}.csv")
    embedding_save_fp = Path(
        embedding_save_dir, f"embeddings_{updated_date}_{part_nr}.npy"
    )

    # It might be worth it to put it in a different format that is better suited for
    # dealing with large quantities of numpy array data. For example, there is
    # `numpy.savez_compressed``, and Huggingface Datasets uses Arrow Parquet files in
    # the background. Depending on where we want to put the data it might need to be
    # very small, or easy to read from disk, or easy to append to / read in chunks.
    df.to_csv(
        text_save_fp,
        index=False,
    )
    np.save(embedding_save_fp, embeddings)


def process_chunk(
    df: pd.DataFrame, model: SentenceTransformer
) -> tuple[pd.DataFrame, np.ndarray]:
    """Process one dataframe of OpenAlex data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with OpenAlex works data.
    model : SentenceTransformer
        Model to create the embeddings with.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Tuple (dataframe, embeddings) where the dataframe contains the columns
        'id', 'title', 'abstract'. The identifier is in short form (so 'W12345' instead
        of 'https://openalex.org/W12345').
    """
    if df.empty:
        raise EmptyDataFrameError
    df["abstract"] = df.abstract_inverted_index.apply(construct_abstract)
    df = df.loc[df.title.notnull() | df.abstract.notnull()]
    df.loc[:, "id"] = df.id.str.removeprefix(OA_PREFIX)
    texts = (df.title.fillna("") + " " + df.abstract.fillna("")).to_list()
    embeddings = model.encode(texts, show_progress_bar=True)
    used_columns = ["id", "title", "abstract"]
    df = df[used_columns]
    return df, embeddings


def construct_abstract(
    abstract_inverted_index: dict[str, list[int]] | None
) -> str | None:
    """Construct an abstract from an abstract inverted index.

    Parameters
    ----------
    abstract_inverted_index : dict[str, list[int]] | None
        Dicitonary of the form {token: [list of token indices]}

    Returns
    -------
    str
        Abstract.
    """
    if pd.isnull(abstract_inverted_index):
        return None
    token_index_pairs = [
        (token, idx)
        for token, indices in abstract_inverted_index.items()
        for idx in indices
    ]
    token_index_pairs.sort(key=lambda x: x[1])
    abstract = " ".join(x[0] for x in token_index_pairs)
    if len(abstract) == 0:
        return None
    else:
        return abstract


class SentenceTransformerWithPrefix(SentenceTransformer):
    """Wrapper around a sentence transformers model that prefixes texts by a common
    prefix before passing them to encode. If prefix=None, the behavior is the same as a
    normal sentence_transformers model.
    
    Example
    -------
    The model `intfloat/multilingual-e5-small` expects the input texts to have the
    prefix 'query: '. The model card states that performance is worse if this prefix is
    not added.
    """

    prefix = None

    def set_prefix(self, prefix: str) -> None:
        self.prefix = prefix

    def encode(self, sentences, **kwargs):
        if self.prefix is not None:
            sentences = [self.prefix + sentence for sentence in sentences]
        return super().encode(sentences=sentences, **kwargs)


if __name__ == "__main__":
    data_dir = Path(Path.cwd(), "data", "source_data")
    save_dir = Path(Path.cwd(), "data", "processed_data")
    if not save_dir.exists():
        logging.info(f"Making directory {save_dir}")
        save_dir.mkdir(parents=True)

    # MODEL_NAME = "intfloat/multilingual-e5-large"
    # PREFIX = "query: "
    # # Takes ~21h for a single part file of ~850MB. 
    # # Needed to lower encode batch_size to 16.
    # MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # PREFIX = None
    # # Takes ~9h for a single part file of ~850MB
    MODEL_NAME = "intfloat/multilingual-e5-small"
    PREFIX = "query: "
    # Takes ~3h for a single part file of ~850MB.
    # MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # PREFIX = None
    # # Takes ~3h for a single part file of ~850MB.

    DEVICE = "cuda"
    CHUNKSIZE = 5 * 10**4

    logging.info("Loading model")
    model = SentenceTransformerWithPrefix(MODEL_NAME, device=DEVICE)
    model.set_prefix(PREFIX)
    # Make sure to set max_seq_length to maximum allowed. Some sentence_transformers
    # models set this lower than allowed, but by experience getting all the tokens is
    # more important than the slight decrease in model performance.
    model.max_seq_length = 512

    embedding_save_dir = Path(save_dir, MODEL_NAME.replace("/", "__"))
    if not embedding_save_dir.exists():
        logging.info(f"Making directory {embedding_save_dir}")
        embedding_save_dir.mkdir(parents=True)

    for fp in sorted(
        data_dir.glob("*/part_*.gz"), key=lambda x: (x.parent.stem, x.stem)
    ):
        logging.info(f"Processing {str(fp)}")
        try:
            process_works(
                fp=fp,
                save_dir=save_dir,
                embedding_save_dir=embedding_save_dir,
                model=model,
                chunksize=CHUNKSIZE,
            )
        except EmptyDataFrameError:
            logging.warning(f"File {fp} does not contain rows with title or abstract")
            continue
