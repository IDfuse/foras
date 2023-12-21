import os
from pathlib import Path

from datasets import load_dataset, disable_caching
from dotenv import load_dotenv

load_dotenv()
disable_caching()


data_files = {
    "train": str(
        Path(
            os.environ["DATA_DIR"],
            "foras",
            "updated_date=2023-10-20",
            "intfloat__multilingual-e5-small",
            "embeddings_*.parquet",
        )
    )
}

dataset = load_dataset(
    "parquet",
    data_files=data_files,
    split="train",
    cache_dir=str(Path(os.environ["DATA_DIR"], "foras", ".cache", "huggingface", "datasets")),
).select_columns(["id", "embedding"])
dataset.push_to_hub("GlobalCampus/openalex-multilingual-embeddings", private=True)
