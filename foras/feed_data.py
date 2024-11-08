import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from vespa.application import Vespa
from vespa.io import VespaResponse

load_dotenv()

formatter = logging.Formatter("%(asctime)15s - %(levelname)-8s - %(message)s")
base_handler = logging.StreamHandler()
base_handler.setFormatter(formatter)
base_handler.setLevel(logging.INFO)
error_handler = logging.FileHandler(os.environ.get("FEED_LOG_FP", "feed.log"))
error_handler.setFormatter(formatter)
error_handler.setLevel(logging.ERROR)
logger = logging.getLogger("feed")
logger.setLevel(logging.INFO)
logger.addHandler(base_handler)
logger.addHandler(error_handler)


MODEL_NAME = "intfloat/multilingual-e5-small"
data_dir = Path(
    os.environ["DATA_DIR"],
    "foras",
    "updated_date=2023-10-20",
    MODEL_NAME.replace("/", "__"),
)
app = Vespa(url=f"http://{os.environ['VESPA_IP']}", port=os.environ["PORT"])


def feeding_callback(response: VespaResponse, id: str) -> None:
    if not response.is_successful():
        logger.error(
            f"Failed to feed document {id} with status code"
            f" {response.status_code}: Reason {response.get_json()}"
        )


def data_generator(df: pd.DataFrame) -> dict:
    for _, row in df.iterrows():
        yield {
            "id": row["id"],
            "fields": {"id": row["id"], "embedding": row["embedding"]},
        }


total = 0
for embeddings_fp in sorted(
    data_dir.glob("embeddings_*.parquet"), key=lambda x: x.stem
):
    part_nr = embeddings_fp.stem.replace("embeddings_", "")
    logger.info(f"Feeding {part_nr}")
    metadata_fp = Path(data_dir, f"data_{part_nr}.parquet")
    embeddings = pd.read_parquet(embeddings_fp)
    metadata = pd.read_parquet(metadata_fp)
    dataset = pd.concat(
        [embeddings[["id", "embedding"]], metadata["publication_year"]], axis=1
    )
    dataset = dataset[dataset.publication_year >= 2015][["id", "embedding"]]
    if len(dataset) == 0:
        continue
    dataset["embedding"] = dataset.embedding.apply(lambda x: x.tolist())
    app.feed_iterable(
        iter=data_generator(dataset),
        schema="works",
        callback=feeding_callback,
    )
    total += len(dataset)
    logger.info(f"Number of document fed: {len(dataset)}. Total: {total}")
