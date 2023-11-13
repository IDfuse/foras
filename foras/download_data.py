import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)


def get_manifest() -> dict:
    """Get the manifest describing the Works data.

    (See:
    https://docs.openalex.org/download-all-data/snapshot-data-format#the-manifest-file)

    Returns
    -------
    dict
        Dictionary of the form:
        ```
        {
            'entries': [{'url': file_url, 'meta': meta_dict}, ...],
            'meta': meta_dict,
        }
        ```
        where `meta_dict` is a dictionary of the form
        ```
        {
            'content_length': int,
            'record_count': int,
        }
        ```
    """
    url = "https://openalex.s3.amazonaws.com/data/works/manifest"
    res = requests.get(url)
    if res.status_code == 404:
        raise FileNotFoundError(
            "Manifest file for works was not found in the OpenAlex S3 bucket. "
            "Maybe OpenAlex is updating the data?"
        )
    res.raise_for_status()
    return res.json()


def download_works(save_dir: Path) -> None:
    """Download all the work files from the OpenAlex S3 bucket.

    Parameters
    ----------
    save_dir : Path
        Directory where to save the files. The files will be saved in
        `save_fp / updated_date=YYYY-MM-DD / file_name`.
    """
    manifest = get_manifest()
    for file_data in sorted(manifest["entries"], key=lambda x: x["url"]):
        bucket_url = file_data["url"]
        fp = get_fp_from_url(url=bucket_url, save_dir=save_dir)
        directory_part = bucket_url.removeprefix("s3://openalex/")
        url = f"https://openalex.s3.amazonaws.com/{directory_part}"
        if not fp.parent.exists():
            logging.info(f"Making directory {fp.parent}")
            fp.parent.mkdir()
        logging.info(f"Downloading {url}")
        res = requests.get(url)
        res.raise_for_status()
        with open(fp, "wb") as f:
            f.write(res.content)


def get_fp_from_url(url: str, save_dir: Path) -> Path:
    """Get the save path for a file from it's S3 url.

    Parameters
    ----------
    url : str
        URL of the file in the OpenAlex S3 bucket.
    save_dir : Path
        Directory where to save all files.

    Returns
    -------
    Path
        Path of the form `save_fp / updated_date=YYYY-MM-DD / file_name`.
    """
    url_parts = url.split("/")
    updated_dir = url_parts[-2]
    file_name = url_parts[-1]
    return Path(save_dir, updated_dir, file_name)


if __name__ == "__main__":
    save_dir = Path(Path.cwd(), "data", "source_data")
    if not save_dir.exists():
        logging.info(f"Making directory {save_dir}")
        save_dir.mkdir(parents=True)
    download_works(save_dir=save_dir)
