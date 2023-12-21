# Foras
Create datasets for the FORAS project using OpenAlex

## Downloading the OpenAlex data
To download the OpenAlex data on works, run `python foras/download_data.py` from the
root of the repository. This will download all the files and put them in a directory
called `data`. Note that this is approximately 330GB of data (October 2023 snapshot).

## Pre-existing OpenAlex embeddings
There are no suitable pre-existing OpenAlex embeddings.
The only openly available set of OpenAlex embeddings I could find was here:
https://github.com/colonelwatch/abstracts-search/
They used the 'all-MiniLM-L6-v2' sentence-transformers model. This model is purely
English, so at least we need different embeddings for the multilingual embeddings.
Moreover there are better purely english models available (the chosen model is faster).
Finally it looks like only the first 128 tokens were used to create the embeddings.
Using the maximum possible 512 tokens should give better results. 

## Choosing an embedding model
I checked the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard),
looking at the subset of models that has 'multilingual' in the name. The best performing
models for the Semantic Text Similarity (STS) category were:
- https://huggingface.co/intfloat/multilingual-e5-large with 1024-D vectors
- https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2 with 768-D vectors
- https://huggingface.co/intfloat/multilingual-e5-small with 384-D vectors
Here I ignored models that were not open source, or at first sight not compatible with
the sentence_transformers package, or that only support a small set of languages.

In local testing, processing with the smallest model took ~3h per file of 850MB,
the medium model ~9h and for the largest model I had to lower the encoding batch size,
when the process would take ~21h.

## Starting the application
- Make sure that Docker is installed
- Create a file `.env` in the root of the repository and copy the contents of `.env.example`. Fill in the desired values. 
    - `PORT`: Port on which the database will listen to queries.
    - `ADMIN_PORT`: Port where the database listens to admin operations.
    - `DATA_DIR`: This directory will be used to store the data downloaded from OpenAlex and the files created after processing this data.
    - `MOUNT_DIR`: This can be the absolute path to the `mount` directory in this repository, or any other preferred directory. There will be two directories mounted on the container, `$MOUNT_DIR/var` and `$MOUNT_DIR/logs`, which will contain the database and the database logs. Make sure this directory is owned by user/group `1000:1000`.
    - `VESPA_VERSION`: Which version of Vespa to use.
- From the root of the repository run `docker compose up -d`. This starts the container.
- Run `docker exec foras-vespa-1 vespa deploy /srv/app`. This will deploy the database.

Now the database is ready for feeding or querying.

## Size of the vectorized dataset.
Total records in OpenAlex: 246M
Records with publication_year >= 2015: 85M
Records with abstract: 126M
Records with abstract and publication_year >= 2015: 49M

## Feeding
To feed the identifiers and embeddings to the application, set the `VESPA_IP` environment variable in `.env` and run `python foras/feed_data.py`.

## Getting the original dataset
- `python -m synergy_dataset get -d van_de_Schoot_2018 -o $DATA_DIR/synergy -v 'doi,title,abstract,id` and say yes to converting inverted abstract to plaintext. Here `$DATA_DIR` should be be replaced by the same path as in `.env`.

## Citations dataset
The script `foras/find_citations.py` can be used to get a dataset containing the works in OpenAlex that directly reference one of the included records in the original dataset, or the works that reference one of the directly referencing works. It needs two variables from the `.env` file:
- `DATA_DIR`: The directory where to put the dataset. It will end up at `$DATA_DIR/citations.csv`.
- `OPENALEX_EMAIL`: Optional, used in `find_citations.py`. Email adress to send along with API calls to OpenAlex. See: https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication#the-polite-pool

In short, what the script does is:
- Read the original dataset (the `van_de_Schoot_2018` dataset from Synergy)
- Mark the records with the following 4 DOI's as excluded:
    - "10.1037/a0020809"
    - "10.1097/BCR.0b013e3181cb8ee6"
    - "10.1007/s00520-015-2960-x"
    - "10.1016/j.pain.2010.02.013"
- For each of the included works, collect all works in OpenAlex that reference to it, and mark them as `primary`.
- For each of the primary records, collect all works in OpenAlex that reference to it, and mark them as `secondary`.
- Combine this into a single CSV file.

This CSV file has the columns:
- `id`: OpenAlex identifier.
- `doi`
- `title`
- `abstract`
- `referenced_works`: List of OpenAlex identifiers.
- `publication_date`
- `level`: Can have two values:
    - `primary`: Directly references one of the included works of the original dataset.
    - `secondary`: References one of the works in the set of primary records.

For the `van_de_Schoot_2018` dataset this gives the following numbers at the time of writing (2023-18-12):

|        | Total | Primary | Secondary |
|--------|-------|---------|-----------|
| All    | 9016  | 465     | 8551      |
| >=2015 | 8682  | 451     | 8231      |
