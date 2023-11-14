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
- Make sure that Docker is istalled. 
- Create a file `.env` in the root of the repository and copy the contents of `.env.example`. Fill in the desired values. 
    - `PORT`: Port on which the database will listen to queries.
    - `ADMIN_PORT`: Port where the database listens to admin operations.
    - `DATA_DIR`: This directory will be used to store the data downloaded from OpenAlex and the files created after processing this data.
    - `MOUNT_DIR`: There will be two directories mounted on the container, `$MOUNT_DIR/var` and `$MOUNT_DIR/logs`, which will contain the database and the database logs.
    - `VESPA_VERSION`: Which version of Vespa to use.
- From the root of the repository run `docker-compose up -d`. This starts the container.
- Run `docker exec vespa vespa deploy /srv/app`. This will deploy the database.

Now the database is ready for feeding or querying.
