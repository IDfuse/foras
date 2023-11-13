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