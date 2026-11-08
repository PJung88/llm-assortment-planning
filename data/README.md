# Data

## Source

Product metadata comes from the **Amazon Reviews 2023** dataset (McAuley Lab, UC San Diego),
category *Grocery and Gourmet Food*, file `meta_Grocery_and_Gourmet_Food.jsonl`.

> Hou, Y., Li, J., Fu, X., He, Z., Yan, A., Chen, X., & McAuley, J. (2024).
> *Bridging Language and Items for Retrieval and Recommendation.* arXiv:2403.03952.

Dataset page: <https://amazon-reviews-2023.github.io/> ·
Hugging Face: <https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023>

## What is included here

| Folder | Contents |
|---|---|
| `processed/` | The 60 sampled coffee products (title, features, description, details, price) in the forms used by each stage, plus the sampling report |
| `../outputs/` | Results derived from those 60 products: embedding scores, LLM rankings with the models' one-line reasoning, extracted attributes, shares, scenarios and assortment results |

The full category file (about 1.4 GB) and the intermediate filter outputs for the
7,406-product coffee pool are **not** included. To rebuild Stage 01 from scratch,
download the metadata file into `data/raw/` and run `src/pipeline/01_filter_and_sample.py`.
Stages 07-10 do not need it.

## Terms

The dataset maintainers state that they are not in a position to assign a license and
that the data is available primarily for research purposes. The 60-product subset is
included here only so the research can be reproduced and inspected. It is not covered
by this repository's MIT license, which applies to the source code only. If you are a
rights holder and want something removed, please open an issue.
