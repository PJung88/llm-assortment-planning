"""
Central config for the cold-start substitution pipeline.

All paths are anchored to the repository root (the folder that contains
src/), so scripts work from any working directory.

Override DATA_DIR via the SUBSTITUTION_DATA_DIR environment variable if
you need to point at a different data location.
"""
import os

# Repository root = the folder that contains src/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Root directories
DATA_DIR    = os.environ.get(
    "SUBSTITUTION_DATA_DIR",
    os.path.join(_PROJECT_ROOT, "data"),
)
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")

# Raw data
RAW_FILE = os.path.join(DATA_DIR, "raw", "meta_Grocery_and_Gourmet_Food.jsonl")

# Interim: one file per pipeline filter stage
INTERIM_DIR          = os.path.join(DATA_DIR, "interim")
COFFEE_ONLY_FILE     = os.path.join(INTERIM_DIR, "CoffeeOnly.jsonl")
FIELD_FILE           = os.path.join(INTERIM_DIR, "Field.jsonl")
PRICE_COFFEE_FILE    = os.path.join(INTERIM_DIR, "PriceCoffee.jsonl")
TEXT_FILTERED_FILE   = os.path.join(INTERIM_DIR, "TextFilteredCoffee.jsonl")

# Processed: final 60-SKU exports
PROCESSED_DIR        = os.path.join(DATA_DIR, "processed")
SAMPLE_FILE          = os.path.join(PROCESSED_DIR, "StratifiedSample60.jsonl")
REPORT_FILE          = os.path.join(PROCESSED_DIR, "StratifiedSelectionReport.txt")
MODEL_INPUT_FILE           = os.path.join(PROCESSED_DIR, "ModelInput60.jsonl")
FORMATTED_MODEL_INPUT_FILE = os.path.join(PROCESSED_DIR, "ModelInput60_formatted.jsonl")
EMBEDDING_INPUT_FILE       = os.path.join(PROCESSED_DIR, "EmbeddingInput60.jsonl")

# Outputs: one subfolder per pipeline stage
ALL_PAIRS_SCORES     = os.path.join(OUTPUTS_DIR, "embeddings", "all_pairs_scores.parquet")

# Stage 05 - ranking elicitation outputs
# raw_rankings.parquet: 960 rows = 4 models x 60 SKUs x 4 candidate orderings.
RANKINGS_DIR         = os.path.join(OUTPUTS_DIR, "rankings")
RANKINGS_FILE        = os.path.join(RANKINGS_DIR, "raw_rankings.parquet")

# Pipeline parameters
RANDOM_SEED    = 42
LENGTH_FLOOR   = 295

TARGET_CATEGORY_PREFIX    = ["Grocery & Gourmet Food", "Beverages", "Coffee"]
EXCLUDE_CATEGORY_TOKENS   = {"coffee substitutes", "gift baskets", "gifting", "gifts"}
KEEP_FIELDS               = ["title", "features", "description", "price",
                              "categories", "details", "parent_asin"]

QUOTAS = {
    "Ground Coffee":   {"Budget": 5, "Mid-Range": 5, "Premium": 5},
    "Whole Bean":      {"Budget": 5, "Mid-Range": 5, "Premium": 5},
    "Pods / Capsules": {"Budget": 5, "Mid-Range": 5, "Premium": 5},
    "Instant Coffee":  {"Budget": 3, "Mid-Range": 3, "Premium": 3},
    "RTD & Cold Brew": {"Budget": 2, "Mid-Range": 2, "Premium": 2},
}

PATH_TO_STRATUM = {
    "Grocery & Gourmet Food > Beverages > Coffee > Single-Serve Capsules & Pods": "Pods / Capsules",
    "Grocery & Gourmet Food > Beverages > Coffee > Ground Coffee":                 "Ground Coffee",
    "Grocery & Gourmet Food > Beverages > Coffee > Whole Coffee Beans > Roasted Coffee Beans":   "Whole Bean",
    "Grocery & Gourmet Food > Beverages > Coffee > Whole Coffee Beans > Unroasted Coffee Beans": "Whole Bean",
    "Grocery & Gourmet Food > Beverages > Coffee > Whole Coffee Beans":            "Whole Bean",
    "Grocery & Gourmet Food > Beverages > Coffee > Instant Coffee":                "Instant Coffee",
    "Grocery & Gourmet Food > Beverages > Coffee > Iced Coffee & Cold-Brew":       "RTD & Cold Brew",
    "Grocery & Gourmet Food > Beverages > Coffee > Ready to Drink Liquid Coffee":  "RTD & Cold Brew",
}
UNCATEGORIZED_PATH = "Grocery & Gourmet Food > Beverages > Coffee"

KEYWORD_RULES = [
    ("Pods / Capsules", ["pod", "capsule", "k-cup", "kcup", "keurig"]),
    ("Instant Coffee",  ["instant"]),
    ("RTD & Cold Brew", ["ready to drink", "cold brew", "cold-brew",
                          "iced coffee", "canned coffee", "bottled coffee", "rtd"]),
    ("Ground Coffee",   ["ground coffee", "ground"]),
    ("Whole Bean",      ["whole bean", "whole beans", "coffee beans",
                          "roasted bean", "unroasted bean", "green coffee bean"]),
]

# Ranking elicitation parameters (Stage 05)

# Candidate retrieval: top-k embedding neighbours considered per OOS SKU.
# These candidates are what the LLM ranks - not all 59 other SKUs.
# k=7 keeps each ranking task short enough to judge reliably.
EMBEDDING_TOP_K = 7

# Number of times each model ranks each OOS SKU's candidate set.
# 4 permutations: original order, fully reversed, and two seeded random
# shuffles. The original + reversed pair guarantees every candidate appears
# in both the first and last position at least once, directly targeting
# primacy and recency bias. The two random shuffles cover interior positions.
RANKING_REPEATS = 4

# 4-model design: 2 frontier (API) + 2 open-weight (Ollama).
# Four different model families, so no single training lineage dominates.
# These are the four models in outputs/rankings/raw_rankings.parquet.
# Keys used by 05_ranking_elicit.py: GEMINI_API_KEY, ANTHROPIC_API_KEY
# (OPENAI_API_KEY only if a gpt-* model is added; there are no gpt-* rows.)
# The API models and the Ollama models ran on different machines with identical prompt code.

RANKING_MODELS_FRONTIER = [
    "gemini-3.6-flash",      # Google - set GEMINI_API_KEY
    "claude-sonnet-4-5",     # Anthropic - set ANTHROPIC_API_KEY
]

# Open-weight models via Ollama (OLLAMA_URL can point at a remote machine).
# Pull with: ollama pull <model_name>
RANKING_MODELS_LOCAL = [
    "qwen3.6:27b",         # Alibaba
    "deepseek-r1:32b",     # DeepSeek - distilled reasoning model
]

# All four models in one list - used by 05_ranking_elicit.py (--model check).
RANKING_MODELS_ALL = RANKING_MODELS_FRONTIER + RANKING_MODELS_LOCAL

# Ollama base URL - override with the OLLAMA_URL environment variable.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# LLM generation settings for ranking
RANKING_TEMPERATURE = 0.7   # some variance, but low enough to keep the JSON list stable.
                             # Used for Gemini and the Ollama models. The Claude call
                             # passes no temperature, so claude-sonnet-4-5 ran at the
                             # API default.
RANKING_MAX_TOKENS  = 4096   # enough for a ranked list of ~10 items with reasoning
