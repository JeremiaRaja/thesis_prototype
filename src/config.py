import os

# ─── Directory paths ──────────────────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW            = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED      = os.path.join(BASE_DIR, "data", "processed")
FORMAL_PROCESSED    = os.path.join(DATA_PROCESSED, "formal")
INFORMAL_PROCESSED  = os.path.join(DATA_PROCESSED, "informal")
MODELS_DIR          = os.path.join(BASE_DIR, "models")
RESULTS_DIR         = os.path.join(BASE_DIR, "results")

for _d in [DATA_RAW, FORMAL_PROCESSED, INFORMAL_PROCESSED, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
RAW_DATA_FILE    = os.path.join(DATA_RAW, "dataset.xlsx")
FORMAL_FILE      = os.path.join(DATA_RAW, "dataset_formal.xlsx")
INFORMAL_FILE    = os.path.join(DATA_RAW, "dataset_informal.xlsx")

LABEL_MAP        = {0: "negatif", 1: "positif", 2: "netral"}
NUM_LABELS       = len(LABEL_MAP)

# ─── IndoBERT model ───────────────────────────────────────────────────────────
PRETRAINED_MODEL = "indobenchmark/indobert-base-p1"
MODEL_SHORT_NAME = "IndoBERT"

# ─── Training hyperparameters ─────────────────────────────────────────────────
MAX_LEN          = 128
BATCH_SIZE       = 16
EPOCHS           = 5
LEARNING_RATE    = 2e-5
WARMUP_RATIO     = 0.1
WEIGHT_DECAY     = 0.01
DROPOUT          = 0.3

# ─── Train / val / test split ─────────────────────────────────────────────────
TEST_SIZE        = 0.15
VAL_SIZE         = 0.15
RANDOM_STATE     = 42
SEED             = 42
