import os

# ─── Directory paths ──────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")

for _d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
RAW_DATA_FILE  = os.path.join(DATA_RAW, "dataset.xlsx")
LABEL_MAP      = {0: "negatif", 1: "positif", 2: "netral"}
NUM_LABELS     = len(LABEL_MAP)

# ─── BERT model ───────────────────────────────────────────────────────────────
# bert-base-multilingual-cased: pre-trained on 104 languages incl. Indonesian
PRETRAINED_MODEL = "bert-base-multilingual-cased"
MODEL_SHORT_NAME = "mBERT"   # used in filenames / report labels

# ─── Training hyperparameters ─────────────────────────────────────────────────
MAX_LEN        = 128      # max token length per tweet
BATCH_SIZE     = 16
EPOCHS         = 5
LEARNING_RATE  = 2e-5
WARMUP_RATIO   = 0.1
WEIGHT_DECAY   = 0.01
DROPOUT        = 0.3

# ─── Train / val / test split ────────────────────────────────────────────────
TEST_SIZE      = 0.15     # 15 % test
VAL_SIZE       = 0.15     # 15 % validation (from remaining train set)
RANDOM_STATE   = 42

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42