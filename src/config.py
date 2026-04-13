from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_LABELS = 3
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5