from src.config import PRETRAINED_MODEL, MODEL_SHORT_NAME, DATA_RAW, RESULTS_DIR

def main():
    print("=" * 55)
    print("  Thesis — IndoBERT Sentiment Analysis (PPKM Tweets)")
    print("=" * 55)
    print(f"  Model   : {PRETRAINED_MODEL}")
    print(f"  Data    : {DATA_RAW}")
    print(f"  Results : {RESULTS_DIR}")
    print()
    print("Run order:")
    print("  1. python -m src.split_data")
    print("  2. python -m src.preprocessing")
    print("  3. python -m src.train --mode formal")
    print("  4. python -m src.train --mode informal")
    print("  5. python -m src.compare")

if __name__ == "__main__":
    main()
