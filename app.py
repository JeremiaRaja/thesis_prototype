from src.config import DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, PRETRAINED_MODEL, MODEL_SHORT_NAME

def main():
    print("Project thesis_prototype berhasil dijalankan")
    print(f"Model            : {PRETRAINED_MODEL}  ({MODEL_SHORT_NAME})")
    print("Data raw folder  :", DATA_RAW)
    print("Data processed   :", DATA_PROCESSED)
    print("Models folder    :", MODELS_DIR)
    print("Results folder   :", RESULTS_DIR)

    print("\nAvailable commands:")
    print("  python -m src.preprocessing   → clean data & create train/val/test splits")
    print(f"  python -m src.train           → fine-tune {MODEL_SHORT_NAME} (GPU recommended)")
    print(f"  python -m src.compare         → compare {MODEL_SHORT_NAME} vs Naive Bayes / SVM / LR")

if __name__ == "__main__":
    main()
