"""FraudShield — Live Inference Demo for Phases 1-3."""
import json
import joblib
import pandas as pd

from ml.config import ARTIFACTS_DIR, PCA_FEATURES, TARGET
from ml.data import load_data, time_based_split
from ml.features import FeaturePreprocessor


def run_demo():
    # 1. Load artifacts
    model = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    preprocessor = FeaturePreprocessor.load()
    with open(ARTIFACTS_DIR / "threshold.json") as f:
        thresh_data = json.load(f)
    threshold = thresh_data["threshold"]

    # 2. Load test split
    df = load_data()
    _, _, test_df = time_based_split(df)

    feature_cols = ["Time", "Amount"] + PCA_FEATURES

    # 3. Pick 2 real transactions from the test set
    legit_sample = test_df[test_df[TARGET] == 0].iloc[10]
    fraud_sample = test_df[test_df[TARGET] == 1].iloc[0]

    print("=" * 65)
    print("FRAUDSHIELD — LIVE INFERENCE DEMO")
    print("=" * 65)
    print("Production Model:    XGBoost")
    print(f"Optimal Threshold:   {threshold:.4f} (minimizes FN/FP business cost)")
    print("=" * 65)

    samples = [
        ("Legitimate Transaction", legit_sample),
        ("Fraudulent Transaction", fraud_sample),
    ]

    for label, sample in samples:
        sample_df = pd.DataFrame([sample[feature_cols]])
        X_sample = preprocessor.transform(sample_df)
        fraud_prob = float(model.predict_proba(X_sample)[0, 1])
        is_flagged = bool(fraud_prob >= threshold)
        actual = "FRAUD" if sample[TARGET] == 1 else "LEGITIMATE"
        status = "[FLAGGED - FRAUD ALERT]" if is_flagged else "[APPROVED]"
        correct = (is_flagged and actual == "FRAUD") or (not is_flagged and actual == "LEGITIMATE")

        print(f"\nTransaction Type: {label}")
        print(f"  Time (sec):          {sample['Time']:.0f}")
        print(f"  Amount:              ${sample['Amount']:.2f}")
        print(f"  Ground Truth:        {actual}")
        print(f"  Predicted Score:     {fraud_prob:.4f} ({fraud_prob * 100:.2f}%)")
        print(f"  System Action:       {status}")
        print(f"  Decision Accuracy:   {'CORRECT DECISION [OK]' if correct else 'MISCLASSIFIED'}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    run_demo()
