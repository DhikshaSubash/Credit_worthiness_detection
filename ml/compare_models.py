"""
Model Comparison Script
Benchmarking Random Forest vs Logistic Regression (Industry Standard)

"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score

# Add parent directory to path to import data_prep
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data_prep import prepare_data_for_training

def compare_models():
    print("\n" + "="*60)
    print("STARTING MODEL COMPARISON BENCHMARK")
    print("="*60)

    # 1. Get Data
    print("[1/4] Fetching and preparing data...")
    X, y, feature_names = prepare_data_for_training()
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Define Models
    # We compare against Logistic Regression because it is the banking baseline.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    }

    # 4. Train and Evaluate
    print(f"\n[2/4] Training {len(models)} models on {len(X_train)} samples...")
    results = []

    for name, model in models.items():
        print(f"  > Training {name}...")
        try:
            model.fit(X_train, y_train)
            
            # Predictions
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_test, preds)
            roc = roc_auc_score(y_test, probs)
            rec = recall_score(y_test, preds)
            prec = precision_score(y_test, preds)
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "ROC-AUC": roc,
                "Recall (Risk)": rec,
                "Precision": prec
            })
        except Exception as e:
            print(f"    ! Error training {name}: {str(e)}")

    # 5. Display Results Table
    results_df = pd.DataFrame(results).set_index("Model")
    print("\n" + "="*60)
    print("FINAL BENCHMARK RESULTS")
    print("="*60)
    print(results_df)
    print("\n" + "="*60)
    

if __name__ == "__main__":
    compare_models()
