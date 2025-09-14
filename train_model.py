import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

MODEL_FILE = "gas_model.pkl"

def incremental_train(new_csv_files):
    """
    Incrementally trains or retrains a RandomForest model with new CSV files.
    """
    # Load model if exists
    if os.path.exists(MODEL_FILE):
        clf, le = joblib.load(MODEL_FILE)
        print("Loaded existing model.")
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        le = LabelEncoder()
        print("Created new model.")

    # Load new CSV files
    new_dfs = []
    for csv_file in new_csv_files:
        df = pd.read_csv(csv_file)
        df = df[~df['gas_index'].isin([99, 100])]  # ignore invalid indexes
        new_dfs.append(df)
    data = pd.concat(new_dfs, ignore_index=True)

    # Features and labels
    features = ['millis','gas_index','mes_index','temperature','pressure','humidity','gas_resistance']
    X = data[features]
    y = le.fit_transform(data['label'])

    # Train and test split for evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf.fit(X_train, y_train)
    print(f"Model trained with {len(data)} samples.")

    # Predictions
    y_pred = clf.predict(X_test)

    # Accuracy, precision, recall, f1
    print("\n=== Model Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Calculate false positives and false negatives
    FP = cm.sum(axis=0) - cm.diagonal()
    FN = cm.sum(axis=1) - cm.diagonal()
    TP = cm.diagonal()
    TN = cm.sum() - (FP + FN + TP)

    total = cm.sum()

    print("\n=== Error Analysis ===")
    for i, label in enumerate(le.classes_):
        fp_rate = (FP[i] / total) * 100 if total > 0 else 0
        fn_rate = (FN[i] / total) * 100 if total > 0 else 0
        print(f"Class '{label}': False Positives = {FP[i]} ({fp_rate:.2f}%), False Negatives = {FN[i]} ({fn_rate:.2f}%)")

    # Save model
    joblib.dump((clf, le), MODEL_FILE)
    print(f"\nUpdated model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    # Example: train with one dataset
    incremental_train(["ar_cigarro.csv"])
