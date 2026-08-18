import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score

DATA_PATH = "data/NSL_KDD.csv"
MODEL_DIR = "models"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "\nNSL-KDD dataset not found!\n"
        "Put your CSV file here:\n"
        "data/NSL_KDD.csv\n"
    )

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

target_candidates = ["label", "class", "attack"]
target = next((c for c in target_candidates if c in df.columns), None)

if target is None:
    raise ValueError(
        f"No target column found. Expected one of: {target_candidates}. "
        f"Available columns: {list(df.columns)}"
    )

df = df.dropna().reset_index(drop=True)
X = df.drop(columns=[target]).copy()
y = df[target].astype(str)

categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
encoders = {}

for col in categorical_columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col].astype(str))
    encoders[col] = encoder

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

feature_columns = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

classifier = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, pred)

anomaly_model = IsolationForest(
    n_estimators=150,
    contamination=0.10,
    random_state=42
)
anomaly_model.fit(X_scaled)

joblib.dump(classifier, f"{MODEL_DIR}/classifier.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(anomaly_model, f"{MODEL_DIR}/anomaly_model.pkl")
joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")
joblib.dump(target_encoder, f"{MODEL_DIR}/target_encoder.pkl")
joblib.dump(feature_columns, f"{MODEL_DIR}/feature_columns.pkl")

with open(f"{MODEL_DIR}/model_info.txt", "w", encoding="utf-8") as f:
    f.write(f"Rows used: {len(df)}\n")
    f.write(f"Features: {len(feature_columns)}\n")
    f.write(f"Target: {target}\n")
    f.write(f"Classification accuracy: {accuracy:.4f}\n")

print("=" * 55)
print("MODEL TRAINING COMPLETE")
print("=" * 55)
print(f"Dataset rows: {len(df)}")
print(f"Features: {len(feature_columns)}")
print(f"Target column: {target}")
print(f"Classification accuracy: {accuracy:.4f}")
print("Model files saved in: models/")
