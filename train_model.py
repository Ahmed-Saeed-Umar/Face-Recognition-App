
import os
import numpy as np
import pickle
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


IMG_SIZE = (64, 64)          # resize all images to this
DATASET_DIR = "dataset"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

#LOADING IMAGES

def load_images(folder, label):
    data, labels = [], []
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(supported)]

    if not files:
        print(f"⚠️  No images found in {folder}")
        return data, labels

    for filename in files:
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Skipping unreadable file: {filename}")
            continue
        img = cv2.resize(img, IMG_SIZE)
        data.append(img.flatten())
        labels.append(label)

    print(f"  Loaded {len(data)} images from '{folder}'")
    return data, labels


print("\n📂 Loading dataset...")
me_data, me_labels         = load_images(os.path.join(DATASET_DIR, "ME"),     label=1)
not_me_data, not_me_labels = load_images(os.path.join(DATASET_DIR, "NOT_ME"), label=0)

if not me_data or not not_me_data:
    raise SystemExit(
        "\n❌ Dataset is empty!\n"
        "Add images to dataset/ME/ and dataset/NOT_ME/ then re-run."
    )

X = np.array(me_data + not_me_data, dtype=np.float32)
y = np.array(me_labels + not_me_labels)
print(f"  Total samples: {len(X)}  (ME={len(me_data)}, NOT_ME={len(not_me_data)})")


# 2.SPLITTING THE DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


#4. TRAINING AND COMPARING MODELS
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    "Decision Tree":       DecisionTreeClassifier(max_depth=10),
    "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
}

print("\n🏋️  Training models...\n")
best_name, best_model, best_acc = None, None, 0.0

for name, clf in candidates.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"{'─'*50}")
    print(f"  Model : {name}")
    print(f"  Accuracy : {acc*100:.1f}%")
    print(classification_report(y_test, preds, target_names=["NOT ME", "ME"]))
    if acc > best_acc:
        best_acc, best_name, best_model = acc, name, clf

print(f"{'═'*50}")
print(f"🏆  Best model : {best_name}  ({best_acc*100:.1f}% accuracy)")

# Confusion matrix for best model
cm = confusion_matrix(y_test, best_model.predict(X_test))
print(f"\n  Confusion Matrix:\n{cm}")
print(f"  (rows = actual, cols = predicted | 0=NOT ME, 1=ME)\n")


# 5. Save best model + scaler

with open(os.path.join(MODEL_DIR, "sklearn_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(f"✅  Model saved  → model/sklearn_model.pkl")
print(f"✅  Scaler saved → model/scaler.pkl")
print("\n🚀  Now run:  python app.py\n")
