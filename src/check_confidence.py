import joblib
import numpy as np
from tensorflow.keras.models import load_model

data = joblib.load("saved_models/preprocessed_data.pkl")
X_test, y_test = data["X_test"], data["y_test"]

model = load_model("saved_models/CNN.keras")
probs = model.predict(X_test, verbose=0)
confidences = probs.max(axis=1)
preds = probs.argmax(axis=1)

correct_conf = confidences[preds == y_test]
wrong_conf = confidences[preds != y_test]

print(f"Correct predictions ({len(correct_conf)}): min={correct_conf.min():.3f} mean={correct_conf.mean():.3f}")
if len(wrong_conf) > 0:
    print(f"Wrong predictions ({len(wrong_conf)}): min={wrong_conf.min():.3f} max={wrong_conf.max():.3f} mean={wrong_conf.mean():.3f}")
else:
    print("No wrong predictions in test set")

print(f"\nAll confidence scores sorted (lowest 15):")
print(np.sort(confidences)[:15])