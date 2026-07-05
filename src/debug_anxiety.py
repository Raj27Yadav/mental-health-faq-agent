import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("data/faq_data_augmented.csv")
matches = df[df["question"].str.lower().str.contains("anxiety")]
print(matches[["question", "intent"]].to_string())

tokenizer = joblib.load("saved_models/tokenizer.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")
model = load_model("saved_models/CNN.keras")
data = joblib.load("saved_models/preprocessed_data.pkl")
max_len = data["max_len"]

print(f"\nmax_len used by model: {max_len}")

# Test each anxiety-related question directly from the dataset
for q in matches["question"].tolist():
    seq = tokenizer.texts_to_sequences([q.lower()])
    padded = pad_sequences(seq, maxlen=max_len)
    probs = model.predict(padded, verbose=0)[0]
    pred_intent = label_encoder.inverse_transform([probs.argmax()])[0]
    print(f"Q: {q!r} -> confidence={probs.max():.3f}, predicted intent={pred_intent}")