import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = joblib.load("saved_models/tokenizer.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")
model = load_model("saved_models/CNN.keras")
data = joblib.load("saved_models/preprocessed_data.pkl")
max_len = data["max_len"]

query = "What is anxiety?"
seq = tokenizer.texts_to_sequences([query.lower()])
print("Tokenized sequence:", seq)
padded = pad_sequences(seq, maxlen=max_len)
probs = model.predict(padded, verbose=0)[0]
print("Max confidence:", probs.max())
print("Predicted intent:", label_encoder.inverse_transform([probs.argmax()])[0])