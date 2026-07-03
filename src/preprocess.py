import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("data/faq_data_augmented.csv")
df = df.dropna(subset=["question", "answer"])

# Drop intents with fewer than 2 examples — can't be stratified
intent_counts = df["intent"].value_counts()
valid_intents = intent_counts[intent_counts >= 2].index
dropped = intent_counts[intent_counts < 2]
print(f"Dropping {len(dropped)} intents with only 1 example (no successful paraphrase)")

df = df[df["intent"].isin(valid_intents)]
print(f"Remaining: {len(df)} rows across {df['intent'].nunique()} intents")

# Stratified 80/20 split
from sklearn.model_selection import train_test_split as tts

train_parts, test_parts = [], []
for intent, group in df.groupby("intent"):
    if len(group) >= 5:
        # Enough examples: do a real 80/20 split for this intent
        tr, te = tts(group, test_size=0.2, random_state=42)
    else:
        # Too few examples to hold any out — keep all in training
        tr, te = group, group.iloc[0:0]
    train_parts.append(tr)
    test_parts.append(te)

train_df = pd.concat(train_parts).reset_index(drop=True)
test_df = pd.concat(test_parts).reset_index(drop=True)

print(f"Intents with test coverage: {test_df['intent'].nunique()} / {df['intent'].nunique()}")

print(f"Train: {len(train_df)}  Test: {len(test_df)}")

# Tokenizer fit ONLY on training questions
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["question"])
vocab_size = len(tokenizer.word_index) + 1
max_len = max(len(q.split()) for q in train_df["question"])
print(f"Vocab size: {vocab_size}  Max length: {max_len}")

X_train = pad_sequences(tokenizer.texts_to_sequences(train_df["question"]), maxlen=max_len, padding="post")
X_test  = pad_sequences(tokenizer.texts_to_sequences(test_df["question"]),  maxlen=max_len, padding="post")

# Encode intent labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["intent"])
y_test  = label_encoder.transform(test_df["intent"])
num_classes = len(label_encoder.classes_)
print(f"Num intents (classes): {num_classes}")

# Save everything needed later
joblib.dump(tokenizer, "saved_models/tokenizer.pkl")
joblib.dump(label_encoder, "saved_models/label_encoder.pkl")
joblib.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
             "max_len": max_len, "vocab_size": vocab_size, "num_classes": num_classes},
            "saved_models/preprocessed_data.pkl")

# Save intent -> answer lookup for inference later
intent_to_answer = df.drop_duplicates(subset="intent").set_index("intent")["answer"].to_dict()
joblib.dump(intent_to_answer, "saved_models/intent_to_answer.pkl")

print("Saved tokenizer, label encoder, preprocessed data, and answer lookup to saved_models/")