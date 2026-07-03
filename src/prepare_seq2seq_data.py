import pandas as pd
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("data/faq_data_augmented.csv")
df = df.dropna(subset=["question", "answer"])

# Cap extremely long answers — keeps training tractable and avoids the decoder
# being dominated by padding tokens
MAX_ANSWER_WORDS = 100
df = df[df["answer"].str.split().apply(len) <= MAX_ANSWER_WORDS]
print(f"Rows remaining after length cap: {len(df)}")

# Deduplicate answers so we don't train on the same answer 6x per intent (one per unique answer is enough signal)
# But keep all question variants, since more question-side variety helps the encoder
questions = df["question"].tolist()
answers = ["<start> " + a + " <end>" for a in df["answer"].tolist()]

# Tokenizer for questions (encoder side)
q_tokenizer = Tokenizer(oov_token="<OOV>")
q_tokenizer.fit_on_texts(questions)
q_vocab_size = len(q_tokenizer.word_index) + 1
max_len_q = max(len(q.split()) for q in questions)

# Tokenizer for answers (decoder side) — filters must NOT strip < > or our start/end tokens get destroyed
a_tokenizer = Tokenizer(oov_token="<OOV>", filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
a_tokenizer.fit_on_texts(answers)
a_vocab_size = len(a_tokenizer.word_index) + 1
max_len_a = max(len(a.split()) for a in answers)

print(f"Question vocab: {q_vocab_size}, max len: {max_len_q}")
print(f"Answer vocab: {a_vocab_size}, max len: {max_len_a}")

encoder_input = pad_sequences(q_tokenizer.texts_to_sequences(questions), maxlen=max_len_q, padding="post")
decoder_full = pad_sequences(a_tokenizer.texts_to_sequences(answers), maxlen=max_len_a, padding="post")

# Teacher forcing: decoder input = answer without last token, decoder target = answer without first token
decoder_input = decoder_full[:, :-1]
decoder_target = decoder_full[:, 1:]

joblib.dump({
    "encoder_input": encoder_input,
    "decoder_input": decoder_input,
    "decoder_target": decoder_target,
    "q_vocab_size": q_vocab_size,
    "a_vocab_size": a_vocab_size,
    "max_len_q": max_len_q,
    "max_len_a": max_len_a,
}, "saved_models/seq2seq_data.pkl")

joblib.dump(q_tokenizer, "saved_models/q_tokenizer.pkl")
joblib.dump(a_tokenizer, "saved_models/a_tokenizer.pkl")

print(f"Saved {encoder_input.shape[0]} training pairs")