import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

q_tokenizer = joblib.load("saved_models/q_tokenizer.pkl")
a_tokenizer = joblib.load("saved_models/a_tokenizer.pkl")
data = joblib.load("saved_models/seq2seq_data.pkl")
max_len_q = data["max_len_q"]
max_len_a_decoder = data["decoder_input"].shape[1]  # decoder input length (answer max len - 1)

encoder_model = load_model("saved_models/seq2seq_encoder.keras")
decoder_model = load_model("saved_models/seq2seq_decoder.keras")

# Reverse lookup: index -> word
a_index_word = {v: k for k, v in a_tokenizer.word_index.items()}
start_token_id = a_tokenizer.word_index["<start>"]
end_token_id = a_tokenizer.word_index["<end>"]

def generate_answer(question, max_output_len=60):
    seq = q_tokenizer.texts_to_sequences([question.lower()])
    seq = pad_sequences(seq, maxlen=max_len_q, padding="post")

    states_value = encoder_model.predict(seq, verbose=0)

    target_seq = np.zeros((1, max_len_a_decoder))
    target_seq[0, 0] = start_token_id

    decoded_words = []
    for i in range(max_output_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, i, :]) if i < output_tokens.shape[1] else np.argmax(output_tokens[0, -1, :])
        sampled_word = a_index_word.get(sampled_token_index, "")

        if sampled_word == "<end>" or sampled_word == "":
            break
        decoded_words.append(sampled_word)

        if i + 1 < max_len_a_decoder:
            target_seq[0, i + 1] = sampled_token_index
        states_value = [h, c]

    return " ".join(decoded_words)

if __name__ == "__main__":
    test_questions = [
        "What is anxiety?",
        "How can I find a therapist?",
        "What is depression?",
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {generate_answer(q)}")