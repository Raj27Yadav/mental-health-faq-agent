import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.random.set_seed(42)

data = joblib.load("saved_models/seq2seq_data.pkl")
encoder_input = data["encoder_input"]
decoder_input = data["decoder_input"]
decoder_target = data["decoder_target"]
q_vocab_size = data["q_vocab_size"]
a_vocab_size = data["a_vocab_size"]
max_len_q = data["max_len_q"]
max_len_a = data["max_len_a"]

print(f"Training pairs: {encoder_input.shape[0]}")
print(f"Q vocab: {q_vocab_size}, A vocab: {a_vocab_size}")
print(f"Max Q len: {max_len_q}, Max A len (decoder): {decoder_input.shape[1]}")

EMBED_DIM = 128
LATENT_DIM = 128

# ---------- Training model ----------
encoder_inputs = Input(shape=(max_len_q,), name="encoder_input")
enc_emb_layer = Embedding(q_vocab_size, EMBED_DIM, mask_zero=True, name="encoder_embedding")
enc_emb = enc_emb_layer(encoder_inputs)
encoder_lstm = LSTM(LATENT_DIM, return_state=True, name="encoder_lstm")
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(decoder_input.shape[1],), name="decoder_input")
dec_emb_layer = Embedding(a_vocab_size, EMBED_DIM, mask_zero=True, name="decoder_embedding")
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(a_vocab_size, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

decoder_target_expanded = np.expand_dims(decoder_target, -1)

early_stop = EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=6, min_lr=1e-5)

history = model.fit(
    [encoder_input, decoder_input],
    decoder_target_expanded,
    batch_size=8,
    epochs=200,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

model.save("saved_models/seq2seq_training_model.keras")
print("Saved training model")

# ---------- Build separate INFERENCE models (encoder + decoder) ----------
# Encoder inference model: question -> final states
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.save("saved_models/seq2seq_encoder.keras")

# Decoder inference model: one token at a time, states passed in and out
decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb_inf = dec_emb_layer(decoder_inputs)
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(dec_emb_inf, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)
decoder_model.save("saved_models/seq2seq_decoder.keras")

print("Saved encoder and decoder inference models")