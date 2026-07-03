import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, SimpleRNN, LSTM, Bidirectional, GRU,
    Conv1D, GlobalMaxPooling1D, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.random.set_seed(42)

data = joblib.load("saved_models/preprocessed_data.pkl")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
vocab_size = data["vocab_size"]
max_len = data["max_len"]
num_classes = data["num_classes"]

print(f"Training on {X_train.shape[0]} examples, testing on {X_test.shape[0]}")
print(f"Vocab: {vocab_size}, Max len: {max_len}, Classes: {num_classes}")

def build_model(layer_type, embed_dim=64, units=64):
    inp = Input(shape=(max_len,))
    x = Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    if layer_type == "RNN":
        x = SimpleRNN(units)(x)
    elif layer_type == "LSTM":
        x = LSTM(units)(x)
    elif layer_type == "BiLSTM":
        x = Bidirectional(LSTM(units))(x)
    elif layer_type == "GRU":
        x = GRU(units)(x)
    elif layer_type == "CNN":
        x = Conv1D(units, 3, activation="relu")(x)
        x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

architectures = ["RNN", "LSTM", "BiLSTM", "GRU", "CNN"]
results = {}

for arch in architectures:
    print(f"\n{'='*40}\nTraining {arch}\n{'='*40}")
    model = build_model(arch)

    early_stop = EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=8, min_lr=1e-5)

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=8,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{arch} test accuracy: {test_acc:.4f}")
    results[arch] = test_acc
    model.save(f"saved_models/{arch}.keras")

print("\n\n=== Summary: Top-1 Accuracy ===")
for arch, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{arch}: {acc:.4f}")

joblib.dump(results, "saved_models/retrieval_results.pkl")