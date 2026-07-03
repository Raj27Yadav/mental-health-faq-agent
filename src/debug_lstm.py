import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

data = joblib.load("saved_models/preprocessed_data.pkl")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
vocab_size, max_len, num_classes = data["vocab_size"], data["max_len"], data["num_classes"]

inp = Input(shape=(max_len,))
x = Embedding(vocab_size, 64, mask_zero=True)(inp)
x = LSTM(64)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(num_classes, activation="softmax")(x)
model = Model(inp, out)

model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=8, min_lr=1e-5)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=8,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print(history.history["loss"])
print(history.history["accuracy"])