import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Mental Health FAQ Assistant", page_icon="💬")

@st.cache_resource
def load_assets():
    tokenizer = joblib.load("saved_models/tokenizer.pkl")
    label_encoder = joblib.load("saved_models/label_encoder.pkl")
    model = load_model("saved_models/CNN.keras")
    intent_to_answer = joblib.load("saved_models/intent_to_answer.pkl")
    data = joblib.load("saved_models/preprocessed_data.pkl")
    return tokenizer, label_encoder, model, intent_to_answer, data["max_len"]

tokenizer, label_encoder, model, intent_to_answer, max_len = load_assets()

CONFIDENCE_THRESHOLD = 0.55

st.title("💬 Mental Health FAQ Assistant")
st.caption(
    "This assistant provides general information only and is not a substitute "
    "for professional diagnosis, treatment, or crisis support."
)
st.info("💡 Try asking specific questions like: *\"What's the difference between anxiety and stress?\"* or *\"Where can I find a support group?\"*")

with st.expander("⚠️ If you're in crisis, please read this"):
    st.write(
        "If you or someone you know is in immediate danger, please call your local "
        "emergency number. In the US, you can call or text **988** (Suicide & Crisis Lifeline) "
        "any time, day or night."
    )

query = st.text_input("Ask a question about mental health:")

if query:
    seq = pad_sequences(tokenizer.texts_to_sequences([query.lower()]), maxlen=max_len)
    probs = model.predict(seq, verbose=0)[0]
    confidence = float(probs.max())
    intent = label_encoder.inverse_transform([probs.argmax()])[0]

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            "I'm not confident I understood your question correctly. "
            "For accurate guidance, please consider speaking with a licensed mental "
            "health professional, or contact a crisis line if this is urgent "
            "(in the US: call or text 988)."
        )
    else:
        st.write(intent_to_answer[intent])
        st.caption(f"Confidence: {confidence:.0%}")
