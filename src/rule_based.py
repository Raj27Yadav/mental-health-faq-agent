import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

df = pd.read_csv("data/faq_data_augmented.csv")

# Rebuild the same train/test split logic used in preprocess.py for consistency
data = joblib.load("saved_models/preprocessed_data.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")
intent_to_answer = joblib.load("saved_models/intent_to_answer.pkl")

# Recreate train/test question text (same row order as preprocess.py used)
from sklearn.model_selection import train_test_split as tts

train_parts, test_parts = [], []
for intent, group in df.groupby("intent"):
    if len(group) >= 5:
        tr, te = tts(group, test_size=0.2, random_state=42)
    else:
        tr, te = group, group.iloc[0:0]
    train_parts.append(tr)
    test_parts.append(te)

train_df = pd.concat(train_parts).reset_index(drop=True)
test_df = pd.concat(test_parts).reset_index(drop=True)

# TF-IDF fit on training questions only
vectorizer = TfidfVectorizer()
faq_matrix = vectorizer.fit_transform(train_df["question"])

def rule_based_response(query):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, faq_matrix)[0]
    idx = sims.argmax()
    return train_df.iloc[idx]["answer"], train_df.iloc[idx]["intent"], sims[idx]

smoothie = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

correct = 0
bleu_scores, rouge_scores = [], []

for _, row in test_df.iterrows():
    pred_answer, pred_intent, confidence = rule_based_response(row["question"])
    true_answer = row["answer"]

    if pred_intent == row["intent"]:
        correct += 1

    bleu = sentence_bleu([true_answer.split()], pred_answer.split(), smoothing_function=smoothie)
    rl = rouge.score(true_answer, pred_answer)['rougeL'].fmeasure
    bleu_scores.append(bleu)
    rouge_scores.append(rl)

top1_acc = correct / len(test_df)
print(f"Rule-based (TF-IDF): acc={top1_acc:.4f}  BLEU-4={np.mean(bleu_scores):.4f}  ROUGE-L={np.mean(rouge_scores):.4f}")

joblib.dump(vectorizer, "saved_models/tfidf_vectorizer.pkl")
joblib.dump(train_df, "saved_models/tfidf_train_df.pkl")

result = {"model": "TF-IDF", "top1_accuracy": top1_acc, "bleu4": np.mean(bleu_scores), "rougeL": np.mean(rouge_scores)}

existing = pd.read_csv("saved_models/retrieval_evaluation.csv")
combined = pd.concat([existing, pd.DataFrame([result])], ignore_index=True).sort_values("top1_accuracy", ascending=False)
combined.to_csv("saved_models/all_models_evaluation.csv", index=False)
print("\n", combined.to_string(index=False))