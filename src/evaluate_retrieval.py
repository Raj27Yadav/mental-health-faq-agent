import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

data = joblib.load("saved_models/preprocessed_data.pkl")
X_test, y_test = data["X_test"], data["y_test"]

label_encoder = joblib.load("saved_models/label_encoder.pkl")
intent_to_answer = joblib.load("saved_models/intent_to_answer.pkl")

df = pd.read_csv("data/faq_data_augmented.csv")
true_answers = []
for intent_id in label_encoder.inverse_transform(y_test):
    true_answers.append(intent_to_answer[intent_id])

smoothie = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

architectures = ["RNN", "LSTM", "BiLSTM", "GRU", "CNN"]
summary = []

for arch in architectures:
    model = load_model(f"saved_models/{arch}.keras")
    preds = model.predict(X_test, verbose=0)
    pred_intents = label_encoder.inverse_transform(preds.argmax(axis=1))
    pred_answers = [intent_to_answer[i] for i in pred_intents]

    bleu_scores, rouge_scores = [], []
    for true_ans, pred_ans in zip(true_answers, pred_answers):
        bleu = sentence_bleu([true_ans.split()], pred_ans.split(), smoothing_function=smoothie)
        rl = rouge.score(true_ans, pred_ans)['rougeL'].fmeasure
        bleu_scores.append(bleu)
        rouge_scores.append(rl)

    top1_acc = (pred_intents == label_encoder.inverse_transform(y_test)).mean()
    summary.append({
        "model": arch,
        "top1_accuracy": top1_acc,
        "bleu4": np.mean(bleu_scores),
        "rougeL": np.mean(rouge_scores)
    })
    print(f"{arch}: acc={top1_acc:.4f}  BLEU-4={np.mean(bleu_scores):.4f}  ROUGE-L={np.mean(rouge_scores):.4f}")

results_df = pd.DataFrame(summary).sort_values("top1_accuracy", ascending=False)
results_df.to_csv("saved_models/retrieval_evaluation.csv", index=False)
print("\n", results_df.to_string(index=False))