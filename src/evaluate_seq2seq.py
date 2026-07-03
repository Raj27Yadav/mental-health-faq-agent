import joblib
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
from generate_answer import generate_answer

df = pd.read_csv("data/faq_data_augmented.csv")
df = df.dropna(subset=["question", "answer"])
df = df[df["answer"].str.split().apply(len) <= 100]  # same cap used in training

# Same-style split as before, just a plain 80/20 here since this is a smaller, non-stratified set
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

smoothie = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

bleu_scores, rouge_scores = [], []
examples = []

for _, row in test_df.iterrows():
    true_answer = row["answer"]
    pred_answer = generate_answer(row["question"])

    bleu = sentence_bleu([true_answer.split()], pred_answer.split(), smoothing_function=smoothie)
    rl = rouge.score(true_answer, pred_answer)['rougeL'].fmeasure
    bleu_scores.append(bleu)
    rouge_scores.append(rl)
    examples.append((row["question"], pred_answer[:80]))

print(f"Seq2Seq: BLEU-4={np.mean(bleu_scores):.4f}  ROUGE-L={np.mean(rouge_scores):.4f}")
print(f"\nSample generations:")
for q, a in examples[:5]:
    print(f"Q: {q}\nA: {a}\n")

result = {"model": "Seq2Seq-LSTM", "top1_accuracy": np.nan, "bleu4": np.mean(bleu_scores), "rougeL": np.mean(rouge_scores)}

existing = pd.read_csv("saved_models/all_models_evaluation.csv")
combined = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)
combined.to_csv("saved_models/all_models_evaluation.csv", index=False)
print("\n", combined.to_string(index=False))