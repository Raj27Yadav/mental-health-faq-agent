import pandas as pd

df = pd.read_csv("data/faq_data_augmented.csv")
df["answer_len"] = df["answer"].str.split().apply(len)

print(df["answer_len"].describe())
print("\nLongest answers:")
print(df.sort_values("answer_len", ascending=False)[["answer_len", "answer"]].head(5).to_string())