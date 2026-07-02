import pandas as pd

df = pd.read_csv("data/Mental_Health_FAQ.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head())


df = df.rename(columns={"Questions": "question", "Answers": "answer"})
df["intent"] = df["Question_ID"]
df.to_csv("data/faq_data.csv", index=False)
print("Saved data/faq_data.csv")