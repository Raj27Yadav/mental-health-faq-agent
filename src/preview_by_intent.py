import pandas as pd

df = pd.read_csv('data/faq_data_augmented.csv')
sample_intents = df['intent'].drop_duplicates().sample(10, random_state=1)

for intent in sample_intents:
    print(f"\n--- intent: {intent} ---")
    print(df[df['intent'] == intent]['question'].to_string(index=False))