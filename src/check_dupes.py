import pandas as pd

df = pd.read_csv('data/faq_data_augmented.csv')
df['q_norm'] = df['question'].str.lower().str.strip()
dupes = df[df.duplicated(subset='q_norm', keep=False)].sort_values('q_norm')
print(f"Total rows: {len(df)}")
print(f"Duplicate rows (normalized): {df['q_norm'].duplicated().sum()}")
print(dupes[['question', 'intent']].to_string())