import pandas as pd
import nltk
import random
import re
from nltk.corpus import wordnet

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---------- Method 1: Question templates ----------
# Rewrite common question openers into equivalent phrasings
TEMPLATE_RULES = [
    (r'^what is (.+)\?$',            lambda m: f"Can you explain what {m.group(1)} is?"),
    (r'^what is (.+)\?$',            lambda m: f"How would you define {m.group(1)}?"),
    (r'^what is (.+)\?$',            lambda m: f"Tell me about {m.group(1)}."),
    (r'^what are (.+)\?$',           lambda m: f"Can you explain what {m.group(1)} are?"),
    (r'^what are (.+)\?$',           lambda m: f"How would you describe {m.group(1)}?"),
    (r'^how (do|can) i (.+)\?$',     lambda m: f"What's the best way to {m.group(2)}?"),
    (r'^how (do|can) i (.+)\?$',     lambda m: f"What steps should I take to {m.group(2)}?"),
    (r'^why (do|does) (.+)\?$',      lambda m: f"What is the reason {m.group(2)}?"),
    (r'^can (.+)\?$',                lambda m: f"Is it possible {m.group(1)}?"),
    (r'^is (.+)\?$',                 lambda m: f"Would you say {m.group(1)}?"),
    (r'^when (should|do) i (.+)\?$', lambda m: f"At what point should I {m.group(2)}?"),
]

def template_paraphrase(question):
    q = question.strip().lower()
    results = []
    for pattern, transform in TEMPLATE_RULES:
        match = re.match(pattern, q)
        if match:
            try:
                results.append(transform(match))
            except Exception:
                pass
    return results

# ---------- Method 2: WordNet synonym replacement ----------
import nltk
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
from nltk import pos_tag

# Map Penn Treebank tags to WordNet POS
def get_wordnet_pos(tag):
    if tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    return None  # skip adverbs/other — too risky

# Words we never want to touch, even if WordNet has entries for them
BLOCK_WORDS = {'help', 'get', 'find', 'kids', 'drugs', 'health', 'seek', 'need',
               'brain', 'learn', 'evidence', 'cure', 'cures', 'worried', 'worry'}

CLINICAL_TERMS = {
    'anxiety', 'depression', 'schizoid', 'schizophrenia', 'schizophrenic',
    'bipolar', 'disorder', 'disorders', 'ptsd', 'ocd', 'adhd', 'autism',
    'personality', 'psychosis', 'psychotic', 'trauma', 'suicidal', 'suicide',
    'therapy', 'therapist', 'medication', 'diagnosis', 'symptom', 'symptoms',
    'counseling', 'counselling', 'eating', 'binge', 'addiction', 'substance'
}
def get_synonym(word, pos):
    if word.lower() in BLOCK_WORDS:
        return word
    synsets = wordnet.synsets(word, pos=pos)
    if not synsets:
        return word
    lemmas = [l.name().replace('_', ' ') for l in synsets[0].lemmas()]
    lemmas = [l for l in lemmas if l.lower() != word.lower() and l.isalpha()]
    return lemmas[0] if lemmas else word

def synonym_paraphrase(sentence, n_variants=1, n_replacements=1):
    words = sentence.rstrip('?').split()

    if any(w.strip('.,?').lower() in CLINICAL_TERMS for w in words):
        return []

    tagged = pos_tag(words)
    variants = []
    candidates = [
        i for i, (w, tag) in enumerate(tagged)
        if w.isalpha() and len(w) > 4 and get_wordnet_pos(tag) is not None
        and w.lower() not in BLOCK_WORDS
    ]
    for _ in range(n_variants):
        if not candidates:
            break
        idx = random.choice(candidates)
        word, tag = tagged[idx]
        wn_pos = get_wordnet_pos(tag)
        new_word = get_synonym(word, wn_pos)
        if new_word == word:
            continue
        new_words = words.copy()
        new_words[idx] = new_word
        variants.append(" ".join(new_words) + "?")
    return list(set(variants))

# ---------- Combine ----------
def generate_paraphrases(question):
    out = set()
    out.update(template_paraphrase(question))
    out.update(synonym_paraphrase(question, n_variants=2))
    return [p for p in out if p.strip().lower() != question.strip().lower()]

# ---------- Run over the dataset ----------
df = pd.read_csv("data/faq_data.csv")
print(f"Loaded {len(df)} original FAQ rows")

augmented_rows = []
for _, row in df.iterrows():
    augmented_rows.append({"question": row["question"], "answer": row["answer"], "intent": row["intent"]})
    for p in generate_paraphrases(row["question"]):
        augmented_rows.append({"question": p, "answer": row["answer"], "intent": row["intent"]})

aug_df = pd.DataFrame(augmented_rows).drop_duplicates(subset=["question"])
aug_df.to_csv("data/faq_data_augmented.csv", index=False)

print(f"Saved {len(aug_df)} rows to data/faq_data_augmented.csv")
print(aug_df["intent"].value_counts().describe())