# Mental Health FAQ Chatbot

A conversational FAQ agent benchmarking rule-based, retrieval-based, and generative NLP architectures on a mental health FAQ dataset.

## Overview

This project compares 7 approaches to answering mental health FAQ questions:
- **Rule-based**: TF-IDF + cosine similarity
- **Retrieval-based**: RNN, LSTM, BiLSTM, GRU, CNN (intent classifiers)
- **Generative**: Seq2Seq LSTM (encoder-decoder)

## Results

| Model | Top-1 Accuracy | BLEU-4 | ROUGE-L |
|---|---|---|---|
| CNN | 99.5% | 0.999 | 0.999 |
| BiLSTM | 98.9% | 0.997 | 0.999 |
| LSTM | 98.9% | 0.997 | 0.999 |
| GRU | 98.4% | 0.992 | 0.994 |
| TF-IDF | 94.2% | 0.959 | 0.965 |
| RNN | 91.1% | 0.920 | 0.929 |
| Seq2Seq LSTM | — | 0.008 | 0.123 |

CNN achieved the best top-1 retrieval accuracy. The generative Seq2Seq model, trained on a smaller filtered subset (161 pairs, answers ≤100 words), scored far lower on BLEU-4/ROUGE-L — an expected finding, since open-ended generation needs substantially more data than retrieval-based classification to produce coherent, on-topic text.

## Dataset

Base data: [Mental Health FAQ for Chatbot](https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot) (98 Q&A pairs), expanded to 583 rows via template-based and WordNet-based paraphrase augmentation. Clinical/diagnostic terms are excluded from synonym substitution to prevent semantic drift.

**Scope limitation**: the assistant answers questions within its 98-topic FAQ coverage. Questions outside this scope correctly trigger a low-confidence fallback rather than a fabricated answer.

## Safety

The deployed app includes a confidence-based fallback: low-confidence predictions redirect users to professional mental health resources and crisis lines instead of guessing. Long-form answers containing crisis-specific information are excluded from generative model training and served only via retrieval, to preserve exact wording.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live demo

[https://mental-health-faq-agent.streamlit.app/]

## Disclaimer

This tool provides general information only and is not a substitute for professional diagnosis, treatment, or crisis support.