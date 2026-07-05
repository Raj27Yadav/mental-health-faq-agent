from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('Vamsi/T5_Paraphrase_Paws')
model = T5ForConditionalGeneration.from_pretrained('Vamsi/T5_Paraphrase_Paws')

text = "paraphrase: What is anxiety? </s>"
enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
out = model.generate(
    input_ids=enc["input_ids"],
    attention_mask=enc["attention_mask"],
    max_length=64,
    do_sample=True,
    top_k=120,
    top_p=0.95,
    temperature=1.5,
    num_return_sequences=4,
    repetition_penalty=1.3,
)
print([tokenizer.decode(o, skip_special_tokens=True) for o in out])