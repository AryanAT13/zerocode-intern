
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

question = "How do I start investing with $1000?"

inputs = tokenizer(question, return_tensors="pt").input_ids
outputs = model.generate(inputs, max_length=50)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Base model answer:", answer)
