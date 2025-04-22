import time

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# Загрузка дообученной модели
model_name = 'project-nn' 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_solution(question, max_length=512, temperature=0.7):
    prompt = f"{question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    solution = tokenizer.decode(output[0], skip_special_tokens=True)
    print(solution)
    print()

    return solution.replace(prompt, "").strip()

start = time.time()

print(generate_solution(input('Введите вопрос: ')))

print('Generating time:', time.time()-start)
