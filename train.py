from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import json
import torch
import os


torch.cuda.empty_cache()


def prepare_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        try:
            if item['type'] == 'theory':
                prompt = f"Теория: {item['content']}"
                completion = ""  # Можно оставить пустым или повторить теорию
        except Exception:
            prompt = f"Вопрос: {item['question']}\nРешение:"
            completion = f" {item['solution']}"
        formatted_data.append({"prompt": prompt, "completion": completion})
    
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in formatted_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')


input_json = 'data.json'
output_jsonl = 'formatted_tasks.jsonl'


prepare_data(input_json, output_jsonl)
dataset = load_dataset('json', data_files={'train': output_jsonl})


model_name = "meta-llama/Llama-3.1-8B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '<EOS>'})
    model.resize_token_embeddings(len(tokenizer))


def tokenize_function(batch):
    prompts = batch["prompt"]
    completions = batch["completion"]
    
    full_texts = [p + c for p, c in zip(prompts, completions)]
    
    tokenized = tokenizer(
        full_texts,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors=None
    )
    
    prompt_encodings = tokenizer(
        prompts,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors=None
    )
    
    labels = []
    for i in range(len(prompts)):
        prompt_length = 0
        for token_id in prompt_encodings['input_ids'][i]:
            if token_id == tokenizer.pad_token_id:
                break
            prompt_length += 1
        
        label = tokenized["input_ids"][i].copy()
        
        if prompt_length > 0:
            label[:prompt_length] = [-100] * prompt_length
        
        labels.append(label)
    
    tokenized["labels"] = labels
    return tokenized


tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['prompt', 'completion'])


split = tokenized_datasets['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
eval_dataset = split['test']

print(f"Размер обучающего набора: {len(train_dataset)}")
print(f"Размер валидационного набора: {len(eval_dataset)}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  
)


training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=10,                   
    per_device_train_batch_size=2,        
    per_device_eval_batch_size=2,         
    gradient_accumulation_steps=8,        
    eval_strategy="epoch",                
    save_strategy="epoch",                
    logging_dir='./logs',
    logging_steps=100,                    
    learning_rate=3e-5,                   
    weight_decay=0.0,                     
    fp16=False,                           
    save_total_limit=1,                   
    seed=42,                              
    evaluation_strategy="epoch",          
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,  
    data_collator=data_collator,
)


trainer.train()


output_dir = "project-nn"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Дообученная модель сохранена в директории: {output_dir}")
