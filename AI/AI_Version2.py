import pandas as pd
import json
with open('intents.json','r') as f:
  data=json.load(f)

texts=[]
tags=[]
for intent in data['intents']:
  for pattern in intent['patterns']:
    texts.append(pattern)
    tags.append(intent['tag'])

df=pd.DataFrame({'text':texts,'tag':tags})
unique_tags=sorted(list(df['tag'].unique()))
tag2id={tag: i for i, tag in enumerate(unique_tags)}
id2tag={i: tag for i, tag in enumerate(unique_tags)}
df['label']=df['tag'].map(tag2id)

model_save_path = "./my_nlu_model"
import os
os.makedirs(model_save_path, exist_ok=True) 
label_mappings = {'id2tag': id2tag, 'tag2id': tag2id}
with open(f"{model_save_path}/label_mappings.json", 'w') as f:
    json.dump(label_mappings, f)   
print("Label mappings saved.")

num_labels = len(unique_tags)

from datasets import Dataset
from transformers import AutoTokenizer

dataset=Dataset.from_pandas(df)
model_name='distilbert-base-uncased'
tokenizer=AutoTokenizer.from_pretrained(model_name)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets=dataset.map(tokenize_function,batched=True)
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score
model=AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2tag,
    label2id=tag2id
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=30,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
)

def compute_metrics(eval_pred):
  logits,labels=eval_pred
  predictions=np.argmax(logits,axis=-1)
  return {"accuracy":accuracy_score(labels,predictions)}

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model saved to {model_save_path}")

from transformers import pipeline
classifier=pipeline("text-classification", model="./my_nlu_model")
user_input_1 = "I feel so anxious and worried all the time"
result_1 = classifier(user_input_1)
print(f"Input: '{user_input_1}'")
print(f"Predicted Intent: {result_1[0]['label']} (Score: {result_1[0]['score']:.4f})")

print("-" * 20)

user_input_2 = "hi there, can we talk?"
result_2 = classifier(user_input_2)
print(f"Input: '{user_input_2}'")
print(f"Predicted Intent: {result_2[0]['label']} (Score: {result_2[0]['score']:.4f})")