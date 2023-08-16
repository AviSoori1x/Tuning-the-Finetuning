# Databricks notebook source
# MAGIC %pip install transformers==4.31.0 datasets==2.13.0 peft==0.4.0 accelerate==0.21.0 bitsandbytes==0.40.2 trl==0.4.7

# COMMAND ----------

from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from transformers.trainer_callback import TrainerCallback
import os
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
import mlflow

# COMMAND ----------

# MAGIC %sql
# MAGIC USE description_generator;

# COMMAND ----------

df = spark.sql("SELECT * FROM product_name_to_description").toPandas()
df['text'] = df["prompt"]+df["response"]
df.drop(columns=['prompt', 'response'], inplace=True)
display(df), df.shape

# COMMAND ----------

from datasets import load_dataset
from datasets import Dataset
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.05, seed=42)

# COMMAND ----------

target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
#or
# target_modules = ['q_proj','v_proj']

lora_config = LoraConfig(
    r=8,#or r=16
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    target_modules = target_modules,
    task_type="CAUSAL_LM",
)

base_dir = "<base_dir_location>"

per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = 'adamw_hf'
learning_rate = 1e-5
max_grad_norm = 0.3
warmup_ratio = 0.03
lr_scheduler_type = "linear"

# COMMAND ----------

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir=base_dir,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs = 3.0,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)
    

# COMMAND ----------

nf4_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16
)

# COMMAND ----------

model_path = 'openlm-research/open_llama_3b_v2'

# COMMAND ----------

tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# COMMAND ----------

model = LlamaForCausalLM.from_pretrained(
    model_path, device_map='auto', quantization_config=nf4_config,
)

# COMMAND ----------

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# COMMAND ----------

trainer = SFTTrainer(
    model,
    train_dataset=dataset['train'],
    eval_dataset = dataset['test'],
    dataset_text_field="text",
    max_seq_length=256,
    args=training_args,
)
#Upcast layer norms to float 32 for stability
for name, module in trainer.model.named_modules():
  if "norm" in name:
    module = module.to(torch.float32)

# COMMAND ----------
# Initiate the training process
with mlflow.start_run(run_name=’run_name_of_choice’):
    trainer.train()

# COMMAND ----------

# #https://github.com/NVIDIA/apex/issues/965
# for param in model.parameters():
#     # Check if parameter dtype is  Half (float16)
#     if param.dtype == torch.float16:
#         param.data = param.data.to(torch.float32)

# COMMAND ----------

# MAGIC %md
# MAGIC ### If loading from saved adapter

# COMMAND ----------

dbutils.fs.ls('<base_dir_location>')

# COMMAND ----------

model_path = 'openlm-research/open_llama_3b_v2'

# COMMAND ----------

tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# COMMAND ----------

model = LlamaForCausalLM.from_pretrained(
    model_path, load_in_8bit=True, device_map='auto',
)

# COMMAND ----------

peft_model_id = '<adapter_final_checkpoint_location>'

# COMMAND ----------

peft_model = PeftModel.from_pretrained(model, peft_model_id)

# COMMAND ----------

test_strings = ["Create a detailed description for the following product: Corelogic Smooth Mouse, belonging to category: Optical Mouse",
"Create a detailed description for the following product: Hoover Lightspeed, belonging to category: Cordless Vacuum Cleaner",
"Create a detailed description for the following product: Flattronic Cinematron, belonging to category: High Definition Flatscreen TV"]

# COMMAND ----------

predictions = []
for test in test_strings:
  prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  {}

  ### Response:""".format(test)
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

  generation_output = model.generate(
      input_ids=input_ids, max_new_tokens=156
  )
  predictions.append(tokenizer.decode(generation_output[0]))

# COMMAND ----------

def extract_response_text(input_string):
    start_marker = '### Response:'
    end_marker = '###'
    
    start_index = input_string.find(start_marker)
    if start_index == -1:
        return None
    
    start_index += len(start_marker)
    
    end_index = input_string.find(end_marker, start_index)
    if end_index == -1:
        return input_string[start_index:]
    
    return input_string[start_index:end_index].strip()

# COMMAND ----------

# predictions[2]

# COMMAND ----------

for i in range(3): 
  pred = predictions[i]
  text = test_strings[i]
  print(text+'\n')
  print(extract_response_text(pred))
  print('--------')

# COMMAND ----------


