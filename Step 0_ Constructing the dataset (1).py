# Databricks notebook source
from datasets import load_dataset , Dataset, concatenate_datasets 
import numpy as np
import pandas as pd
import random

# COMMAND ----------

rd_ds = load_dataset("xiyuez/red-dot-design-award-product-description")
rd_df = pd.DataFrame(rd_ds['train'])
display(rd_df)

# COMMAND ----------

rd_df['instruction'] = 'Create a detailed description for the following product: '+ rd_df['product']+', belonging to category: '+ rd_df['category']
rd_df = rd_df[['instruction', 'description']]
display(rd_df)

# COMMAND ----------

rd_df_sample = rd_df.sample(n=5000, random_state=42)
display(rd_df_sample)

# COMMAND ----------

template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:\n"""

# COMMAND ----------

rd_df_sample['prompt'] = rd_df_sample["instruction"].apply(lambda x: template.format(x))

# COMMAND ----------

rd_df_sample.rename(columns={'description': 'response'}, inplace=True)

# COMMAND ----------

rd_df_sample['response'] = rd_df_sample['response'] +  "\n### End"
rd_df_sample = rd_df_sample[['prompt', 'response']]
display(rd_df_sample)


# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS description_generator;
# MAGIC USE description_generator;
# MAGIC      

# COMMAND ----------

spark.createDataFrame(rd_df_sample).write.saveAsTable('product_name_to_description')

