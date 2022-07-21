""" Imitates the structure of the training script
    but its only purpose to provide a test ground
    in a controlled, isolated environment.
"""

import sys
import os
import yaml
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import LongformerForSequenceClassification, LongformerConfig

sys.path.append(os.getcwd())
from src.utils.attention_mapper import random_map
from src.data_processer.process import get_longformer_tokens

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_finetune_config():
    config_dir = 'src/config/test.yml'
    with open(config_dir,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

if __name__ == '__main__':
    
    #Initilize WandB from its config file.
    finetune_config = get_finetune_config()
    log_name = finetune_config['model'] + '_' + datetime.now().strftime("%Y-%m-%d-%H%M")
    
    #Define the available device andd clear the cache.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.empty_cache()

    # Utilize the model with custom config file specifiyng classification labels.
    print('Initializing the Longformer model...')
    longformer_config = LongformerConfig.from_json_file('src/config/model_configs/longformer.json')
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', config=longformer_config)
    model.gradient_checkpointing_enable()

    # Initilize Huggingface accelerator to manage GPU assingments of the data. No need to .to(device) after this.

    train_data = get_longformer_tokens(load_tokens=True, test_data_only=False, train_sample_size=finetune_config['train_sample_size'])
    patent_ids = train_data['patent_id']
    train_data = train_data.remove_columns(['patent_id', 'ipc_class', 'subclass'])
    train_data.set_format("torch")
    
    train_dataloader = DataLoader(train_data, batch_size=finetune_config['train_batch_size'])
    #validation_dataloader = DataLoader(validation_data, batch_size=finetune_config['validation_batch_size'])


    num_training_step = int(finetune_config['epochs'] * (finetune_config['train_sample_size'] / finetune_config['train_batch_size']))

    progress_bar = tqdm(range(num_training_step))

    # Load the sparse tfidf matrix and the feature_names(containing input_ids as words)
    #tfidf_sparse = pd.read_pickle('data/refined_patents/tfidf/longformer_tokenizer_no_stopwords/train_tfidf_sparse.pkl')
    #f_names = pd.read_pickle('data/refined_patents/tfidf/longformer_tokenizer_no_stopwords/train_f_list.pkl')

    step_counter=0
    for epoch in range(finetune_config['epochs']):
        for batch_id, batch in enumerate(train_dataloader):
            # Determine the range of tfidf sparse matrix for the current batch. These document scores will match exatcly with the current batch input docs.
            tfidf_range_start = finetune_config['train_batch_size']*batch_id
            tfidf_range_end = tfidf_range_start + finetune_config['train_batch_size']

            global_attention_map = random_map(batch['input_ids'], 'cpu')

            outputs = model(batch['input_ids'], 
                            labels=batch['labels'],
                            attention_mask=batch['attention_mask'],
                            global_attention_mask=global_attention_map
                            )