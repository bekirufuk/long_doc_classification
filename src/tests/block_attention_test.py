import sys, os
import yaml
import wandb
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import BigBirdForSequenceClassification, BigBirdConfig, get_cosine_schedule_with_warmup

sys.path.append(os.getcwd())
from src.data_processer.process import get_big_bird_tokens
from src.utils.attention_mapper import block_map_pmi

from sklearn.metrics import accuracy_score

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

    # Initilize Huggingface accelerator to manage GPU assingments of the data. No need to .to(device) after this.
    accelerator = Accelerator(fp16=True)

    train_data = get_big_bird_tokens(load_tokens=True, test_data_only=False, train_sample_size=finetune_config['train_sample_size'])
    train_data = train_data.remove_columns(['patent_id', 'ipc_class', 'subclass'])
    train_data.set_format("torch")

    # Utilize the model with custom config file specifiyng classification labels.
    print('Initializing the Big Bird model...')
    bigbirg_config = BigBirdConfig.from_json_file('src/config/model_configs/bigbird.json')
    model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', config=bigbirg_config)
    model.gradient_checkpointing_enable()

    train_dataloader = DataLoader(train_data, batch_size=finetune_config['train_batch_size'])
    #validation_dataloader = DataLoader(validation_data, batch_size=finetune_config['validation_batch_size'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_config['learning_rate'], weight_decay=finetune_config['weight_decay'])

    # Load the data with accelerator for a better GPU performence. No need to send it into the device.
    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer
    )

    num_training_step = int(finetune_config['epochs'] * (finetune_config['train_sample_size'] / finetune_config['train_batch_size']))
    num_warmup_steps = int(finetune_config['warmup_rate'] * num_training_step)

    # Learning rate scheduler for dynamic learning rate.
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_step
    )

    progress_bar = tqdm(range(num_training_step))
    log_interval = int(num_training_step/finetune_config['log_count'])

    print('Loading PMI Scores...')
    pmi_scores = pd.read_pickle('data/refined_patents/stats/bigram_pmi_scores_w10_int.pkl')
    print('PMI Scores Loaded')

    step_counter=0
    model.train()
    print("\n----------\n TRAINING STARTED \n----------\n")
    for epoch in range(finetune_config['epochs']):
        for batch_id, batch in enumerate(train_dataloader):

            pmi_batch_map = block_map_pmi(pmi_scores, batch['input_ids'], device)

            outputs = model(batch['input_ids'], 
                            labels=batch['labels'],
                            attention_mask=batch['attention_mask'],
                            special_random_blocks = pmi_batch_map
                            )
                            
            optimizer.zero_grad()
            accelerator.backward(outputs.loss)

            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
            step_counter+=1
