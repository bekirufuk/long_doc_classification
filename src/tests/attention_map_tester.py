""" Imitates the structure of the training script
    but its only purpose to provide a test ground
    in a controlled, isolated environment.
"""

import sys
import os
import yaml
import wandb
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
from transformers import LongformerForSequenceClassification, LongformerConfig, get_cosine_schedule_with_warmup

sys.path.append(os.getcwd())
from src.utils.attention_mapper import map_tfidf
from src.data_processer.process import get_longformer_tokens

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_finetune_config():
    config_dir = 'src/config/finetune.yml'
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

    train_data = get_longformer_tokens(load_tokens=True, test_data_only=False, train_sample_size=finetune_config['train_sample_size'])
    patent_ids = train_data['patent_id']
    train_data = train_data.remove_columns(['patent_id', 'ipc_class', 'subclass'])
    train_data.set_format("torch")

    # Utilize the model with custom config file specifiyng classification labels.
    print('Initializing the Longformer model...')
    longformer_config = LongformerConfig.from_json_file('src/config/model_configs/longformer.json')
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', config=longformer_config)
    model.gradient_checkpointing_enable()

    for param in model.base_model.parameters():
        param.requires_grad = False
    
    train_dataloader = DataLoader(train_data, batch_size=finetune_config['train_batch_size'])
    #validation_dataloader = DataLoader(validation_data, batch_size=finetune_config['validation_batch_size'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_config['learning_rate'], weight_decay=finetune_config['weight_decay'])

    num_training_step = int(finetune_config['epochs'] * (finetune_config['train_sample_size'] / finetune_config['train_batch_size']))
    num_warmup_steps = int(finetune_config['warmup_ratio'] * num_training_step)

    # Learning rate scheduler for dynamic learning rate.
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_step
    )

    progress_bar = tqdm(range(num_training_step))
    log_interval = int(num_training_step/finetune_config['log_count'])

    # Dict to keep track of the running avg. scores
    train_tracker = {'running_loss':0, 'running_loss_counter':0, 'running_accuracy':0, 'running_accuracy_counter':0}
    best_train_accuracy=0

    # Load the sparse tfidf matrix and the feature_names(containing input_ids as words)
    tfidf_sparse = pd.read_pickle('data/refined_patents/tfidf/longformer_tokenizer_no_stopwords/train_tfidf_sparse.pkl')
    f_names = pd.read_pickle('data/refined_patents/tfidf/longformer_tokenizer_no_stopwords/train_f_list.pkl')

    step_counter=0
    model.train()
    print("\n----------\n TRAINING STARTED \n----------\n")
    for epoch in range(finetune_config['epochs']):
        for batch_id, batch in enumerate(train_dataloader):
            # Determine the range of tfidf sparse matrix for the current batch. These document scores will match exatcly with the current batch input docs.
            tfidf_range_start = finetune_config['train_batch_size']*batch_id
            tfidf_range_end = tfidf_range_start + finetune_config['train_batch_size']

            global_attention_map = map_tfidf(tfidf_sparse[tfidf_range_start:tfidf_range_end], f_names, batch['input_ids'])
            #print(global_attention_map)

            '''outputs = model(batch['input_ids'], 
                            labels=batch['labels'],
                            attention_map=batch['attention_map'],
                            global_attention_map=global_attention_map
                            )

            optimizer.zero_grad()
            accelerator.backward(outputs.loss)

            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
            step_counter+=1'''


    print("\n----------\n TRAINING FINISHED \n----------\n")

    """print("\n----------\n EVALUATION STARTED \n----------\n")
    torch.cuda.empty_cache()

    # Load the best model.
    #model = LongformerForSequenceClassification.from_pretrained('models/{model_type}/{model}'.format(model_type=finetune_config['model_type'], model=finetune_config['model']))

    test_data = get_longformer_tokens(test_data_only=True,load_tokens=True, test_sample_size=finetune_config['test_sample_size'])
    test_dataloader = DataLoader(test_data, batch_size=finetune_config['test_batch_size'])

    num_test_step = len(test_dataloader)
    progress_bar = tqdm(range(num_test_step))

    test_running_accuracy = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(test_dataloader):

            outputs = model(**batch)

            predictions = torch.argmax(outputs.logits, dim=-1)

            test_running_accuracy += accuracy_score(batch["labels"].cpu(), predictions.cpu())

            progress_bar.update(1)

    test_accuracy = test_running_accuracy / (batch_id+1)

    wandb.log({"Test Accuracy":test_accuracy})

    print("\n----------\n EVALUATION FINISHED \n----------\n")"""