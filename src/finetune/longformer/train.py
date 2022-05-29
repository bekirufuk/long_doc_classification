import sys
import os
import yaml
import wandb
from tqdm.auto import tqdm
from datetime import datetime

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import LongformerForSequenceClassification, LongformerConfig, get_cosine_schedule_with_warmup

sys.path.append(os.getcwd())
from src.data_processer.process import get_longformer_tokens
from src.utils.evaluator import get_accuracy


def get_wandb_config():
    config_dir = 'src/config/wandb.yml'
    with open(config_dir,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def get_model(load_model=False):
    if load_model:
        #Load a local checkpoint of the  model.
        model = LongformerForSequenceClassification.from_pretrained('models\longformer')
    else:
        # Utilize the model with custom config file specifiyng classification labels.
        print('Initializing the Longformer model...')
        longformer_config = LongformerConfig.from_json_file('src/config/model_configs/longformer.json')
        model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
            config=longformer_config
        )
    return model

if __name__ == '__main__':
    
    #Initilize WandB from its config file.
    wandb_config = get_wandb_config()
    log_name = wandb_config['model'] + '_' + datetime.now().strftime("%Y-%m-%d-%H%M")
    """wandb.init(project="custom_transformers", 
                entity="bekirufuk", 
                config=wandb_config,
                job_type='finetuning_test',
                name=log_name,
                )"""
    
    #Define the available device and clear the cache.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.empty_cache()

    accelerator = Accelerator(fp16=True)

    train_data, validation_data = get_longformer_tokens(load_tokens=True)
    model = get_model()

    train_dataloader = DataLoader(train_data, batch_size=8)
    validation_dataloader = DataLoader(validation_data, batch_size=wandb_config['batch_size'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb_config['learning_rate'], weight_decay=wandb_config['weight_decay'])

    # Load the data with accelerator for a better GPU performence. No need to send it into the device.
    train_dataloader, test_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, validation_dataloader, model, optimizer
    )

    num_training_step = int(wandb_config['epochs'] * (wandb_config['num_train_samples'] / wandb_config['batch_size']))
    num_warmup_steps = int(0.1 * num_training_step)
    # Scheduler for dynamic learning rate.
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_step
    )
    progress_bar = tqdm(range(num_training_step))
    log_interval = int(num_training_step/wandb_config['log_count'])

    train_tracker = {'running_train_loss':0, 'running_train_loss_counter':0, 'running_train_accuracy':0, 'running_train_accuracy_counter':0}

    step_counter=0
    model.train()
    print("\n----------\n TRAINING STARTED \n----------\n")
    for epoch in range(wandb_config['epochs']):
        for batch_id, batch in enumerate(train_dataloader):

            is_log_step = step_counter % log_interval == 0 and step_counter!=0

            if is_log_step:
                outputs = model(**batch)

                optimizer.zero_grad()
                accelerator.backward(outputs.loss)

                optimizer.step()
                lr_scheduler.step()

                progress_bar.update(1)
                step_counter+=1

                #Update the loss and accuracy tracker values along with their counters
                train_tracker['running_train_loss'] += outputs.loss
                train_tracker['running_train_loss_counter'] += 1

                predictions = torch.argmax(outputs.logits, dim=-1)
                train_tracker['running_train_accuracy'] += get_accuracy(batch["labels"].cpu(), predictions.cpu())
                train_tracker['running_train_accuracy_counter'] += 1

                #Log the log to WandB
                wandb.log({
                    "Training Loss": train_tracker['running_train_loss'] / train_tracker['running_train_loss_counter'],
                    "Training Accuracy": train_tracker['running_train_accuracy'] / train_tracker['running_train_accuracy_counter']
                })

                # Reset the tracker values since this is a logging step. It 
                train_tracker = {'running_train_loss':0, 'running_train_loss_counter':0, 'running_train_accuracy':0, 'running_train_accuracy_counter':0}


            else:
                outputs = model(**batch)

                optimizer.zero_grad()
                accelerator.backward(outputs.loss)

                optimizer.step()
                lr_scheduler.step()

                progress_bar.update(1)
                step_counter+=1

                train_tracker['running_train_loss'] += outputs.loss
                train_tracker['running_train_loss_counter'] += 1

                predictions = torch.argmax(outputs.logits, dim=-1)
                train_tracker['running_train_accuracy'] += get_accuracy(batch["labels"].cpu(), predictions.cpu())
                train_tracker['running_train_accuracy_counter'] += 1
