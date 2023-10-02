import sys
import os
import yaml
import wandb
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import LongformerForSequenceClassification, LongformerConfig, get_cosine_schedule_with_warmup

sys.path.append(os.getcwd())
from src.data_processer.process import get_tokens
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_finetune_config():
    config_dir = 'src/config/finetune.yml'
    with open(config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

if __name__ == '__main__':
    
    #Initilize WandB from its config file.
    finetune_config = get_finetune_config()
    log_name = finetune_config['model'] + '_' + datetime.now().strftime("%Y-%m-%d-%H%M")
    if finetune_config['log_to_wandb']:
        wandb.init(project=finetune_config['project'], 
                    entity="bekirufuk", 
                    config=finetune_config,
                    job_type='finetuning',
                    name=log_name,
                    )
    
    #Define the available device andd clear the cache.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize Huggingface accelerator to manage GPU assignments of the data. No need to .to(device) after this.
    accelerator = Accelerator(mixed_precision='fp16')

    train_data = get_tokens(finetune_config['tokenizer'], test_data_only=False, train_sample_size=finetune_config['train_sample_size'])
    train_data = train_data.remove_columns(['patent_id', 'ipc_class', 'subclass'])
    train_data.set_format("torch")

    # Utilize the model with custom config file specifying classification labels.
    print('Initializing the Longformer model...')
    longformer_config = LongformerConfig.from_json_file('src/config/model_configs/longformer.json')
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', config=longformer_config)
    model.gradient_checkpointing_enable()
    model = torch.compile(model)

    if finetune_config['freeze_layer_count'] > 0:
        print("{} Layers frozen.".format(finetune_config['freeze_layer_count']))
        freezing_modules = [model.longformer.encoder.layer[:finetune_config['freeze_layer_count']]]

        for module in freezing_modules:
            for param in module.parameters():
                param.requires_grad = False

    train_dataloader = DataLoader(train_data, batch_size=finetune_config['train_batch_size'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_config['learning_rate'], weight_decay=finetune_config['weight_decay'])

    # Load the data with accelerator for a better GPU performance. No need to send it into the device.
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

    # Dict to keep track of the running avg. scores
    train_tracker = {'running_loss':0, 'running_loss_counter':0, 'running_accuracy':0, 'running_accuracy_counter':0}
    best_train_accuracy=0

    step_counter=0
    model.train()
    print("\n----------\n TRAINING STARTED \n----------\n")
    for epoch in range(finetune_config['epochs']):
        for batch_id, batch in enumerate(train_dataloader):

            is_log_step = step_counter % log_interval == 0 and step_counter!=0

            if is_log_step:
                outputs = model(batch['input_ids'], 
                            labels=batch['labels'],
                            attention_mask=batch['attention_mask'],
                            )

                optimizer.zero_grad()
                accelerator.backward(outputs.loss)

                optimizer.step()
                lr_scheduler.step()

                progress_bar.update(1)
                step_counter+=1

                # Update the loss and accuracy tracker values along with their counters
                train_tracker['running_loss'] += outputs.loss.cpu()
                train_tracker['running_loss_counter'] += 1

                predictions = torch.argmax(outputs.logits, dim=-1)
                train_tracker['running_accuracy'] += accuracy_score(batch["labels"].cpu(), predictions.cpu())
                train_tracker['running_accuracy_counter'] += 1

                mean_train_loss = train_tracker['running_loss'] / train_tracker['running_loss_counter']
                mean_train_accuracy = train_tracker['running_accuracy'] / train_tracker['running_accuracy_counter']

                # Save and overwrite the model if performs better than the last model.
                if mean_train_accuracy > best_train_accuracy:
                    best_train_accuracy = mean_train_accuracy
                    model.save_pretrained('models/{model_type}/{model}'.format(model_type=finetune_config['model_type'], model=finetune_config['model']))
                
                if finetune_config['log_to_wandb']:
                    wandb.log({
                        "Training Loss": mean_train_loss,
                        "Training Accuracy": mean_train_accuracy,
                        "Epoch":epoch+1,
                        "Learning Rate": lr_scheduler.get_last_lr()[0]
                    })

                # Reset the tracker values since this is a logging step. It 
                train_tracker = {'running_loss':0, 'running_loss_counter':0, 'running_accuracy':0, 'running_accuracy_counter':0}

            else:
                outputs = model(batch['input_ids'], 
                            labels=batch['labels'],
                            attention_mask=batch['attention_mask'],
                            )

                optimizer.zero_grad()
                accelerator.backward(outputs.loss)

                optimizer.step()
                lr_scheduler.step()

                progress_bar.update(1)
                step_counter+=1

                train_tracker['running_loss'] += outputs.loss.cpu()
                train_tracker['running_loss_counter'] += 1

                predictions = torch.argmax(outputs.logits, dim=-1)
                train_tracker['running_accuracy'] += accuracy_score(batch["labels"].cpu(), predictions.cpu())
                train_tracker['running_accuracy_counter'] += 1
    print("\n----------\n TRAINING FINISHED \n----------\n")

    print("\n----------\n EVALUATION STARTED \n----------\n")
    torch.cuda.empty_cache()

    # Load the best model.
    model = LongformerForSequenceClassification.from_pretrained('models/{model_type}/{model}'.format(model_type=finetune_config['model_type'], model=finetune_config['model']))
    model = torch.compile(model)

    test_data = get_tokens(finetune_config['tokenizer'], test_data_only=True, test_sample_size=finetune_config['test_sample_size'])
    test_data = test_data.remove_columns(['patent_id', 'ipc_class', 'subclass'])
    test_data.set_format("torch")

    test_dataloader = DataLoader(test_data, batch_size=finetune_config['test_batch_size'])

    # Load the data with accelerator for a better GPU performance. No need to send it into the device.
    test_dataloader, model = accelerator.prepare(
        test_dataloader, model
    )

    num_test_step = len(test_dataloader)
    progress_bar = tqdm(range(num_test_step))

    test_running_accuracy = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(test_dataloader):

            outputs = model(batch['input_ids'], 
                        labels=batch['labels'],
                        attention_mask=batch['attention_mask'],
                        )

            predictions = torch.argmax(outputs.logits, dim=-1)

            test_running_accuracy += accuracy_score(batch["labels"].cpu(), predictions.cpu())

            progress_bar.update(1)

    test_accuracy = test_running_accuracy / (batch_id+1)

    if finetune_config['log_to_wandb']:
        wandb.log({"Test Accuracy":test_accuracy})

    print("\n----------\n EVALUATION FINISHED \n----------\n")