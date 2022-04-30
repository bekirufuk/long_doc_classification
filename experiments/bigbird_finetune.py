import os
from random import seed
import sys
import config, utils
import pandas as pd
import wandb

from tqdm.auto import tqdm

from accelerate import Accelerator

import torch
from torch.utils.data import DataLoader

from transformers import BigBirdTokenizerFast, BigBirdForSequenceClassification, BigBirdConfig, get_scheduler, get_cosine_schedule_with_warmup

from datasets import Features, Value, ClassLabel, load_dataset, load_from_disk


def batch_tokenizer(batch, tokenizer):
    return tokenizer(batch["text"], padding='max_length', truncation=True)

# Load the existing tokenized data if wanted. Create a new one otherwise.
def get_tokenized_data():
    if config.load_saved_tokens:
        print("Using tokenized data...")
        tokenized_data = load_from_disk(os.path.join(config.data_dir, "bigbird_tokenized/"))
    else:
        print("Loading dataset from csv to be tokenized...")

        class_names = config.labels_list
        features = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})
        data_files = os.path.join(config.data_dir, 'chunks/*.csv')
        dataset = load_dataset('csv', data_files=data_files, features=features,
                                cache_dir=os.path.join(config.data_dir, 'cache/'),
                                )
        dataset = dataset['train'].train_test_split(test_size=config.test_split_ratio)
        dataset = dataset.shuffle(seed=config.seed)
        print('Dataset Loaded:')
        print(dataset)

        # Upload to huggingfacehub if True and repo name specified
        if config.upload_to_hf and config.repo_name != '':
            dataset.push_to_hub("ufukhaman/uspto_patents_2019", private=True)

        # Utilize the tokenizer and run it on the dataset with batching.
        print("Tokenization Started...")
        tokenizer = BigBirdTokenizerFast.from_pretrained('google/bigbird-roberta-base', max_length=config.max_length)
        tokenized_data = dataset.map(batch_tokenizer, batched=True, remove_columns=['text'], fn_kwargs= {"tokenizer":tokenizer})
        tokenized_data = tokenized_data.rename_column("label","labels")
        print("Tokenization Completed")

    tokenized_data.set_format("torch")
    return tokenized_data

# Save tokenized data to be loaded afterwards
def save_tokenized_data():
    print("Saving tokenized data...")
    tokenized_data.save_to_disk(os.path.join(config.data_dir, "bigbird_tokenized/"))

def get_model():
    if config.load_local_checkpoint:
        model = BigBirdForSequenceClassification.from_pretrained(os.path.join(config.root_dir,"models/{}".format(config.model_name)))
    else:
        # Utilize the model with custom config file specifiyng classification labels.
        bigbird_config = BigBirdConfig.from_json_file(os.path.join(config.root_dir, 'bigbird_config.json'))
        model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base',
                                                                config=bigbird_config)
    return model

# Trim and partition the dataset based on sample sizes.
def get_partitions():
    if config.downsample:
        train_data = tokenized_data["train"].select(range(config.num_train_samples))
        test_data = tokenized_data["test"].select(range(config.num_test_samples))
    else:
        train_data = tokenized_data["train"]
        test_data = tokenized_data["test"]
    return train_data, test_data

if __name__ == '__main__':
    
    wandb.init(project="custom_transformers", 
                entity="bekirufuk", 
                config=config.wandb_config,
                job_type='finetuning_test',
                name=config.log_name,
                )
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.empty_cache()
    accelerator = Accelerator(fp16=True)

    tokenized_data = get_tokenized_data()

    if config.save_tokens and not config.load_saved_tokens:
        save_tokenized_data()

    train_data, test_data = get_partitions()

    # Dataloaders for PyTorch implementation. Since the dataset already shuffled, no extra shuffle here.
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size)

    model = get_model()
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Load the data with accelerator for a better GPU performence. No need to send it into the device.
    train_dataloader, test_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, test_dataloader, model, optimizer
    )
    
    # Epoch times number of batches gives the total step count. Feed it into tqdm for a progress bar with this many steps.
    num_training_steps = config.num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    # Scheduler for dynamic learning rate.
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Construct a global_attention_mask to decide which tokens will have glabal attention.
    #global_attention_mask = utils.attention_mapper(device)

    # Start the logger
    model.train()
    print("\n----------\n TRAINING STARTED \n----------\n")
    step_counter=config.initial_step
    for epoch in range(config.num_epochs):
        for batch_id, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            if step_counter % config.log_interval == 0:
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                batch_result = utils.compute_metrics(predictions=predictions.cpu(), references=batch["labels"].cpu())['accuracy']
                wandb.log({"Training Loss": loss, "Training Accuracy":batch_result})

                if step_counter % (config.log_interval*20) == 0 and step_counter != config.initial_step:
                    model.save_pretrained(os.path.join(config.root_dir,"models/{}_{}".format(config.log_name,step_counter)))

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            step_counter+=1
    accelerator.free_memory()
    print("\n----------\n TRAINING FINISHED \n----------\n")

    num_test_steps = len(test_dataloader)
    progress_bar = tqdm(range(num_test_steps))
    model.eval()
    print("\n----------\n EVALUATION STARTED \n----------\n")
    running_acc = 0
    for batch_id, batch in enumerate(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        batch_result = utils.compute_metrics(predictions=predictions.cpu(), references=batch["labels"].cpu())
        running_acc += batch_result['f1']
        progress_bar.update(1)

        wandb.log({"Test Loss": outputs.loss,
                "Test Accuracy":batch_result['accuracy'],
                "Test F1":batch_result['f1'],
                })
    mean_f1 = running_acc/num_test_steps
    wandb.log({"Test Mean-F1":mean_f1})
    print("\n----------\n EVALUATION FINISHED \n----------\n")

    print("Mean of {} batches F1: {}".format(num_test_steps,mean_f1))