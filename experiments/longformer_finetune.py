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

from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, LongformerConfig, get_scheduler, get_cosine_schedule_with_warmup

from datasets import Features, Value, ClassLabel, load_dataset, load_from_disk


def batch_tokenizer(batch, tokenizer):
    return tokenizer(batch["text"], padding='max_length', truncation=True)

# Load the existing tokenized data if wanted. Create a new one otherwise.
def get_tokenized_data():
    if config.load_saved_tokens:
        print("Using tokenized data...")
        tokenized_data = load_from_disk(os.path.join(config.data_dir, "longformer_tokenized/"))
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
        #tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=config.max_length)
        
        tokenized_data = dataset.map(batch_tokenizer, batched=True, remove_columns=['text'], fn_kwargs= {"tokenizer":tokenizer})
        tokenized_data = tokenized_data.rename_column("label","labels")
        print("Tokenization Completed")

    tokenized_data.set_format("torch")
    return tokenized_data

# Save tokenized data to be loaded afterwards
def save_tokenized_data():
    print("Saving tokenized data...")
    tokenized_data.save_to_disk(os.path.join(config.data_dir, "tokenized/"))

def get_model():
    if config.load_local_checkpoint:
        model = LongformerForSequenceClassification.from_pretrained(os.path.join(config.root_dir,"models/{}".format(config.model_name)))
    else:
        # Utilize the model with custom config file specifiyng classification labels.
        longformer_config = LongformerConfig.from_json_file(os.path.join(config.root_dir, 'longformer_config.json'))
        model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
            config=longformer_config
        )
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
    
    # You might want to change the wandb setings in order to training logger to work properly.
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
    del tokenized_data

    # Dataloaders for PyTorch implementation. Since the dataset already shuffled, no extra shuffle here.
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size)
    del train_data, test_data

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

    # Read the TF-IDF info in order to create a global attention map from it.
    f_names = pd.read_pickle('data/patentsview/meta/longformer_tokens_tfidf_feature_names.pkl')
    tfidf = pd.read_pickle('data/patentsview/meta/longformer_tokens_tfidf_sparse.pkl')

    # Start the logger
    model.train()
    print("\n----------\n TRAINING STARTED \n----------\n")
    step_counter=config.initial_step
    for epoch in range(config.num_epochs):
        for batch_id, batch in enumerate(train_dataloader):
            # Construct a global_attention_mask to decide which tokens will have glabal attention.
            # Pick a slice of tfidf that matches the documents of the current batch.
            # Since, they are created from the same tokenized data, tfidf document index and train data document index exactly marches.
            global_attention_mask = utils.idf_attention_mapper(device,
                                                            batch['input_ids'],
                                                            )

            outputs = model(**batch, global_attention_mask=global_attention_mask)
          
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
    running_f1 = 0
    running_pre = 0
    running_rec = 0

    for batch_id, batch in enumerate(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            global_attention_mask = utils.idf_attention_mapper(device,
                                                            batch['input_ids'],
                                                            )
            outputs = model(**batch, global_attention_mask=global_attention_mask)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        batch_result = utils.compute_metrics(predictions=predictions.cpu(), references=batch["labels"].cpu())
        running_acc += batch_result['accuracy']
        running_f1 += batch_result['f1']
        running_pre += batch_result['precision']
        running_rec += batch_result['recall']
        progress_bar.update(1)

        wandb.log({"Test Loss": outputs.loss,
                "Test Accuracy":batch_result['accuracy'],
                "Test F1":batch_result['f1'],
                })
    mean_f1 = running_f1/num_test_steps
    mean_acc = running_acc/num_test_steps
    mean_rec = running_rec/num_test_steps
    mean_pre = running_pre/num_test_steps
    wandb.log({"Test Mean-F1":mean_f1})
    wandb.log({"Test Mean-Acc":mean_acc})
    wandb.log({"Test Mean-Recall":mean_rec})
    wandb.log({"Test Mean-Precision":mean_pre})
    print("\n----------\n EVALUATION FINISHED \n----------\n")

    print("Mean of {} batches F1: {}".format(num_test_steps,mean_f1))