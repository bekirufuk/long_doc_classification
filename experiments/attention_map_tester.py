import os
import sys
import config, utils
import pandas as pd
import numpy as np
from random import seed
import wandb

import plotly.express as px

import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import normalize
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, LongformerConfig, get_cosine_schedule_with_warmup
from datasets import Features, Value, ClassLabel, load_dataset, load_from_disk


def batch_tokenizer(batch, tokenizer):
    return tokenizer(list(map(lambda x: x.lower(), batch["text"])), padding='max_length', truncation=True)

# Load the existing tokenized data if wanted. Create a new one otherwise.
def get_tokenized_data():
    if config.load_saved_tokens:
        print("Using tokenized data...")
        tokenized_data = load_from_disk(os.path.join(config.data_dir, "longformer_tokenized/"))
    else:
        print("Loading dataset from csv to be tokenized...")
        features = Features({'text': Value('string'),
                             'label': ClassLabel(names=config.labels_list),
                             'ipc_class': Value('string'),
                             'subclass': Value('string'),
                            })
        data_files = os.path.join(config.data_dir, 'chunks/*.csv')
        dataset = load_dataset('csv', data_files=data_files, features=features,
                                #cache_dir=os.path.join(config.data_dir, 'cache/'),
                                )
        dataset = dataset['train'].train_test_split(test_size=config.test_split_ratio)
        dataset = dataset.shuffle(seed=config.seed)
        print('Dataset Loaded:')

        # Upload to huggingfacehub if True and repo name specified
        if config.upload_to_hf and config.upload_repo_name != '':
            dataset.push_to_hub(config.upload_repo_name, private=True)
            
        # Utilize the tokenizer and run it on the dataset with batching.
        print("Tokenization Started...")
        tokenized_data = dataset.map(batch_tokenizer, batched=True, remove_columns=['text', 'ipc_class', 'subclass'], fn_kwargs= {"tokenizer":tokenizer})
        tokenized_data = tokenized_data.rename_column("label","labels")
        print("Tokenization Completed")

    tokenized_data.set_format("torch")
    return tokenized_data

# Save tokenized data to be loaded afterwards
def save_tokenized_data(tokenized_data):
    print("Saving tokenized data...")
    tokenized_data.save_to_disk(os.path.join(config.data_dir, "longformer_tokenized/"))

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
def get_partitions(tokenized_data):
    if config.downsample:
        train_data = tokenized_data["train"].select(range(config.num_train_samples))
        test_data = tokenized_data["test"].select(range(config.num_test_samples))
    else:
        train_data = tokenized_data["train"]
        test_data = tokenized_data["test"]
    return train_data, test_data

def empty_confmatrix():
    return np.zeros((config.num_labels,config.num_labels), dtype=np.int32)


if __name__ == '__main__':
    
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=config.max_length)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.empty_cache()
    accelerator = Accelerator(fp16=True)

    tokenized_data = get_tokenized_data()

    if config.save_tokens and not config.load_saved_tokens:
        save_tokenized_data(tokenized_data)

    train_data, test_data = get_partitions(tokenized_data)
    del tokenized_data

    # Dataloaders for PyTorch implementation. Since the dataset already shuffled, no extra shuffle here.
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size)
    del train_data, test_data

    model = get_model()
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Scheduler for dynamic learning rate.
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.num_train_steps
    )

    token_idf = pd.read_pickle(os.path.join(config.data_dir, 'meta/token_idf.pkl'))
    confmatrix = empty_confmatrix()

    progress_bar = tqdm(range(config.num_train_steps))
    model.train()
    print("\n----------\n TRAINING STARTED \n----------\n")
    step_counter=config.initial_step
    for epoch in range(config.num_epochs):
        for batch_id, batch in enumerate(train_dataloader):
            # Construct a global_attention_mask to decide which tokens will have glabal attention.
            #global_attention_mask = utils.idf_attention_mapper(device, batch['input_ids'], token_idf)


            outputs = model(**batch)#, global_attention_mask=global_attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accelerator.backward(outputs.loss)
            print('Loss: ', outputs.loss)
            confmatrix += confusion_matrix(batch["labels"].cpu(), predictions.cpu(), labels=range(config.num_labels))

            if step_counter % config.log_interval == 0 and step_counter!=0:
                
                batch_accuracy = accuracy_score(batch["labels"].cpu(), predictions.cpu())

                fig = px.imshow(confmatrix, text_auto=True, aspect='equal',
                                color_continuous_scale ='Blues',
                                x=config.labels_list,
                                y=config.labels_list,
                                labels={'x':'Prediction', 'y':'Actual'}
                                )

                norm_fig = px.imshow(np.round_(normalize(confmatrix, norm='l1', axis=0), decimals=4), text_auto=True, aspect='equal',
                                color_continuous_scale ='Blues',
                                x=config.labels_list,
                                y=config.labels_list,
                                labels={'x':'Prediction', 'y':'Actual'}
                                )
                
                """wandb.log({'Training Confusion Matrix': wandb.data_types.Plotly(fig)})
                wandb.log({'Training Norm Confusion Matrix': wandb.data_types.Plotly(norm_fig)})

                wandb.log({"Training Loss": outputs.loss,
                           "Training Accuracy":batch_accuracy,
                           "Epoch":epoch+1,
                           "Learning Rate": lr_scheduler.get_last_lr()[0],
                           })"""
                confmatrix = empty_confmatrix()

                if step_counter % (config.log_interval*20) == 0 and step_counter != config.initial_step:
                    model.save_pretrained(os.path.join(config.root_dir,"models/{}_{}".format(config.log_name,step_counter)))

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            step_counter+=1
    accelerator.free_memory()
    print("\n----------\n TRAINING FINISHED \n----------\n")

    confmatrix = empty_confmatrix()
    progress_bar = tqdm(range(config.num_test_batches))
    model.eval()
    print("\n----------\n EVALUATION STARTED \n----------\n")

    all_predictions = torch.zeros((config.batch_size), device='cpu')
    all_labels = torch.zeros((config.batch_size), device='cpu')

    for batch_id, batch in enumerate(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            global_attention_mask = utils.idf_attention_mapper(device, batch['input_ids'], token_idf)
            outputs = model(**batch, global_attention_mask=global_attention_mask)

        predictions = torch.argmax(outputs.logits, dim=-1)

        all_predictions = torch.cat((all_predictions, predictions.cpu()))
        all_labels = torch.cat((all_labels, batch['labels'].cpu()))

        confmatrix += confusion_matrix(batch['labels'].cpu(), predictions.cpu(), labels=range(config.num_labels))
        fig = px.imshow(confmatrix, text_auto=True, aspect='equal',
                        color_continuous_scale ='Blues',
                        x=config.labels_list,
                        y=config.labels_list,
                        labels={'x':'Prediction', 'y':'Actual'}
                        )

        norm_fig = px.imshow(np.round_(normalize(confmatrix, norm='l1', axis=0), decimals=4), text_auto=True, aspect='equal',
                color_continuous_scale ='Blues',
                x=config.labels_list,
                y=config.labels_list,
                labels={'x':'Prediction', 'y':'Actual'}
                )

        # wandb.log({'Test Confusion Matrix': wandb.data_types.Plotly(fig)})
        # wandb.log({'Test Norm Confusion Matrix': wandb.data_types.Plotly(norm_fig)})

        progress_bar.update(1)

    test_results = utils.compute_metrics(predictions=all_predictions, references=all_labels)
    """wandb.log({"Classification Report":test_results['cls_report'],
               "Test Accuracy":test_results['accuracy'],
               "Test F1":test_results['f1'],
               "Test Precision":test_results['precision'],
               "Test Recall":test_results['recall'],
             })"""
    print("\n----------\n EVALUATION FINISHED \n----------\n")