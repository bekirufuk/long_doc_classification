import os
from random import seed
import sys
import config, utils
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import LongformerTokenizerFast
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
        
        tokenized_data = dataset.map(batch_tokenizer, batched=True, remove_columns=['text'], fn_kwargs= {"tokenizer":tokenizer})
        tokenized_data = tokenized_data.rename_column("label","labels")
        print("Tokenization Completed")

    tokenized_data.set_format("torch")
    return tokenized_data

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.empty_cache()

    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=config.max_length)
    tokenized_data = get_tokenized_data()

    train_data, test_data = get_partitions()
    del tokenized_data

    # Dataloaders for PyTorch implementation. Since the dataset already shuffled, no extra shuffle here.
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size)
    del train_data, test_data

    for batch_id, batch in enumerate(train_dataloader):
        global_attention_mask = utils.idf_attention_analyzer(device,
                                                        batch['input_ids'],
                                                        batch['labels'],
                                                        tokenizer
                                                        )