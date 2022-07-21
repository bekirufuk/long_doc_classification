import sys, os
import yaml
import wandb
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import BigBirdForSequenceClassification

sys.path.append(os.getcwd())
from src.data_processer.process import get_big_bird_tokens
from sklearn.metrics import accuracy_score

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
    wandb.init(project="Long Document Classification", 
                entity="bekirufuk", 
                config=finetune_config,
                job_type='finetuning',
                name=log_name,
                )
    
    #Define the available device andd clear the cache.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.empty_cache()

    # Initilize Huggingface accelerator to manage GPU assingments of the data. No need to .to(device) after this.
    accelerator = Accelerator(fp16=True)

    print("\n----------\n EVALUATION STARTED \n----------\n")
    torch.cuda.empty_cache()

    # Load the best model.
    model = BigBirdForSequenceClassification.from_pretrained('models/{model_type}/{model}'.format(model_type=finetune_config['model_type'], model=finetune_config['model']))

    test_data = get_big_bird_tokens(load_tokens=True, test_data_only=True, test_sample_size=finetune_config['test_sample_size'])
    test_data = test_data.remove_columns(['patent_id', 'ipc_class', 'subclass'])
    test_data.set_format("torch")

    test_dataloader = DataLoader(test_data, batch_size=finetune_config['test_batch_size'])

    # Load the data with accelerator for a better GPU performence. No need to send it into the device.
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

    wandb.log({"Test Accuracy":test_accuracy})

    print("\n----------\n EVALUATION FINISHED \n----------\n")