import sys, os
import yaml
import wandb
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import BigBirdForSequenceClassification, BigBirdConfig

sys.path.append(os.getcwd())
from src.data_processer.process import get_big_bird_tokens
from sklearn.metrics import accuracy_score, confusion_matrix
from src.utils.attention_mapper import block_map_tfidf

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
    wandb.init(project="Long Document Classification Small Scale", 
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

    # Load the sparse tfidf matrix and the feature_names(containing input_ids as words)
    #tfidf_sparse = pd.read_pickle('data/refined_patents/tfidf/big_bird/test_tfidf_sparse.pkl')
    #f_names = pd.read_pickle('data/refined_patents/tfidf/big_bird/test_f_list.pkl')

    print("\n----------\n EVALUATION STARTED \n----------\n")
    torch.cuda.empty_cache()

    # Load the best model.
    model = BigBirdForSequenceClassification.from_pretrained('models/{model_type}/{model}'.format(model_type=finetune_config['model_type'], model=finetune_config['model']))

    '''# Utilize the model with custom config file specifiyng classification labels.
    print('Initializing the Big Bird model...')
    bigbirg_config = BigBirdConfig.from_json_file('src/config/model_configs/bigbird.json')
    model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', config=bigbirg_config)
    model.gradient_checkpointing_enable()'''

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

    predictions_list = []
    len_list= []
    is_equal_list = []

    confmatrix = np.zeros((finetune_config['classes'],finetune_config['classes']), dtype=np.int32)
    with torch.no_grad():
        for batch_id, batch in enumerate(test_dataloader):

            # Determine the range of tfidf sparse matrix for the current batch. These document scores will match exatcly with the current batch input docs.
            #tfidf_range_start = finetune_config['test_batch_size']*batch_id
            #tfidf_range_end = tfidf_range_start + finetune_config['test_batch_size']

            #tfidf_batch_map = block_map_tfidf(tfidf_sparse[tfidf_range_start:tfidf_range_end], f_names, batch['input_ids'], device)

            outputs = model(batch['input_ids'], 
                        labels=batch['labels'],
                        attention_mask=batch['attention_mask'],
                        special_random_blocks = None
                        )

            predictions = torch.argmax(outputs.logits, dim=-1)

            test_running_accuracy += accuracy_score(batch["labels"].cpu(), predictions.cpu())

            # Calculate the token count omitting padding tokens
            input_len_without_padding = torch.count_nonzero(batch['input_ids'], dim=1)

            confmatrix += confusion_matrix(batch['labels'].cpu(), predictions.cpu(), labels=range(finetune_config['classes']))

            len_list.extend(input_len_without_padding.cpu().detach().numpy())
            predictions_list.extend(predictions.cpu().detach().numpy())
            is_equal_list.extend(torch.eq(predictions, batch['labels']).cpu().detach().numpy())
            progress_bar.update(1)
    
    confmatrix = px.imshow(confmatrix, text_auto=True, aspect='equal',
                        color_continuous_scale ='Blues',
                        x=finetune_config['labels_list'],
                        y=finetune_config['labels_list'],
                        labels={'x':'Prediction', 'y':'Actual'}
                        )
    
    lengthwise_results = pd.DataFrame(data={'len':len_list, 'prediction':predictions_list, 'label':test_data['labels'], 'is_correct':is_equal_list})
    test_accuracy = test_running_accuracy / (batch_id+1)

    wandb.log({"Test Accuracy":test_accuracy,
                'Lengthwise Performance':wandb.data_types.Plotly(px.histogram(lengthwise_results, 
                                                                              x='len',
                                                                              marginal="violin",
                                                                              color='is_correct',
                                                                              hover_data=lengthwise_results.columns,
                                                                              text_auto=True,
                                                                              color_discrete_map={'True':'Green','False':'Red'},
                                                                              barmode='group',
                                                                              nbins=30)),
                'Confusion Matrix': wandb.data_types.Plotly(confmatrix)
                })
    print(np.histogram(lengthwise_results[lengthwise_results['is_correct']==True]['len'], bins= 30)[0])
    print(np.histogram(lengthwise_results[lengthwise_results['is_correct']==False]['len'], bins= 30)[0])
    print("\n----------\n EVALUATION FINISHED \n----------\n")