import os
import sys
import pandas as pd
import config
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(predictions, references):

    precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average='micro')
    acc = accuracy_score(references, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def tfidf_attention_mapper(device, global_attention_mask, f_names, tfidf):
    # Function takes input_ids with the name of global_attention_mask.
    # The variable initially consist of input_ids however, its value will be changed completely to be equal to desired global_attention_mapping


    # Get ready the tfidf scores for the current batch
    tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

    # Put the mask in cpu temporarily for .apply_() process.
    global_attention_mask = global_attention_mask.type(torch.float32).cpu()

    for i, item in enumerate(global_attention_mask): # For every document(item) in the batch.
         
        # x is 1 for padding tokens, so it would be zero, all other tokens matches to their tfidf scores.
        globals = item.apply_(lambda x: 0 if x == 1 else tfidf[str(int(x))][i])

        # Obtain the index for the highest 128 tfidf scored 128 tokens.
        globals = torch.argsort(globals, dim=0, descending=True)[:128]

        # Mark the found indexes as 1, indicating a global connection to the all other tokens in the document.
        global_attention_mask[i][globals] = 1

    # Mark the first token being the <CLS> (noted as <s> in longformer) token of all of the batch as 1 gor global connection.
    global_attention_mask[:, 0] = 1
    global_attention_mask = torch.tensor(global_attention_mask.numpy(), dtype=torch.long, device=device)
    
    return global_attention_mask

def idf_attention_mapper(device, input_ids, token_idf):

    input_ids = input_ids.to('cpu')
    global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=device)

    for i, item in enumerate(input_ids): # For every document(item) in the batch.

        global_token_ids = torch.argsort(item.apply_(lambda x: token_idf[str(x)]), dim=0, descending=True)[:128]
        
        # Mark the found indexes as 1, indicating a global connection to the all other tokens in the document.
        global_attention_mask[i][global_token_ids] = 1

    # Mark the first token being the <CLS> (noted as <s> in longformer) token of all of the batch as 1 for global connection.
    global_attention_mask[:, 0] = 1
    return global_attention_mask

def idf_attention_analyzer(device, input_ids, labels, tokenizer):

    global_attention_mask = input_ids.to('cpu')
    token_idf = pd.read_pickle(os.path.join(config.data_dir, 'meta/token_idf.pkl'))

    #global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

    for i, item in enumerate(global_attention_mask): # For every document(item) in the batch.

        global_token_ids = torch.argsort(item.apply_(lambda x: token_idf[str(x)]), dim=0, descending=True)[:128]

        print('###################################')
        print('Class: {}'.format(config.id2label[labels[i].item()]))
        for k in range(global_token_ids.shape[0]):
            print(tokenizer.convert_ids_to_tokens([input_ids[i][global_token_ids[k]].item()]), item[global_token_ids[k]].item())
            #print(tokenizer.convert_ids_to_tokens([input_ids[i][global_token_ids[k].item()].item()]), global_token_ids[i][k])
        

        # Obtain the index for the highest 128 tfidf scored 128 tokens.
        # globals = torch.argsort(globals, dim=0, descending=True)[:128]

        # Mark the found indexes as 1, indicating a global connection to the all other tokens in the document.
        # global_attention_mask[i][globals] = 1

    # Mark the first token being the <CLS> (noted as <s> in longformer) token of all of the batch as 1 gor global connection.
    global_attention_mask[:, 0] = 1
    global_attention_mask = torch.tensor(global_attention_mask.numpy(), dtype=torch.long, device=device)


def visualize_attention_map(map):
    map_line = np.array(map)
    att_window = config.model_config['attention_window'][0]

    map = []
    for x in range(len(map_line)):
        copy_map_line = map_line.copy()

        starting_pos = int(max(x-att_window/2, 0))
        ending_pos = int(min(x+att_window/2, len(map_line)))

        copy_map_line[starting_pos:ending_pos] = 1  
        map.append(list(copy_map_line))
    plt.spy(map, precision = 0.1, markersize = 5)
    plt.show()
    return plt