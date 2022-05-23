import os
import sys
import config
import random

import numpy as np
import pandas as pd
import torch

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import warnings
warnings.filterwarnings("ignore")

def compute_metrics(predictions, references):

    accuracy = accuracy_score(references, predictions)

    cls_report = classification_report(references, predictions,
                                        labels=range(config.num_labels),
                                        target_names=config.labels_list,
                                        output_dict=True,
                                        zero_division=0,
                                        )

    precision, recall, f1, _ = precision_recall_fscore_support(references, predictions,
                                                                labels=range(config.num_labels),
                                                                average='weighted',
                                                                zero_division=0
                                                                )
    
    return {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1':f1, 'cls_report':cls_report}
    

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


def visualize_attention_map(map, step_id):
    global_list = np.array(map.cpu())
    token_len = len(global_list)

    att_window = config.model_config['attention_window'][0]

    s_map = np.zeros((token_len, token_len))
    
    for i in range(token_len):
        if global_list[i] == 1:
            s_map[i] = 1
            s_map[:, i] = 1
    
        starting_pos = int(max(i-att_window/2, 0))
        ending_pos = int(min(i+att_window/2, token_len))

        s_map[i][starting_pos:ending_pos] = 1  

    plt.imshow(s_map, cmap='Greys')
    save_dir = os.path.join(config.root_dir, 'longformer_global_attention_maps/')
    plt.savefig(save_dir+str(step_id)+'.jpg')
    plt.close()





dim_reducer = TSNE(n_components=2)
def visualize_layerwise_embeddings(hidden_states,masks,labels,epoch,layers_to_visualize):

    num_layers = len(layers_to_visualize)
    
    fig = plt.figure(figsize=(24,(num_layers/4)*6)) #each subplot of size 6x6, each row will hold 4 plots
    ax = [fig.add_subplot(int(num_layers/4),4,i+1) for i in range(num_layers)]
    
    labels = labels.cpu().numpy().reshape(-1)
    for i,layer_i in enumerate(layers_to_visualize):
        layer_embeds = hidden_states[layer_i]
        
        layer_averaged_hidden_states = torch.div(layer_embeds.sum(dim=1),masks.sum(dim=1,keepdim=True))
        layer_dim_reduced_embeds = dim_reducer.fit_transform(layer_averaged_hidden_states.cpu().detach().numpy())
        
        df = pd.DataFrame.from_dict({'x':layer_dim_reduced_embeds[:,0],'y':layer_dim_reduced_embeds[:,1],'label':labels})
        
        sns.scatterplot(data=df,x='x',y='y',hue='label',ax=ax[i])

    save_dir = os.path.join(config.root_dir, 'longformer_meta_info/layer_embeddings_'+str(epoch)+'.png')    
    plt.savefig(save_dir,format='png',pad_inches=0)
