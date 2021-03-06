'''
    This script consist of various functions that takes a batch of 'input_ids' as input and outputs a same size matrix consisting ones and zeros.
    Ones representing the global connection between that token (represented by tokenizers input_id) and all the others for that input document.
    
    Detailed explanation for the usage of global_attentions in custom transformer models can be found at the docstring of
    'LongformerSequenceClassifierOutput' class in the HuggingFace repo: https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py 
'''

import numpy as np
import pandas as pd
from torch import zeros, long

# Uncomment below code for 'tfidf_qual_analysis' function.
from transformers import LongformerTokenizerFast
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=4096)
id2label =  {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
label2description= {'A': 'Human Necessities', 'B':'Performing Operations; Transporting', 'C':'Chemistry; Metallurgy','D':'Textiles; Paper',
                    'E':'Fixed Constructions','F':'Mechanical Engineering; Lighting; Heating; Weapons; Blasting','G':'Physics','H':'Electricity'}

def tfidf_qual_analysis(tfidf, f_names, input_ids, labels):

    # Convert the tfidf scores as DataFrame for the current batch.
    tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

    # Turn GPU Tensor input_ids to CPU DataFrame of strings, so the input_id values could be matched with tfidf column names (which are the same as input_ids).
    input_ids = pd.DataFrame(input_ids.cpu().detach().numpy(), dtype=str)
    
    for i in range(input_ids.shape[0]): # Iterate over the batch

        # From the tfidf matrix, obtain the values from row i and the colums that matches the corresponding documents input_ids.
        # Column names of the tfidf matrix are the same with the all input_ids, we just select the ones that match the current documents input_ids.
        tf_idf_values_of_input = tfidf.loc[i, input_ids.loc[i]]

        # Cast the tfidf values to numpy array and obtain the indices of a number of highest tfidf scores.
        # These indices correspond the indices of 'to be globally connected tokens' for the model forward. 
        tf_idf_values_of_input = np.array(tf_idf_values_of_input, dtype=np.float32)
        top_tfidf_idxs = np.argsort(tf_idf_values_of_input)[-128:]

        # Uncomment below code for a detailed view of the selected input_ids.
        '''print('sorted scores: ', np.sort(tf_idf_values_of_input,)[-128:])
        print('argsort idxs: ',top_tfidf_idxs)
        print('argsort values: ',tf_idf_values_of_input[top_tfidf_idxs])
        print('input ids: ', list(input_ids.loc[i][top_tfidf_idxs]))'''
        
        # Print the label description of the document along with the determinded global tokens.
        print(label2description[id2label[labels[i].item()]],':')
        print(tokenizer.convert_ids_to_tokens(list(input_ids.loc[i][top_tfidf_idxs])))
        print('\n################\n')

def map_tfidf(tfidf, f_names, input_ids, device):
    try:
        # Convert the tfidf scores as DataFrame for the current batch.
        tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

        # Turn GPU Tensor input_ids to CPU DataFrame of strings, so the input_id values could be matched with tfidf column names (which are the same as input_ids).
        input_ids = pd.DataFrame(input_ids.cpu().detach().numpy(), dtype=str)

        # Initilize an empty global_attention_map with the same size as the input.
        global_attention_map = zeros(input_ids.shape, dtype=long, device=device)
        
        for i in range(input_ids.shape[0]): # For every document(item) in the batch.

            # From the tfidf matrix, obtain the values from row i and the colums that matches the corresponding documents input_ids.
            # Column names of the tfidf matrix are the same with the all input_ids, we just select the ones that match the current documents input_ids.
            tf_idf_values_of_input = tfidf.loc[i, input_ids.loc[i]]
            
            # Cast the tfidf values to numpy array and obtain the indices of a number of highest tfidf scores.
            # These indices correspond the indices of 'to be globally connected tokens' for the model forward.
            tf_idf_values_of_input = np.array(tf_idf_values_of_input, dtype=np.float32)
            top_tfidf_idxs = np.argsort(tf_idf_values_of_input)[-128:]

            # Assign the value one to the positions of determined global tokens.
            global_attention_map[i][top_tfidf_idxs] = 1

        # Assign one to the first token of every document since it is the classification ([CLS]) token and should always be globally connected.
        global_attention_map[:, 0] = 1  
        return global_attention_map
    except Exception as e:
        print('Error: ',e)
        global_attention_map = zeros(input_ids.shape, dtype=long, device=device)
        global_attention_map[:, 0] = 1  
        return global_attention_map

def unique_tfidf_qual_analysis(tfidf, f_names, input_ids, labels):
    # Convert the tfidf scores as DataFrame for the current batch.
    tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

    # Turn GPU Tensor input_ids to CPU DataFrame of strings, so the input_id values could be matched with tfidf column names (which are the same as input_ids).
    input_ids = pd.DataFrame(input_ids.cpu().detach().numpy(), dtype=str)
    
    for i in range(input_ids.shape[0]): # Iterate over the batch

        # From the tfidf matrix, obtain the values from row i and the colums that matches the corresponding documents input_ids.
        # Column names of the tfidf matrix are the same with the all input_ids, we just select the ones that match the current documents input_ids.
        tf_idf_values_of_input = tfidf.loc[i, input_ids.loc[i]]
 
        tfidf_map = pd.DataFrame(data={'tfidf_values':tf_idf_values_of_input.values, 'input_ids':tf_idf_values_of_input.index})

        unique_tfidf_map = tfidf_map.drop_duplicates(subset=['input_ids'])

        top_tfidf_idxs = unique_tfidf_map.sort_values(by='tfidf_values')[-128:].index
      
        # Print the label description of the document along with the determinded global tokens.
        print(label2description[id2label[labels[i].item()]],':')
        print(tokenizer.convert_ids_to_tokens(list(input_ids.loc[i][top_tfidf_idxs])))
        print('\n################\n')

def map_unique_tfidf(tfidf, f_names, input_ids, device):
    try:
        # Convert the tfidf scores as DataFrame for the current batch.
        tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

        # Turn GPU Tensor input_ids to CPU DataFrame of strings, so the input_id values could be matched with tfidf column names (which are the same as input_ids).
        input_ids = pd.DataFrame(input_ids.cpu().detach().numpy(), dtype=str)

        # Initilize an empty global_attention_map with the same size as the input.
        global_attention_map = zeros(input_ids.shape, dtype=long, device=device)

        for i in range(input_ids.shape[0]): # Iterate over the batch

            # From the tfidf matrix, obtain the values from row i and the colums that matches the corresponding documents input_ids.
            # Column names of the tfidf matrix are the same with the all input_ids, we just select the ones that match the current documents input_ids.
            tf_idf_values_of_input = tfidf.loc[i, input_ids.loc[i]]
    
            tfidf_map = pd.DataFrame(data={'tfidf_values':tf_idf_values_of_input.values, 'input_ids':tf_idf_values_of_input.index})

            unique_tfidf_map = tfidf_map.drop_duplicates(subset=['input_ids'])

            top_tfidf_idxs = unique_tfidf_map.sort_values(by='tfidf_values')[-128:].index

            # Assign the value one to the positions of determined global tokens.
            global_attention_map[i][top_tfidf_idxs] = 1
        
        # Assign one to the first token of every document since it is the classification ([CLS]) token and should always be globally connected.
        global_attention_map[:, 0] = 1  
        return global_attention_map

    except Exception as e:
        print('Error: ',e)
        global_attention_map = zeros(input_ids.shape, dtype=long, device=device)
        global_attention_map[:, 0] = 1  
        return global_attention_map

def random_map(input_ids, device):
    # Initilize an empty global_attention_map with the same size as the input.
    global_attention_map = zeros(input_ids.shape, dtype=long, device=device)

    for i in range(input_ids.shape[0]):
        # Upper bound for the random index is the number of non padding tokens for the current input document.
        upper_bound = (input_ids[i]!=1).sum()
        random_idxs = np.random.choice(range(0,upper_bound), size=128, replace=False)

        # Assign the value one to the positions of random global tokens.
        global_attention_map[i][random_idxs] = 1

    # Assign one to the first token of every document since it is the classification ([CLS]) token and should always be globally connected.
    global_attention_map[:, 0] = 1  
    return global_attention_map

def first_n_map(input_ids, device):
    # Initilize an empty global_attention_map with the same size as the input.
    global_attention_map = zeros(input_ids.shape, dtype=long, device=device)

    first_n_idxs = range(0,129)

    # Assign one to the first token of every document since it is the classification ([CLS]) token and should always be globally connected.
    global_attention_map[:, first_n_idxs] = 1  
    return global_attention_map
