'''
    This script consist of various functions that takes a batch of 'input_ids' as input and outputs a same size matrix consisting ones and zeros.
    Ones representing the global connection between that token (represented by tokenizers input_id) and all the others for that input document.
    
    Detailed explanation for the usage of global_attentions in custom transformer models can be found at the docstring of
    'LongformerSequenceClassifierOutput' class in the HuggingFace repo: https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py 
'''

import sys
import numpy as np
import pandas as pd
from torch import zeros, long
import matplotlib.pyplot as plt
from itertools import permutations

# Uncomment below code for 'tfidf_qual_analysis' function.
'''from transformers import LongformerTokenizerFast
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=4096)
id2label =  {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
label2description= {'A': 'Human Necessities', 'B':'Performing Operations; Transporting', 'C':'Chemistry; Metallurgy','D':'Textiles; Paper',
                    'E':'Fixed Constructions','F':'Mechanical Engineering; Lighting; Heating; Weapons; Blasting','G':'Physics','H':'Electricity'}'''

def tfidf_qual_analysis(tfidf, f_names, input_ids, labels, device):

    # Convert the tfidf scores as DataFrame for the current batch.
    tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

    # Assingn zero to some specific columns of tfidf matrix. These are the column names for the following tokens: ['[PAD]', '.', ',', '...' ].
    tfidf.loc[:,['0','1', '4', '6', '38917']] = 0
    
    # Turn GPU Tensor input_ids to CPU DataFrame of strings, so the input_id values could be matched with tfidf column names (which are the same as input_ids).
    input_ids = pd.DataFrame(input_ids.cpu().detach().numpy(), dtype=str)

    # Initilize an empty global_attention_map with the same size as the input.
    global_attention_map = zeros(input_ids.shape, dtype=long, device=device)

    for i in range(input_ids.shape[0]): # Iterate over the batch

        # From the tfidf matrix, obtain the values from row i and the colums that matches the corresponding documents input_ids.
        # Column names of the tfidf matrix are the same with the all input_ids, we just select the ones that match the current documents input_ids.
        tf_idf_values_of_input = tfidf.loc[i, input_ids.loc[i]]

        # Cast the tfidf values to numpy array and obtain the indices of a number of highest tfidf scores.
        # These indices correspond the indices of 'to be globally connected tokens' for the model forward. 
        tf_idf_values_of_input = np.array(tf_idf_values_of_input, dtype=np.float32)
        top_tfidf_idxs = np.argsort(tf_idf_values_of_input)[-512:]
        global_attention_map[i][top_tfidf_idxs] = 1

        '''# Uncomment below code for a detailed view of the selected input_ids.
        print('sorted scores: ', np.sort(tf_idf_values_of_input,)[-1023:])
        print('argsort idxs: ',top_tfidf_idxs)
        print('argsort values: ',tf_idf_values_of_input[top_tfidf_idxs])
        print('input ids: ', list(input_ids.loc[i][top_tfidf_idxs]))'''
        
        # Print the label description of the document along with the determinded global tokens.
        #print(label2description[id2label[labels[i].item()]],':')
        #print(tokenizer.convert_ids_to_tokens(list(input_ids.loc[i][top_tfidf_idxs])))
        #print('\n################\n')
    global_attention_map[:, 0] = 1  
    visualize_attention_map(global_attention_map[0])

def map_tfidf(tfidf, f_names, input_ids, device):
    try:
        # Convert the tfidf scores as DataFrame for the current batch.
        tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

        # Assingn zero to some specific columns of tfidf matrix. These are the column names for the following tokens: ['[PAD]', '.', ',', '...' ].
        tfidf.loc[:,['0','1', '4', '6', '38917']] = 0

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

    # Assingn zero to some specific columns of tfidf matrix. These are the column names for the following tokens: ['[PAD]', '.', ',', '...' ].
    tfidf.loc[:,['0','1', '4', '6', '38917']] = 0

    # Turn GPU Tensor input_ids to CPU DataFrame of strings, so the input_id values could be matched with tfidf column names (which are the same as input_ids).
    input_ids = pd.DataFrame(input_ids.cpu().detach().numpy(), dtype=str)
    
    for i in range(input_ids.shape[0]): # Iterate over the batch

        # From the tfidf matrix, obtain the values from row i and the colums that matches the corresponding documents input_ids.
        # Column names of the tfidf matrix are the same with the all input_ids, we just select the ones that match the current documents input_ids.
        tf_idf_values_of_input = tfidf.loc[i, input_ids.loc[i]]
 
        tfidf_map = pd.DataFrame(data={'tfidf_values':tf_idf_values_of_input.values, 'input_ids':tf_idf_values_of_input.index})

        unique_tfidf_map = tfidf_map.drop_duplicates(subset=['input_ids'])

        top_tfidf_idxs = unique_tfidf_map.sort_values(by='tfidf_values')[-511:].index

        # Uncomment below code for a detailed view of the selected input_ids.
        print('sorted scores: ', np.sort(tf_idf_values_of_input,)[-511:])
        #print('argsort idxs: ',top_tfidf_idxs)
        print('argsort values: ',tf_idf_values_of_input[top_tfidf_idxs])
        print('input ids: ', list(input_ids.loc[i][top_tfidf_idxs]))

        # Print the label description of the document along with the determinded global tokens.
        print(label2description[id2label[labels[i].item()]],':')
        print(tokenizer.convert_ids_to_tokens(list(input_ids.loc[i][top_tfidf_idxs])))
        print('\n################\n')

def map_unique_tfidf(tfidf, f_names, input_ids, device):
    try:
        # Convert the tfidf scores as DataFrame for the current batch.
        tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

        # Assingn zero to some specific columns of tfidf matrix. These are the column names for the following tokens: ['[PAD]', '.', ',', '...' ].
        tfidf.loc[:,['0','1', '4', '6', '38917']] = 0

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

def get_attention_mask(tfidf, f_names, input_ids, device):
    try:
        # Convert the tfidf scores as DataFrame for the current batch.
        tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

        # Assingn zero to some specific columns of tfidf matrix. These are the column names for the following tokens: ['[PAD]', '.', ',', '...' ].
        tfidf.loc[:,['0','1', '4', '6', '38917']] = 0

        # Turn GPU Tensor input_ids to CPU DataFrame of strings, so the input_id values could be matched with tfidf column names (which are the same as input_ids).
        input_ids = input_ids.cpu().detach().numpy()
        input_ids = pd.DataFrame(input_ids, dtype=str)

        # Initilize an empty attention_mask with the same size as the input.
        attention_mask = zeros(input_ids.shape, dtype=long, device=device)
        global_attention_mask = zeros(input_ids.shape, dtype=long, device=device)

        for i in range(input_ids.shape[0]): # For every document(item) in the batch.
            token_count = input_ids.loc[i][input_ids.loc[i] != '1'].count()
            # From the tfidf matrix, obtain the values from row i and the colums that matches the corresponding documents input_ids.
            # Column names of the tfidf matrix are the same with the all input_ids, we just select the ones that match the current documents input_ids.
            tf_idf_values_of_input = tfidf.loc[i, input_ids.loc[i]]
            
            # Cast the tfidf values to numpy array and obtain the indices of a number of highest tfidf scores.
            # These indices correspond the indices of 'to be globally connected tokens' for the model forward.
            tf_idf_values_of_input = np.array(tf_idf_values_of_input, dtype=np.float32)
            
            top_tfidf_idxs = np.argsort(tf_idf_values_of_input)[-2047:]
            global_top_tfidf_idxs = np.argsort(tf_idf_values_of_input)[-1023:]

            # Assign the value one to the positions of determined global tokens.
            attention_mask[i][top_tfidf_idxs] = 1
            global_attention_mask[i][global_top_tfidf_idxs] = 1

            # Set padding tokens to 0
            attention_mask[i][-4096+token_count:]=0
            #print(attention_mask.shape, attention_mask)
            # Print the label description of the document along with the determinded global tokens.
            '''print(label2description[id2label[labels[i].item()]],':', token_count)
            print(tokenizer.convert_ids_to_tokens(list(input_ids.loc[i][global_top_tfidf_idxs])))
            print('****************************')

            print(tokenizer.convert_ids_to_tokens(list(input_ids.loc[i][top_tfidf_idxs])))
            print('\n################\n')'''

        # Assign one to the first token of every document since it is the classification ([CLS]) token and should always be globally connected.
        attention_mask[:, 0] = 1
        global_attention_mask[:, 0] = 1
        #print(attention_mask)
        for mask in attention_mask:
            print('Count: ', mask.count_nonzero())
        return attention_mask, global_attention_mask

    except Exception as e:
        print('Error: ',e)
        global_attention_map = zeros(input_ids.shape, dtype=long, device=device)
        global_attention_map[:, 0] = 1  
        return global_attention_map, global_attention_mask

def visualize_attention_map(map):
    global_list = np.array(map.cpu())
    global_list = np.trim_zeros(global_list, 'b')
    token_len = len(global_list)

    s_map = np.zeros((token_len, token_len))
    
    for i in range(token_len):
        if global_list[i] == 1:
            s_map[i] = 1
            s_map[:, i] = 1

    plt.imshow(s_map, cmap='Greys')
    plt.show()
    #save_dir = os.path.join(config.root_dir, 'longformer_global_attention_maps/')
    #plt.savefig(save_dir+str(step_id)+'.jpg')
    plt.close()

def visualize_pmi_attention_map(pmi_scores, input_ids, labels):

    input_ids = input_ids[0]
    print(input_ids)
    token_len = len(input_ids)
    s_map = np.zeros((token_len, token_len))
    for i in range(token_len):
        for k in range(token_len):
            if (str(input_ids[i]), str(input_ids[k])) in pmi_scores.keys():
                #s_map[i,k] = pmi_scores[(str(input_ids[i]), str(input_ids[k]))]
                s_map[i,k] = 1
    
    plt.imshow(s_map, cmap='Greys', interpolation='nearest', aspect='auto' )
    plt.show()
    #save_dir = os.path.join(config.root_dir, 'longformer_global_attention_maps/')
    #plt.savefig(save_dir+str(step_id)+'.jpg')
    plt.close()

def unique_pmi_qual_analysis(pmi_scores, input_ids, labels):
    window = 10 # Window where the bigram permutations will be calculated. 
    idlist = list(range(0,4096))

    input_ids = input_ids.cpu().detach().numpy()
    for i in range(len(input_ids)): # Iterate over the batch.

        # Define an empty pmi map to potentially contain pmi score of each token for every other token. But since we use a window, the map will not be full.
        pmi_map = np.zeros((4096, 4096))

        for k in range(1, 4096-window): # Iterate over window steps.
            for bigram in permutations(idlist[k:k+window], 2): # Iterate over permutation of token indexes within the current window.
                
                # Obtain the corresponding input_id from the current bigram of token indexes.
                x = input_ids[i][bigram[0]] 
                y = input_ids[i][bigram[1]]

                # If bigram exist among pmi_scores, assign it to its corresponding position in pmi_map. Representing the pmi score of token x and y.
                if (x,y) in pmi_scores.keys():
                    pmi_map[bigram[0]][bigram[1]] = pmi_scores[(x,y)]

        # For every token, sum its pmi score with every other token to get a representation of that token with relate to the whole document.
        token_scores = pmi_map.sum(axis=1)

        # Make pmi scores into a DataFrame to be able to drop duplicates on input_ids and still keep the track of pmi_scores and indexes.
        pmi_map = pd.DataFrame(data={'pmi_values':token_scores, 'input_ids':input_ids[i]})
        unique_pmi_map = pmi_map.drop_duplicates(subset=['input_ids'])

        # Sort the scores and obtain the index of the highest 128.
        top_pmi_idxs = unique_pmi_map.sort_values(by='pmi_values')[-128:].index

        # Uncomment below code for a detailed view of the selected input_ids.
        '''print('sorted scores: ', np.sort(token_scores))
        print('argsort idxs: ',top_pmi_idxs)
        print('argsort values: ',token_scores[top_pmi_idxs])
        print('input ids: ', list(input_ids[i][top_pmi_idxs]))'''


        # Print the label description of the document along with the determinded global tokens.
        print(label2description[id2label[labels[i].item()]])  
        print(tokenizer.convert_ids_to_tokens(input_ids[i][top_pmi_idxs]))
        print('\n################\n')

def pmi_qual_analysis(pmi_scores, input_ids, labels):
    window = 10 # Window where the bigram permutations will be calculated. 
    idlist = list(range(0,4096))

    input_ids = input_ids.cpu().detach().numpy()
    for i in range(len(input_ids)): # Iterate over the batch.

        # Define an empty pmi map to potentially contain pmi score of each token for every other token. But since we use a window, the map will not be full.
        pmi_map = np.zeros((4096, 4096))

        for k in range(1, 4096-window): # Iterate over window steps.
            for bigram in permutations(idlist[k:k+window], 2): # Iterate over permutation of token indexes within the current window.
                
                # Obtain the corresponding input_id from the current bigram of token indexes.
                x = input_ids[i][bigram[0]] 
                y = input_ids[i][bigram[1]]

                # If bigram exist among pmi_scores, assign it to its corresponding position in pmi_map. Representing the pmi score of token x and y.
                if (x,y) in pmi_scores.keys():
                    pmi_map[bigram[0]][bigram[1]] = pmi_scores[(x,y)]

        # For every token, sum its pmi score with every other token to get a representation of that token with relate to the whole document.
        token_scores = pmi_map.sum(axis=1)

        pmi_map = pd.DataFrame(data={'pmi_values':token_scores})

        # Sort the scores and obtain the index of the highest 128.
        top_pmi_idxs = pmi_map.sort_values(by='pmi_values')[-128:].index

        # Uncomment below code for a detailed view of the selected input_ids.
        '''print('sorted scores: ', np.sort(token_scores))
        print('argsort idxs: ',top_pmi_idxs)
        print('argsort values: ',token_scores[top_pmi_idxs])
        print('input ids: ', list(input_ids[i][top_pmi_idxs]))'''


        # Print the label description of the document along with the determinded global tokens.
        print(label2description[id2label[labels[i].item()]])  
        print(tokenizer.convert_ids_to_tokens(input_ids[i][top_pmi_idxs]))
        print('\n################\n')

def map_unique_pmi(pmi_scores, input_ids, device):
    window = 10 # Window where the bigram permutations will be calculated. 
    idlist = list(range(0,4096))

    # Initilize an empty global_attention_map with the same size as the input.
    global_attention_map = zeros(input_ids.shape, dtype=long, device=device)

    input_ids = input_ids.cpu().detach().numpy()
    for i in range(len(input_ids)): # Iterate over the batch.

        # Define an empty pmi map to potentially contain pmi score of each token for every other token. But since we use a window, the map will not be full.
        pmi_map = np.zeros((4096, 4096))

        for k in range(1, 4096-window): # Iterate over window steps.
            for bigram in permutations(idlist[k:k+window], 2): # Iterate over permutation of token indexes within the current window.
                
                # Obtain the corresponding input_id from the current bigram of token indexes.
                x = input_ids[i][bigram[0]] 
                y = input_ids[i][bigram[1]]

                # If bigram exist among pmi_scores, assign it to its corresponding position in pmi_map. Representing the pmi score of token x and y.
                if (x,y) in pmi_scores.keys():
                    pmi_map[bigram[0]][bigram[1]] = pmi_scores[(x,y)]

        # For every token, sum its pmi score with every other token to get a representation of that token with relate to the whole document.
        token_scores = pmi_map.sum(axis=1)

        # Make pmi scores into a DataFrame to be able to drop duplicates on input_ids and still keep the track of pmi_scores and indexes.
        pmi_map = pd.DataFrame(data={'pmi_values':token_scores, 'input_ids':input_ids[i]})
        unique_pmi_map = pmi_map.drop_duplicates(subset=['input_ids'])

        # Sort the scores and obtain the index of the highest 128.
        top_pmi_idxs = unique_pmi_map.sort_values(by='pmi_values')[-128:].index

        # Assign the value one to the positions of determined global tokens.
        global_attention_map[i][top_pmi_idxs] = 1
        
    # Assign one to the first token of every document since it is the classification ([CLS]) token and should always be globally connected.
    global_attention_map[:, 0] = 1  
    return global_attention_map