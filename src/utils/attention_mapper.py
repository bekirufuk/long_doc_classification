'''
    This script consist of various functions that takes a batch of 'input_ids' as input and outputs a same size matrix consisting ones and zeros.
    Ones representing the global connection between that token (represented by tokenizers input_id) and all the others for that input document.
    
    Detailed explanation for the usage of global_attentions in custom transformer models can be found at the docstring of
    'LongformerSequenceClassifierOutput' class in the HuggingFace repo: https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py 
'''

import sys
import numpy as np
import pandas as pd
import torch
from torch import zeros, long
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import normalize

# Uncomment below code for 'tfidf_qual_analysis' function.
'''from transformers import AutoTokenizer
tokenizer_name = 'bert_trained_on_patent_data'
tokenizer = AutoTokenizer.from_pretrained("tokenizers/"+tokenizer_name, max_length=4096)
id2label =  {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
label2description= {'A': 'Human Necessities', 'B':'Performing Operations; Transporting', 'C':'Chemistry; Metallurgy','D':'Textiles; Paper',
                    'E':'Fixed Constructions','F':'Mechanical Engineering; Lighting; Heating; Weapons; Blasting','G':'Physics','H':'Electricity'}'''

def tfidf_qual_analysis(tfidf, f_names, input_ids, labels, device):

    # Convert the tfidf scores as DataFrame for the current batch.
    tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

    # Assingn zero to some specific columns of tfidf matrix. These are the column names for the following tokens: ['[PAD]', '.', ',', '...' ].
    
    tfidf[['0', '5', '6', '157', '655', '3456', '2714']] = 0
    
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

        # Uncomment below code for a detailed view of the selected input_ids.
        print('sorted scores: ', np.sort(tf_idf_values_of_input,)[-1023:])
        print('argsort idxs: ',top_tfidf_idxs)
        print('argsort values: ',tf_idf_values_of_input[top_tfidf_idxs])
        print('input ids: ', list(input_ids.loc[i][top_tfidf_idxs]))
        
        # Print the label description of the document along with the determinded global tokens.
        print(label2description[id2label[labels[i].item()]],':')
        print(tokenizer.convert_ids_to_tokens(list(input_ids.loc[i][top_tfidf_idxs])))
        print('\n################\n')
    global_attention_map[:, 0] = 1  
    visualize_attention_map(global_attention_map[0])

def map_tfidf(tfidf, f_names, input_ids, device):
    try:
        # Convert the tfidf scores as DataFrame for the current batch.
        tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

        # Assingn zero to some specific columns of tfidf matrix. These are the column names for the following tokens: ['[PAD]', '.', ',', '...' ].
        tfidf[['0', '5', '6', '157', '655', '3456', '2714']] = 0

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

    window = 20 # Window where the bigram combinations will be calculated. 
    idlist = list(range(0,4096))

    # Initilize an empty global_attention_map with the same size as the input.
    global_attention_map = zeros(input_ids.shape, dtype=long, device=device)

    input_ids = input_ids.cpu().detach().numpy()
    for i in range(len(input_ids)): # Iterate over the batch.

        # Define an empty pmi map to potentially contain pmi score of each token for every other token. But since we use a window, the map will not be full.
        pmi_map = np.zeros((4096, 4096))

        for k in range(1, 4096-window): # Iterate over window steps.
            for bigram in combinations(idlist[k:k+window], 2): # Iterate over combinations of token indexes within the current window.
                
                # Obtain the corresponding input_id from the current bigram of token indexes.
                x = input_ids[i][bigram[0]] 
                y = input_ids[i][bigram[1]]

                # If bigram exist among pmi_scores (we check both cases of (x,y) and (y,x))
                # assign it to its corresponding position in pmi_map. Representing the pmi score of token x and y.
                if (x,y) in pmi_scores.keys():
                    pmi_map[bigram[0]][bigram[1]] = pmi_scores[(x,y)]
                    pmi_map[bigram[1]][bigram[0]] = pmi_scores[(x,y)]
                elif (y,x) in pmi_scores.keys():
                    pmi_map[bigram[0]][bigram[1]] = pmi_scores[(y,x)]
                    pmi_map[bigram[1]][bigram[0]] = pmi_scores[(y,x)]

        # For every token, sum its pmi score with every other token to get a representation of that token with relate to the whole document.
        token_scores = pmi_map.sum(axis=1)

        token_scores = normalize([token_scores], norm='l1')[0]
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

def block_map_pmi(pmi_scores, input_ids, device):

    window = 10 # Window where the bigram permutations will be calculated. 
    idlist = list(range(0,4096))

    pmi_batch_map = np.empty((len(input_ids),12,62,3)) # pmi map to hold scores for the batch in the big bird random block attention format.


    input_ids = input_ids.cpu().detach().numpy()
    for i in range(len(input_ids)): # Iterate over the batch.

        # Define an empty pmi map to potentially contain pmi score of each token for every other token. But since we use a window, the map will not be full.
        pmi_map = np.zeros((4096, 4096))

        # Calculate the token count omitting padding tokens. (We don't need to look for pmi scors of them)
        input_len_without_padding = np.count_nonzero(input_ids[i])

        for k in range(1, input_len_without_padding-window): # Iterate over window steps.
            for bigram in combinations(idlist[k:k+window], 2): # Iterate over combinations of token indexes within the current window.

                # Obtain the corresponding input_id from the current bigram of token indexes.
                x = input_ids[i][bigram[0]] 
                y = input_ids[i][bigram[1]]

                # If bigram exist among pmi_scores (we check both cases of (x,y) and (y,x))
                # assign it to its corresponding position in pmi_map. Representing the pmi score of token x and y.
                if (x,y) in pmi_scores.keys():
                    pmi_map[bigram[0]][bigram[1]] = pmi_scores[(x,y)]
                    pmi_map[bigram[1]][bigram[0]] = pmi_scores[(x,y)]
                elif (y,x) in pmi_scores.keys():
                    pmi_map[bigram[0]][bigram[1]] = pmi_scores[(y,x)]
                    pmi_map[bigram[1]][bigram[0]] = pmi_scores[(y,x)]
        
        # Normalize the pmi_map
        pmi_map = normalize(pmi_map, norm='l1', axis=1)
        
        # Desired shape for random attention in big bird is (batch_size,head_size,block_size-2,random_token_count) being (8,12,62,3) in this case.
        
        # Group the columns in 64 sized of blocks. For 4096, there will be 64 blocks in total.
        pmi_map = pmi_map.reshape(4096,64,64)

        # Sum the normalized values into 64 scores, making up to 1 in total.
        pmi_map = np.sum(pmi_map, axis=2)

        # Now group the rows since the tokens should be grouped into blocks of 64X64.
        pmi_map = pmi_map.reshape(64,64,64)

        # Sum in the columnwise this time, summing up the normalized values of tokens for every other token in a group of 64. Size is 64X64 after the sum.
        pmi_map = np.sum(pmi_map, axis=1)

        # Omit the last and the first blocks.
        pmi_map = pmi_map[1:63,:]

        # Argsort the pmi scores
        pmi_map = np.argsort(pmi_map,axis=1)

        # Pick the top 3 block with the highest pmi score. So, for each block in the query, we have 3 block idxs to calculate the attention of.
        pmi_map = pmi_map[:,-3:]

        # Populate the same map for 12 heads of the big bird. Original implementation since its random have different values for different heads. We use the same values instead.
        pmi_map = np.array([pmi_map for i in range(12)])

        # Add the single input scores to the list of pmi maps for the entire batch.
        pmi_batch_map[i] = pmi_map
    
    return torch.from_numpy(pmi_batch_map).type(torch.long).to(device)

def block_map_tfidf(tfidf, f_names, input_ids, device):

    # tfidf map to hold scores for the batch in the big bird random block attention format.
    tfidf_batch_map = np.empty((len(input_ids),12,62,3))

    # Convert the tfidf scores as DataFrame for the current batch.
    tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

    # Mark unwanted Big Bird token values to zero.
    tfidf[['0', '112', '114', '9333']] = 0

    # Turn GPU Tensor input_ids to CPU DataFrame of strings, so the input_id values could be matched with tfidf column names (which are the same as input_ids).
    input_ids = pd.DataFrame(input_ids.cpu().detach().numpy(), dtype=str)
    
    for i in range(input_ids.shape[0]): # For every document(item) in the batch.

        # From the tfidf matrix, obtain the values from row i and the colums that matches the corresponding documents input_ids.
        # Column names of the tfidf matrix are the same with the all input_ids, we just select the ones that match the current documents input_ids.
        tfidf_values_of_input = tfidf.loc[i, input_ids.loc[i]]
        
        # Cast the tfidf values to numpy array and obtain the indices of a number of highest tfidf scores.
        # These indices correspond the indices of 'to be globally connected tokens' for the model forward.
        tfidf_values_of_input = np.array(tfidf_values_of_input, dtype=np.float32)

        tfidf_map = np.multiply.outer(tfidf_values_of_input, tfidf_values_of_input)
        
        # Normalize the tfidf_map
        tfidf_map = normalize(tfidf_map, norm='l1', axis=1)
        
        # Desired shape for random attention in big bird is (batch_size,head_size,block_size-2,random_token_count) being (8,12,62,3) in this case.
        
        # Group the columns in 64 sized of blocks. For 4096, there will be 64 blocks in total.
        tfidf_map = tfidf_map.reshape(4096,64,64)

        # Sum the normalized values into 64 scores, making up to 1 in total.
        tfidf_map = np.sum(tfidf_map, axis=2)

        # Now group the rows since the tokens should be grouped into blocks of 64X64.
        tfidf_map = tfidf_map.reshape(64,64,64)

        # Sum in the columnwise this time, summing up the normalized values of tokens for every other token in a group of 64. Size is 64X64 after the sum.
        tfidf_map = np.sum(tfidf_map, axis=1)     
        
        # Omit the last and the first blocks.
        tfidf_map = tfidf_map[1:63,:]

        # Argsort the pmi scores
        tfidf_map = np.argsort(tfidf_map,axis=1)

        # Pick the top 3 block with the highest tfidf score. So, for each block in the query, we have 3 block idxs to calculate the attention of.
        tfidf_map = tfidf_map[:,-3:]

        # Populate the same map for 12 heads of the big bird. Original implementation since its random have different values for different heads. We use the same values instead.
        tfidf_map = np.array([tfidf_map for i in range(12)])

        # Add the single input scores to the list of pmi maps for the entire batch.
        tfidf_batch_map[i] = tfidf_map
    
    return torch.from_numpy(tfidf_batch_map).type(torch.long).to(device)

def expanded_unique_tfidf_qual_analysis(tfidf, f_names, input_ids, labels, device):
    # Convert the tfidf scores as DataFrame for the current batch.
    tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

    # Assingn zero to some specific columns of tfidf matrix. These are the column names for the following tokens: ['[PAD]', '.', ',', '...' ].
    tfidf.loc[:,['0','1', '4', '5', '6', '38917']] = 0

    # Turn GPU Tensor input_ids to CPU DataFrame of strings, so the input_id values could be matched with tfidf column names (which are the same as input_ids).
    input_ids = pd.DataFrame(input_ids.cpu().detach().numpy(), dtype=str)
    
    for i in range(input_ids.shape[0]): # Iterate over the batch

        # From the tfidf matrix, obtain the values from row i and the colums that matches the corresponding documents input_ids.
        # Column names of the tfidf matrix are the same with the all input_ids, we just select the ones that match the current documents input_ids.
        tf_idf_values_of_input = tfidf.loc[i, input_ids.loc[i]]
 
        tfidf_map = pd.DataFrame(data={'tfidf_values':tf_idf_values_of_input.values, 'input_ids':tf_idf_values_of_input.index})

        unique_tfidf_map = tfidf_map.drop_duplicates(subset=['input_ids'])

        top_tfidf_idxs = unique_tfidf_map.sort_values(by='tfidf_values')[-102:].index

        expanded_top_tfidf_idxs = np.array([[max(e-2,1), max(e-1,1), e, min(e+1,input_ids.shape[1]), min(e+2,input_ids.shape[1])] for e in top_tfidf_idxs]).flatten()

        # Uncomment below code for a detailed view of the selected input_ids.
        print('sorted scores: ', np.sort(tf_idf_values_of_input,)[-511:])
        #print('argsort idxs: ',top_tfidf_idxs)
        print('argsort values: ',tf_idf_values_of_input[expanded_top_tfidf_idxs])
        print('input ids: ', list(input_ids.loc[i][expanded_top_tfidf_idxs]))

        # Print the label description of the document along with the determinded global tokens.
        print(label2description[id2label[labels[i].item()]],':')
        print(tokenizer.convert_ids_to_tokens(list(input_ids.loc[i][expanded_top_tfidf_idxs])))
        print('\n################\n')

def map_expanded_unique_tfidf(tfidf, f_names, input_ids, device):
    try:
        # Convert the tfidf scores as DataFrame for the current batch.
        tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

        # Assingn zero to some specific columns of tfidf matrix. These are the column names for the following tokens: ['[PAD]', '.', ',', '...' ].
        tfidf[['0','1', '4', '5', '6', '38917']] = 0

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

            top_tfidf_idxs = unique_tfidf_map.sort_values(by='tfidf_values')[-50:].index

            expanded_top_tfidf_idxs = np.array([[max(e-2,1), max(e-1,1), e, min(e+1,input_ids.shape[1]), min(e+2,input_ids.shape[1])] for e in top_tfidf_idxs]).flatten()

            # Assign the value one to the positions of determined global tokens.
            global_attention_map[i][expanded_top_tfidf_idxs] = 1
        
        # Assign one to the first token of every document since it is the classification ([CLS]) token and should always be globally connected.
        global_attention_map[:, 0] = 1  
        return global_attention_map

    except Exception as e:
        print('Error: ',e)
        global_attention_map = zeros(input_ids.shape, dtype=long, device=device)
        global_attention_map[:, 0] = 1  
        return global_attention_map