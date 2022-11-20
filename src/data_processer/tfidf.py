import sys, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from scipy.sparse import csr_matrix

from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer

data_name = 'refined_patents'
tokenizer = 'bert_trained_on_patent_data'
partition = ''

def read_tokenized_data(limit=None):
    ''' 
        Loads tokenized data from a folder that is specified with global data_name and tokenizer variables.
        Loads only the specified partition in the global parameter.
        Converts the stored list of list of tokens data to list of strings. (turns list of tokens into a single string which token_ids seperated by a space)

        Parameters: 
            (global, str)data_name: name of the data folder.
            (global, str)tokenizer: name of the tokenizer to be used.
            (global, str)partition: name of the partition to be returned. Options: ['train', 'validation', 'test']

        Returns: pandas DataFrame
    '''
    print('Loading tokenized data...')
    tokenized_data = load_from_disk("data/refined_patents/tokenized/"+tokenizer)[partition]
    if limit:
        tokenized_data = tokenized_data.select(range(limit))
    tokenized_data = tokenized_data.remove_columns(['ipc_class', 'subclass'])
    print('Tokenized data loaded.')

    df = pd.DataFrame(data={'patent_id':tokenized_data['patent_id'],
                            'input_ids':tokenized_data['input_ids'],
                            'labels':tokenized_data['labels']
                            })

    print('Concat tokens as a single string...')
    df['text'] = [' '.join([str(word) for word in doc]) for doc in tqdm(np.array(df['input_ids']))]
    
    return df

def create_tfidf(data):
    '''
        Creates a tfidf matrix and feature list for a given pandas series that consist of strings of words.
        Converts the sparse tfif into pandas DataFrame.

        Parameters:
                         data: pandas Series object to tfidf be calculated on.
        Returns:
            -pandas DataFrame
    '''
    print('Creating the tfidf vectorizer...')
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b") #Token pattern is changed so it would read single digits also.
    
    print('Fitting the tfidf vectorizer...')
    tfidf_vectors = tfidf_vectorizer.fit_transform(tqdm(data['text']))
    feature_list  = tfidf_vectorizer.get_feature_names_out()

    return tfidf_vectors, feature_list

def save_tfidf(tfidf_vectors, feature_list, label_based=False, label=None):
    '''
        Saves given tfidf_vectors with their features list as a pickle file. 
    '''
    print('Saving pickle files...')

    if label_based:
        save_dir = 'data/'+data_name+'/tfidf/'+tokenizer+'/label_based/'+partition+'_'+str(label)
    else:
        save_dir = 'data/'+data_name+'/tfidf/'+tokenizer+'/'+partition

    pickle.dump(tfidf_vectors, open(save_dir+'_tfidf_sparse.pkl', 'wb'))
    pickle.dump(feature_list, open(save_dir+'_f_list.pkl', 'wb'))

def get_class_based_tfidf(df):
    '''
        Groups the data by their labels.
        Calls "create_tfidf()" function to calculate tfidf scores for every group in the given data.
        Calculation of tfidf scores will only take account word counts within the given label group but not the whole dataset.
        Saves every label groups tfidf scores separately
    '''
    groups = df.groupby('labels')

    tfidf = pd.DataFrame()
    for label, group in tqdm(groups):
        
        tfidf_vectors, feature_list = create_tfidf(group)
        label_based_tfidf = pd.DataFrame(tfidf_vectors.toarray(), columns=feature_list)
        label_based_tfidf['general_index'] = list(group.index)
        label_based_tfidf['patent_id'] = list(group.patent_id)
        
        tfidf = pd.concat([tfidf, label_based_tfidf], ignore_index=True)
    
    del df, label_based_tfidf
    
    tfidf = tfidf.set_index('general_index')
    tfidf = tfidf.sort_index()
    tfidf = tfidf.fillna(0)
    tfidf.index.name = None

    tfidf_sparse = csr_matrix(tfidf.values)

    save_dir = 'data/refined_patents/tfidf/longformer_tokenizer_no_stopwords/label_based/'+partition
    
    pickle.dump(tfidf_sparse, open(save_dir+'_tfidf_sparse.pkl', 'wb'))
    pickle.dump(tfidf.columns, open(save_dir+'_f_list.pkl', 'wb'))

if __name__ == '__main__':

    for p in ['train', 'test']:
        partition = p
        print('Operations for {} \n'.format(partition))
        
        #limit = 20000 if partition == 'train' else 3200
        df = read_tokenized_data(limit=20)

        # Create and save tfidf for whole corpus
        tfidf_vectors, feature_list = create_tfidf(df)
        save_tfidf(tfidf_vectors, feature_list)

        # Create and save label based tfidf for local label corpora
        #get_class_based_tfidf(df)