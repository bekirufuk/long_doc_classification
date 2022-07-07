import sys, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from scipy.sparse import csr_matrix

from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer

data_name = 'refined_patents'
tokenizer = 'longformer'
partition = ''

def read_tokenized_data():
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
    tokenized_data = load_from_disk("data/"+data_name+"/tokenized/"+tokenizer+"_tokenizer/"+partition)
    print('Tokenized data loaded.')

    print('Concat tokens as a single string...')
    df = pd.DataFrame(columns=['patent_id','text', 'labels'])

    df['patent_id'] = tokenized_data['patent_id']
    df['labels'] = tokenized_data['labels']

    tokenized_data.set_format('numpy')
    df['text'] = [' '.join([str(word) for word in doc]) for doc in tqdm(tokenized_data['input_ids'])]
    
    return df

def read_chunks(limit=None):
    '''
        Reads all (if there is no limit) the .csv files inside the chunks/ under the given data_name directory. Combines them into a single dataframe and returns it.

        Parameters:
            (global, str)data_name: name of the data folder.
            (optional, int)limit  : limits to files to be read to a certain number. Default: None. Used for testing purposes.
        
        Returns: pandas DataFrame
    '''

    files = glob.glob("data/"+data_name+"/chunks/*.csv")
    if limit:
        files = files[:limit]

    print('Reading chunk files...')  
    df = pd.DataFrame()
    for file in tqdm(files):
        chunk = pd.read_csv(file)
        df = pd.concat([df,chunk], ignore_index=True)
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
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float32) #Token pattern is changed so it would read single digits also.
    
    print('Fitting the tfidf vectorizer...')
    tfidf_vectors = tfidf_vectorizer.fit_transform(tqdm(data['text']))
    feature_list  = tfidf_vectorizer.get_feature_names_out()

    #tfidf = pd.DataFrame(tfidf_vectors.toarray(), columns=feature_list)
    #tfidf.insert(0, 'patent_id', list(data['patent_id']))
    return tfidf_vectors, feature_list

def save_tfidf(tfidf_vectors, feature_list, label_based=False, label=None):
    '''
        Saves given tfidf_vectors with their features list as a pickle file. 
    '''
    print('Saving pickle files')

    if label_based:
        save_dir = 'data/'+data_name+'/tfidf/longformer_tokenizer_no_stopwords/label_based/'+str(label)+'_'+partition
    else:
        save_dir = 'data/'+data_name+'/tfidf/longformer_tokenizer_no_stopwords/'+partition

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

    for label, group in tqdm(groups):
        
        tfidf_vectors, feature_list = create_tfidf(group)
        tfidf = pd.DataFrame(tfidf_vectors.toarray(), columns=feature_list, dtype=np.float16)
        tfidf['general_index'] = list(group.index)

        save_dir = 'data/'+data_name+'/tfidf/longformer_tokenizer_no_stopwords/label_based/'+partition+'_'+str(label)

        #tfidf.to_csv(save_dir+'_tfidf.csv', index=False)
        pickle.dump(tfidf, open(save_dir+'_tfidf.pkl', 'wb'))


if __name__ == '__main__':

    for p in ['train', 'test', 'validation']:
        partition = p
        print('Operations for {} \n'.format(partition))
        
        df = read_tokenized_data()

        # Create and save tfidf for whole corpus
        tfidf_vectors, feature_list = create_tfidf(df)
        save_tfidf(tfidf_vectors, feature_list)

        # Create and save label based tfidf for local label corpora
        get_class_based_tfidf(df)