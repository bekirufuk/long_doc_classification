import os, sys, glob
import pandas as pd
import tokenizers
from tqdm import tqdm
import pickle

from datasets import Dataset, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer

data_name = 'refined_patents'
tokenizer = 'longformer'
partition = 'train'

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

    df = pd.DataFrame(columns=['text', 'labels'])
    df['text'] = [' '.join([str(word) for word in doc]) for doc in tokenized_data['input_ids']]
    df['labels'] = tokenized_data['labels']
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

        Parameters:
                         data: pandas Series object to tfidf be calculated on.
        Returns:
            -sklearn TfidfVectorizer spare matrix: tfidf_vectors
            -sklearn ndarray of str objects      : feature_list
    '''

    tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b") #Token pattern is changed so it would read single digits also.
    
    print('Fitting the tfidf vectorizer...')
    tfidf_vectors = tfidf_vectorizer.fit_transform(tqdm(data))
    feature_list  = tfidf_vectorizer.get_feature_names_out()
    return tfidf_vectors, feature_list

def save_tfidf(tfidf_vectors, feature_list, label_name=None):
    '''
        Saves given tfidf_vectors with their features list as a pickle file. 
    '''
    print('Saving pickle files')

    if label_name is not None:
        save_dir = 'data/'+data_name+'/tfidf/label_based/'+str(label_name)+'_'+partition
    else:
        save_dir = 'data/'+data_name+'/tfidf/'+partition

    pickle.dump(tfidf_vectors, open(save_dir+'_tfidf.pkl', 'wb'))
    pickle.dump(feature_list, open(save_dir+'_feature_names.pkl', 'wb'))

def get_class_based_tfidf(df):
    '''
        Groups the data by their labels.
        Calls "create_tfidf()" function to calculate tfidf scores for every group in the given data.
        Calculation of tfidf scores will only take account word counts within the given label group but not the whole dataset.
        Saves every label groups tfidf scores separately
    '''
    groups = df.groupby('labels')
    for label, group in tqdm(groups):
        tfidf_vectors, feature_list = create_tfidf(group['text'])
        print(tfidf_vectors)
        save_tfidf(tfidf_vectors, feature_list, label_name=label)


if __name__ == '__main__':
    
    df = read_tokenized_data()

    # Create and save tfidf for whole corpus
    tfidf_vectors, feature_list = create_tfidf(df['text'])
    save_tfidf(tfidf_vectors, feature_list)

    # Create and save lael based tfidf for local label corpora
    get_class_based_tfidf(df)