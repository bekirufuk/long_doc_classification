from linecache import cache
import pandas as pd
import numpy as np
import sys, os
from tqdm import tqdm
import pickle

from nltk import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures

from datasets import load_from_disk
from transformers import LongformerTokenizerFast

sys.path.append(os.getcwd())
from src.data_processer.process import get_longformer_tokens

def get_pmi(w=10):

    print('Loading Data...')
    tokenized_data = load_from_disk('data/refined_patents/tokenized/big_bird_tokenizer_no_padding/train')

    print('Data Processing...')

    # 140 hrs of expected computation time for no padding train data with w=4096
    doc_tokens_str = [str(item) for sublist in tqdm(np.array(tokenized_data['input_ids'])) for item in sublist]

    print('Creating Bigrams...')
    bigram_measures = BigramAssocMeasures()
    bigrams = BigramCollocationFinder.from_words(doc_tokens_str, window_size=w)

    print('Word Filtering...')
    bigrams.apply_word_filter(lambda w: w in ('0', '1', '2'))

    print('Creating PMI Scores...')
    scored = bigrams.score_ngrams(bigram_measures.pmi)

    print('first save...')
    save_dir = 'data/refined_patents/stats/'
    pickle.dump(scored, open(save_dir+'big_bird_bigram_pmi_scores_w'+str(w)+'_non_dict.pkl', 'wb'))

    print('Converting to Dict...')
    pmi_dict = {}
    for i in tqdm(range(len(scored))):
        pmi_dict[tuple(int(item) for item in scored[i][0])] = scored[i][1]

    print('Saving PMI Dict...')
    save_dir = 'data/refined_patents/stats/'
    pickle.dump(pmi_dict, open(save_dir+'big_bird_bigram_pmi_scores_w'+str(w)+'_int.pkl', 'wb'))

def check_pmi_scores(w=10):
    print('Reading Scores...')
    pmi_scores = pd.read_pickle('data/refined_patents/stats/big_bird_bigram_pmi_scores_w'+str(w)+'_int.pkl')
    print('Total Bigram Count: ', len(pmi_scores))
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=4096)

    for i, key in enumerate(pmi_scores.keys()):
        print(tokenizer.convert_ids_to_tokens([int(key[0]), int(key[1])]), pmi_scores[key])
        if(i == 200):
            break

def convert_bigrams_to_int(pmi_scores):
    pmi_scores_int = dict()
    for key in tqdm(pmi_scores.keys()):
        int_tuple = tuple(((int(key[0]), int(key[1]))))
        pmi_scores_int[int_tuple] = pmi_scores[key]

def read_pmi():
    pmi_scores = pd.read_pickle('data/refined_patents/stats/big_bird_bigram_pmi_scores_w10_non_dict.pkl')

    print('Converting to Dict...')
    pmi_dict = {}
    for i in tqdm(range(len(pmi_scores))):
        pmi_dict[tuple(int(item) for item in pmi_scores[i][0])] = pmi_scores[i][1]

    print('Saving PMI Dict...')
    save_dir = 'data/refined_patents/stats/'
    pickle.dump(pmi_dict, open(save_dir+'big_bird_bigram_pmi_scores_w10_int.pkl', 'wb'))


if __name__ == '__main__':
    read_pmi()
