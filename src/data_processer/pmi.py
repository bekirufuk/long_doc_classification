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

def get_pmi(w=2):
    print('Defining Tokenizer...')
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=4096)

    print('Loading Data...')
    tokenized_data = load_from_disk('data/refined_patents/tokenized/longformer_tokenizer_no_padding/train')

    print('Data Processing...')

    # no padding train verisi ve w=4096 için 140 saat
    doc_tokens_str = [str(item) for sublist in tqdm(np.array(tokenized_data['input_ids'])) for item in sublist]

    print('Creating Bigrams...')
    bigram_measures = BigramAssocMeasures()
    bigrams = BigramCollocationFinder.from_words(doc_tokens_str, window_size=w)

    print('Word Filtering...')
    bigrams.apply_word_filter(lambda w: w in ('0', '1'))

    print('Creating PMI Scores...')
    scored = bigrams.score_ngrams(bigram_measures.pmi)

    print(len(scored))
    print('Converting to Dict...')

    pmi_dict = {}
    for i in tqdm(range(len(scored))):
        pmi_dict[scored[i][0]] = scored[i][1]
        print(tokenizer.convert_ids_to_tokens([int(scored[i][0][0]), int(scored[i][0][1])]), scored[i][1])

    print('Saving PMI Dict...')
    save_dir = 'data/refined_patents/stats/'
    pickle.dump(pmi_dict, open(save_dir+'bigram_pmi_scores_w'+str(w)+'.pkl', 'wb'))

def check_pmi_scores(w):
    print('Reading Scores...')
    pmi_scores = pd.read_pickle('data/refined_patents/stats/bigram_pmi_scores_w'+str(w)+'.pkl')
    print('Total Bigram Count: ', len(pmi_scores))
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=4096)

    for i, key in enumerate(pmi_scores.keys()):
        print(tokenizer.convert_ids_to_tokens([int(key[0]), int(key[1])]), pmi_scores[key])
        if(i == 200):
            break

def convert_bigrams_to_int(pmi_scores):
    pmi_scores_int = dict()
    for key in tqdm(pmi_scores.keys()):
        int_tuple = tuple((int(key[0]), int(key[1])))
        pmi_scores_int[int_tuple] = pmi_scores[key]


check_pmi_scores(w=10)
