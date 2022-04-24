import os, glob
import datetime
import json
from tkinter.ttk import Progressbar
import config
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from datasets import Dataset, load_from_disk
from transformers import LongformerTokenizer, BigBirdTokenizer, BertTokenizer, LongformerTokenizerFast

# Merges Detailed ddescription files of patents with therir class info. Then chunks them into .csv files.
class Manager():
    def __init__(self, data_name='patentsview', chunksize=20000, engine='c', testing=False, nrows=100, skiprows=0):

        self.data_name = data_name
        self.data_dir = os.path.expanduser('data/'+self.data_name)

        self.engine = engine
        self.chunksize = chunksize
        self.skiprows = skiprows

        self.ipcr_columns = ['patent_id','section','ipc_class', 'subclass']
        self.ipcr_dtypes = {'patent_id':object, 'section':object, 'ipc_class':object, 'subclass':object}

        self.patent_columns = ['patent_id', 'text']
        self.patent_dtypes = {'patent_id':object, 'text':object}

        self.nrows = nrows
        self.testing = testing

        self.saved_record_count = 0
        self.log_dir = os.path.join(self.data_dir, 'meta/merger_logs.txt')
        
        with open(self.log_dir, 'a') as f:
            f.write('#### Merge Operation Initiated #### chunk size:{}, start date:{} \n'.format(self.chunksize,
                                                                                    datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")))

    def load_ipcr(self):
        # Icpr file holds detailed class information about the patents.
        # We will only investigate section column which consist of 8 distinct classes.
        # But ipc_class and sublass values are also obtained for possible future use.

        print("Ipcr data loading...")
        file_dir = os.path.join(self.data_dir,'ipcr.tsv')
        ipcr = pd.read_csv(file_dir,
            sep="\t",
            usecols=self.ipcr_columns,
            dtype=self.ipcr_dtypes,
            engine=self.engine,
            )
        ipcr.drop_duplicates(subset=['patent_id'], inplace=True, keep='first')
        print("Ipcr data loaded.")
        return ipcr

    def get_patents_list(self):
        # Returns raw patent .tsv files belonging seperate years download from patentsview.
        return glob.glob(os.path.join(self.data_dir, "detail*.tsv"))

    def get_chunks(self, patent_dir):
        print("\n\n ########## Processing {} ##########".format(os.path.basename(patent_dir)))
        if self.testing:
            return pd.read_csv(patent_dir,
                sep="\t",
                usecols=self.patent_columns,
                dtype=self.patent_dtypes,
                engine=self.engine,
                chunksize=self.chunksize,
                skiprows=self.skiprows,
                nrows=self.nrows, # Get only a certain number of records for testing.
                encoding='utf8',
                )
        else:
            return pd.read_csv(patent_dir,
                sep="\t",
                usecols=self.patent_columns,
                dtype=self.patent_dtypes,
                engine=self.engine,
                chunksize=self.chunksize,
                encoding='utf8',
                )

    def merge_chunk(self, chunk, ipcr, file_count, chunk_count):
        # Combine patent with respective class info.
        
        print("================ FILE {} CHUNK {} ============ Total Records:{}".format(file_count, chunk_count, self.saved_record_count))

        chunk = chunk.merge(ipcr, how='left', on='patent_id')
        chunk = chunk[chunk['section'].isin(config.labels_list)]
        chunk.replace({'section':config.label2id}, inplace=True)
        chunk['section'].astype(int)
        chunk.rename(columns = {'section':'label'}, inplace = True)
        return chunk

    def write_chunk(self, chunk, patent_year, chunk_counter):
        file_name = 'chunks/patents_'+patent_year+'_chunk_'+str(chunk_counter).zfill(6)+'.csv'
        chunk.to_csv(os.path.join(self.data_dir, file_name),
            sep=',',
            mode='w',
            index=False,
        )
        chunk_len = chunk.shape[0]
        self.saved_record_count += chunk_len
        self.chunk_log(file_name, str(chunk_len), chunk_counter)

    def get_patent_year(self, patent_dir):
        return os.path.basename(patent_dir)[17:21]
    
    def chunk_log(self, file_name, chunk_len, chunk_counter):
        log_text = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        log_text += "-> Chunk {} saved as {} containing {} records - ".format(chunk_counter, file_name, chunk_len)
        log_text += "Running record count for this merge: {}".format(self.saved_record_count)
        with open(self.log_dir, 'a') as f:
            f.write(log_text + '\n')

    def finish(self):

        with open(self.log_dir, 'a') as f:
            f.write('*** Merge Operation Finished #### chunk size:{}, end date:{}, total recorded:{} \n\n'.format(self.chunksize,
                                                                                                        datetime.datetime.now().strftime("%Y-%m-%d-%H:%M"),
                                                                                                        self.saved_record_count
                                                                                                        ))

    def read_chunk_files(self):
        return glob.glob(os.path.join(self.data_dir, "chunks/*.csv"))

    def batch_tokenizer(self, batch, tokenizer):
        return tokenizer(batch["text"])

    def tokenize(self, dataset, tokenizer):
        return dataset.map(self.batch_tokenizer, batched=True, remove_columns=['patent_id','text','label','ipc_class','subclass'], fn_kwargs= {"tokenizer":tokenizer})

    def special_sampling(self, chunk):
        word_counts = chunk['text'].apply(nltk.word_tokenize).apply(lambda x: len(x))
        chunk = chunk[(word_counts>=512) & (word_counts<=4094)].reset_index(drop=True)
        groups = chunk.groupby('label')
        chunk = groups.apply(lambda x: x.sample(groups.size().min()).reset_index(drop=True))
        return chunk

    def load_tokenized_data(self):
        return load_from_disk(os.path.join(self.data_dir, "tokenized/"))

    def read_chunks(self, limit=None):
        print('Reading chunk files...')
        files = self.read_chunk_files()
        if limit is not None:
            files = files[:limit]
        df = pd.DataFrame()
        prog_bar = tqdm(files)
        for file in files:
            chunk = pd.read_csv(file)
            df = pd.concat([df,chunk], ignore_index=True)
            prog_bar.update(1)
        return df

    def create_word_freqs(self):
        df = self.read_chunks()
        words_list = self.get_words_list()
        count_vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize,
                                            strip_accents='unicode',
                                            vocabulary=words_list,
                                            )
        print('Fitting the count vectorizer...')
        count_vectors= count_vectorizer.fit_transform(tqdm(df['text']))

        words_list = count_vectorizer.get_feature_names_out()
        counts_list = np.asarray(count_vectors.sum(axis=0))[0]

        word_freqs = dict(zip(words_list, counts_list))

        #Remove zero frequent words
        word_freqs = {word:count for word,count in word_freqs.items() if count!=0}
        print('Fitting done. Saving the word frequencies as pickle.')

        save_dir = os.path.join(self.data_dir, 'meta')
        pickle.dump(word_freqs, open(os.path.join(save_dir,'word_freqs.pkl'), "wb"))

    def read_word_freqs_file(self):
        word_freqs = os.path.join(self.data_dir,'meta/word_freqs.pkl')
        return pd.read_pickle(word_freqs)

    def get_word_freqs(self, remove_stopwords=False):
        word_freqs = self.read_word_freqs_file()
        if remove_stopwords:
            stopwords = nltk.corpus.stopwords.words('english')
            word_freqs = {word:count for word,count in word_freqs.items() if word not in stopwords}
        word_freqs = sorted(word_freqs.items(), key=lambda x: x[1])
        print('\nTotal {} words in the vocab.'.format(len(word_freqs)))
        print('\nMost frequent 20 words are:\n{}'.format(word_freqs[-20:]))
        print('\nLeast frequent 20 words are:\n{}'.format(word_freqs[:20]))

    def create_tfidf(self):
        # tokenizer = nltk.word_tokenize
        tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=config.max_length)
        df = self.read_chunks()
        words_list = self.get_words_list()
        tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer,
                                            strip_accents='unicode',
                                            vocabulary=words_list,
                                            )
        print('Fitting the tfidf vectorizer...')
        tfidf_vectors= tfidf_vectorizer.fit_transform(tqdm(df['text']))
        save_dir = os.path.join(self.data_dir, 'meta')
        pickle.dump(tfidf_vectors, open(os.path.join(save_dir,'longformer_tokens_tfidf_sparse.pkl'), "wb"))
        pickle.dump(tfidf_vectorizer.get_feature_names_out(), open(os.path.join(save_dir,'longformer_tokens_tfidf_feature_names.pkl'), "wb"))

    def create_term_doc_matrix(self):
        df = self.read_chunks()
        words_list = self.get_words_list()
        count_vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize,
                                            strip_accents='unicode',
                                            vocabulary=words_list,
                                            )
        print('Fitting the count vectorizer...')
        count_vectors= count_vectorizer.fit_transform(tqdm(df['text']))
        save_dir = os.path.join(self.data_dir, 'meta')
        pickle.dump(count_vectors, open(os.path.join(save_dir,'term_doc_matrix_sparse.pkl'), "wb"))
        pickle.dump(count_vectorizer.get_feature_names_out(), open(os.path.join(save_dir,'term_doc_matrix_feature_names.pkl'), "wb"))

    def create_words_list(self):
        save_dir = os.path.join(self.data_dir, 'meta')
        print('Deduplicating the vocab...')
        words_list = list(map(lambda x: x.lower(), nltk.corpus.words.words()))
        ulist = []
        [ulist.append(x) for x in words_list if x not in ulist]
        pickle.dump(ulist, open(os.path.join(save_dir,'words_list.pkl'), "wb"))

    def get_words_list(self):
        return pd.read_pickle(os.path.join(self.data_dir,'meta/words_list.pkl'))

    def create_doc_stats(self):
        file_dir_doc_term = os.path.join(manager.data_dir, 'meta/term_doc_matrix_sparse.pkl')       
        doc_term = pd.read_pickle(file_dir_doc_term)

        df = self.read_chunks()

        for i in tqdm(range(doc_term.shape[0])):
            doc = np.array(doc_term[i].toarray()[0])
            df.loc[i, ['word_count', 'u_word_count']] = [doc[doc!=0].sum(), len(doc[doc!=0])]
        
        save_dir = os.path.join(self.data_dir, 'meta/doc_word_stats.csv')
        df.to_csv(save_dir, index=None)

def merge(manager):
    print("\n----------\n DATA MERGE STARTED \n----------\n")
    ipcr = manager.load_ipcr()
    patent_file_list = manager.get_patents_list()
    for patent_file_counter, patent_dir in enumerate(patent_file_list):
        patent_year = manager.get_patent_year(patent_dir)
        chunks = manager.get_chunks(patent_dir)
        chunk_counter = 0
        for chunk in chunks:
            chunk_counter += 1
            chunk = manager.merge_chunk(chunk, ipcr, patent_file_counter+1, chunk_counter)
            chunk = manager.special_sampling(chunk)
            manager.write_chunk(chunk, patent_year, chunk_counter)
    manager.finish()
    print("\n----------\n DATA MERGE FINISHED \n----------\n")

def tokenizer_comparison():
    tokenizer_longformer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', max_length=config.max_length)
    tokenizer_bigbird = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

    file = os.path.join(config.data_dir, 'chunks\patents_2020_chunk_000008.csv')
    df = pd.read_csv(file)

    print('\nLongformer: \n', tokenizer_longformer.tokenize(df['text'][0][:300]))
    print('\nBigbird: \n', tokenizer_bigbird.tokenize(df['text'][0][:300]))
    print('\nBERT: \n', tokenizer_bert.tokenize(df['text'][0][:300]) )
    print('\nNLTK: \n', nltk.word_tokenize(df['text'][0][:300]))

def fast_comparison():
    fast = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=config.max_length)
    standard = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', max_length=config.max_length)

    file = os.path.join(config.data_dir, 'chunks\patents_2020_chunk_000008.csv')
    df = pd.read_csv(file)

    print('\nLongformer Fast: \n', fast.tokenize(df['text'][0][:300]))
    print('\nLongformer Standard: \n', standard.tokenize(df['text'][0][:300]))

def token_length_difference_checker():
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    files = glob.glob('data/patentsview/chunks/*csv')
    for file in files[:3]:
        df = pd.read_csv(file)
        splitted = df['text'].apply(lambda x: len(x.split()))
        bert_tokenized = df['text'].apply(lambda x: len( tokenizer_bert.tokenize(x)))
        print('\n Split: ', sum(splitted)/len(splitted))
        print('\n BERT: ', sum(bert_tokenized)/len(bert_tokenized))

def print_word_stats():
    chunks = manager.read_chunks(2)
    chunks['text'] = [chunk.lower() for chunk in chunks['text']]
    df = pd.DataFrame(columns=['word_count', 'u_word_count'])

    for i in tqdm(range(chunks.shape[0])):
        words = nltk.word_tokenize(chunks.loc[i,'text'])
        u_words = []
        [u_words.append(word) for word in words if word not in u_words ]
        df.loc[i] = [len(words), len(u_words)]
    save_dir = os.path.join(manager.data_dir, 'meta/doc_word_stats.csv')
    df.to_csv(save_dir, index=None)
    print('Total Words:')
    print('Min:{}, Max:{} Avg:{}'.format(df['word_count'].min(), df['word_count'].max(), df['word_count'].mean()))

    print('\nUnique Words:')
    print('Min:{}, Max:{} Avg:{}'.format(df['u_word_count'].min(), df['u_word_count'].max(), df['u_word_count'].mean()))

if __name__ == '__main__':
    manager = Manager()
    manager.create_tfidf()

