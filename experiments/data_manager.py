import os, glob
import pandas as pd
import datetime
import config
import collections
from tqdm.auto import tqdm
from collections import Counter
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

    def special_sampling(self, chunk, tokenizer):
        dataset = Dataset.from_pandas(chunk)
        tokenized_dataset = self.tokenize(dataset, tokenizer)
        token_lengths = pd.Series(tokenized_dataset['input_ids']).apply(lambda x: len(x))
        chunk['token_length'] = token_lengths
        chunk = chunk[(chunk['token_length']>=512) & (chunk['token_length']<=4094)].reset_index(drop=True)
        groups = chunk.groupby('label')
        chunk = groups.apply(lambda x: x.sample(groups.size().min()).reset_index(drop=True))
        return chunk

    def load_tokenized_data(self):
        return load_from_disk(os.path.join(self.data_dir, "tokenized/"))

    def create_vocab(self):
        counter = collections.Counter()
        files = self.read_chunk_files()
        progress_bar = tqdm(range(len(files)))
        tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
        for file in files:
            df = pd.read_csv(file)
            df['text'].apply(lambda x: counter.update(Counter(tokenizer_bert.tokenize(x))))
            progress_bar.update(1)
            
        df = pd.DataFrame(columns=['word','freq'])
        df['word'] = counter.keys()
        df['freq'] = counter.values()
        df.to_csv(os.path.join(self.data_dir, "meta/vocab.csv"), index=None)

    def get_vocab_stats(self):
        file = os.path.join(self.data_dir, 'meta/vocab.csv')
        df = pd.read_csv(file)
        print(df)
def merge(manager):
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=config.max_length)
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
            chunk = manager.special_sampling(chunk, tokenizer)
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

def fast_comparison():
    fast = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=config.max_length)
    standard = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', max_length=config.max_length)

    file = os.path.join(config.data_dir, 'chunks\patents_2020_chunk_000008.csv')
    df = pd.read_csv(file)

    print('\n Fast: \n', fast.tokenize(df['text'][0][:300]))
    print('\n Standard: \n', standard.tokenize(df['text'][0][:300]))

def token_length_difference_checker():
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    file_counter = 0
    total_dif = 0
    files = glob.glob('data/patentsview/chunks/*csv')
    for file in files:
        df = pd.read_csv(file)
        splitted = df['text'].apply(lambda x: len(x.split()))
        bert_tokenized = df['text'].apply(lambda x: len( tokenizer_bert.tokenize(x)))
        print('\n splitted: ', sum(splitted)/len(splitted))
        print('\n BERT: ', sum(bert_tokenized)/len(bert_tokenized))
        break

if __name__ == '__main__':
    manager = Manager()
    manager.get_vocab_stats()