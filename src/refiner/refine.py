""" Patent Data Refiner

- Collects patent data in pandas chunks from a folder consist of multiple .tsv files.
- Merges each chunk with ipcr.tsv data which consist of patent classification information.
- Removes patent documents with word count below 512 or above 4096
- Equalizes the number of patents from each class
- Lowercases the documents
- Saves each chunk after refinement as a .csv file

"""
import yaml
import os.path
import datetime
from glob import glob
import pandas as pd
import nltk

def get_config():
    config_dir = 'src/config/refiner.yml'
    with open(config_dir,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


if __name__ == '__main__':

    config = get_config()

    print('Ipcr data is loading...')
    ipcr = pd.read_csv('data/' + config['data_name'] + '/ipcr.tsv',
            sep="\t",
            usecols=config['ipcr_columns'],
            dtype=config['ipcr_dtypes'],
            engine='c',
            )
    ipcr.drop_duplicates(subset=['patent_id'], inplace=True, keep='first')

    print("\n----------\n DATA REFINE STARTED \n----------\n")

    record_count = config['initial_record_count']
    # For all patent data files in raw folder.
    for file_index, file_dir in enumerate(glob('data/' + config['data_name'] + '/raw/*.tsv')):

        # Obtain the year info of the patent data from the file name.
        patent_year = os.path.basename(file_dir)[17:21]

        print("########## Processing {} ##########".format(os.path.basename(file_dir)))

        # Read the current .tsv file in chunks.
        chunks = pd.read_csv(file_dir,
                        sep="\t",
                        usecols=config['patent_columns'],
                        dtype=config['patent_dtypes'],
                        engine='c',
                        chunksize=config['chunksize'],
                        encoding='utf8',
                        )

        for chunk_index, chunk in enumerate(chunks):
            
            # Left joint the patents with their corresponding ipc class on patent_id columns
            chunk = chunk.merge(ipcr, how='left', on='patent_id')

            # Remove mislabeled or unconventional labels by selecting only patents from 8 classes. 
            chunk = chunk[chunk['section'].isin(config['labels'])]

            # Replace the textual section info with their correpondind ids to be suitable for a model input.
            chunk.replace({'section':config['label2id']}, inplace=True)
            
            #Obtain the word count with nltk tokenizer
            word_counts = chunk['text'].apply(nltk.word_tokenize).apply(lambda x: len(x))

            # Select only the patents within the optimal word count interval
            chunk = chunk[(word_counts>=512) & (word_counts<=4094)].reset_index(drop=True)

            # Select the number of patents from each class such that each class group would have the same number of patents with the minimum group.  
            groups = chunk.groupby('section')
            chunk = groups.apply(lambda x: x.sample(groups.size().min()).reset_index(drop=True)).reset_index(drop=True)

            # Lowercase the text
            chunk['text'] = chunk['text'].str.lower()

            # Save the chunk as csv with its year and index info.
            chunk_dir = 'data/' + config['data_name'] + '/chunks/patents_' + patent_year + '_chunk_' + str(chunk_index).zfill(3) + '.csv'
            chunk.to_csv(chunk_dir, index=False)

            # Get the number of patent files for the chunk and append it to the general counter.
            chunk_len = chunk.shape[0]
            record_count += chunk_len

            print("================ FILE {} CHUNK {} ============ Total Records:{}".format(patent_year, chunk_index, record_count))

            #Log the step to the refiner_log.txt
            log_text = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
            log_text += "->Year {} Chunk {} saved containing {} records - ".format(patent_year, chunk_index, chunk_len)
            log_text += "Running record count for this merge: {}".format(record_count)
            
            log_dir = 'data/' + config['data_name'] + '/refiner_logs.txt'
            with open(log_dir, 'a') as f:
                f.write(log_text + '\n')