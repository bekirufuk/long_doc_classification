""" Patent Data Refiner

- Collects patent data in pandas chunks from a folder consist of multiple .tsv files.
- Merges each chunk with ipcr.tsv data which consist of patent classification information.
- Removes patent documents with word count below 512 or above 4096
- Equalizes the number of patents from each class
- Lowercases the documents
- Saves each chunk after refinement as a .csv file

"""
import yaml
import csv
import os.path
import datetime
from glob import glob
import pandas as pd
import nltk
import sys


def get_config():
    config_dir = 'src/config/refiner.yml'
    with open(config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def main():
    config = get_config()

    print('Ipcr data is loading...')
    ipcr = pd.read_csv('data/' + config['data_name'] + '/raw/ipcr.tsv',
                       sep="\t",
                       usecols=config['ipcr_columns'],
                       dtype=config['ipcr_dtypes'],
                       engine='c',
                       quoting=csv.QUOTE_NONNUMERIC
                       )
    ipcr.drop_duplicates(subset=['patent_id'], inplace=True, keep='first')

    print("\n----------\n DATA REFINE STARTED \n----------\n")

    record_count = config['initial_record_count']
    short_record_count = config['initial_record_count']

    files = glob('data/' + config['data_name'] + '/raw/g_detail*.tsv')
    files.sort(reverse=True)
    # For all patent data files in raw folder.
    for file_index, file_dir in enumerate(files):

        # Obtain the year info of the patent data from the file name.
        patent_year = os.path.basename(file_dir)[:-4][-4:]

        print(f"########## Processing {patent_year} ##########")

        # Read the current .tsv file in chunks.
        chunks = pd.read_csv(file_dir,
                             sep="\t",
                             usecols=config['patent_columns'],
                             dtype=config['patent_dtypes'],
                             engine='c',
                             chunksize=config['chunk_size'],
                             encoding='utf8',
                             quoting=csv.QUOTE_NONNUMERIC
                             )

        for chunk_index, chunk in enumerate(chunks):
            chunk.rename(columns={"description_text": "text"}, inplace=True)
            # Left joint the patents with their corresponding ipc class on patent_id columns
            chunk = chunk.merge(ipcr, how='left', on='patent_id')

            # Remove mislabeled or unconventional labels by selecting only patents from 8 classes.
            chunk = chunk[chunk['section'].isin(config['labels'])]

            # Lowercase the text and remove the lines.
            # chunk['text'] = chunk['text'].str.lower()
            # chunk['text'].replace(r'\n', '', regex=True)

            # Replace the textual section info with their corresponding ids to be suitable for a model input.
            chunk.replace({'section': config['label2id']}, inplace=True)

            # Drop NA rows for the given subset of columns
            chunk = chunk.dropna(axis=0, how='any', subset=['patent_id', 'text', 'section']).reset_index(drop=True)

            # Obtain the word count with nltk tokenizer. !! Disabled using default description_length column data instead
            word_tokens = [nltk.word_tokenize(patent) for patent in chunk['text']]
            chunk['word_counts'] = [len(tokens) for tokens in word_tokens]

            # Select only the patents within the optimal word count interval.
            optimal_bound = (chunk['word_counts'] >= 512) & (chunk['word_counts'] <= 4094)

            # Also select a classic bound for comparison. This will create a separate data
            # short_bound = chunk['word_counts'] < 512

            chunk = chunk[optimal_bound].reset_index(drop=True)
            # short_chunk = chunk[short_bound].reset_index(drop=True)

            # It is unused since divided into chunk and short_chunk by the word counts.

            # Select the number of patents from each class such that each class group would have the same number of patents with the minimum group.
            groups = chunk.groupby('section')
            chunk = groups.apply(lambda x: x.sample(groups.size().min()).reset_index(drop=True)).reset_index(drop=True)

            # groups = short_chunk.groupby('section')
            # short_chunk = groups.apply(lambda x: x.sample(groups.size().min()).reset_index(drop=True)).reset_index(drop=True)

            del groups

            # Rename the main classification field to labels for training convenience.
            chunk.rename(columns={'section': 'labels'}, inplace=True)
            # short_chunk.rename(columns={'section': 'labels'}, inplace=True)

            # Save the chunk as csv with its year and index info.
            chunk_dir = 'data/' + config['data_name'] + '/chunks/patents_' + patent_year + '_chunk_' + str(chunk_index).zfill(3) + '.csv'
            # short_chunk_dir = 'data/' + config['data_name'] + '/short_patent_chunks/patents_' + patent_year + '_chunk_' + str(chunk_index).zfill(3) + '.csv'

            chunk.to_csv(chunk_dir, index=False)
            # short_chunk.to_csv(short_chunk_dir, index=False)

            # Get the number of patent files for the chunk and append it to the general counter.
            chunk_len = chunk.shape[0]
            # short_chunk_len = short_chunk.shape[0]

            record_count += chunk_len
            # short_record_count += short_chunk_len

            print(f"================ FILE {patent_year} CHUNK {chunk_index} ============ Total Records:{record_count}")
            # print(f"================ FILE {patent_year} CHUNK {chunk_index} ============ Total Records:{short_record_count} Short Patent")

            # Log the step to the refiner_log.txt
            log_text = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
            log_text += "->Year {} Chunk {} saved containing {} records".format(patent_year, chunk_index, chunk_len)
            log_text += "Running record count for this merge: {}".format(record_count)

            log_dir = 'data/' + config['data_name'] + '/refiner_logs.txt'
            with open(log_dir, 'a') as f:
                f.write(log_text + '\n')

        os.remove(file_dir)
        print(f"{file_dir} removed. \n\n")


if __name__ == '__main__':
    main()
