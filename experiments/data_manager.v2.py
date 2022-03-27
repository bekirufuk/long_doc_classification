import os, glob
import json
import pandas as pd
import config

from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, LongformerConfig, get_scheduler, logging


def create_data_files(patents, ipcr):
    chunk_count = 0
    patent_count = 0
    for chunk in patents:
        # Combine patent with respective section info.
        data = chunk.merge(ipcr, how='left', on='patent_id')

        # Replace the letters with integers to create a suitable training input.
        data.replace({'section':config.label2id}, inplace=True)
        data['section'].astype(int)
        # data.rename(columns = {'section':'label'}, inplace = True)

        # Append the batch to the main data file.
        print(data.info())
        print(data.describe())
        data.to_csv(os.path.join(config.data_dir, 'patents_'+config.patents_year+'.csv'),
            sep=',',
            mode='a',
            index=False,
            columns=['text', 'section'],
            header = None
        )

        # Seperately write the batches as individual files. (optional)
        data.to_csv(os.path.join(config.data_dir, 'chunks/patents_'+config.patents_year+'_chunk_'+str(chunk_count).zfill(6)+'.csv'),
            sep=',',
            mode='w',
            index=False,
            columns=['text', 'section'],
            header = ['text','label']
        )

        patent_count += data.shape[0]
        chunk_count += 1
        print("Chunk {0} -> Total processed patent count: {1}".format(chunk_count, patent_count))

        if config.single_chunk:
            break

    # Write the basic info about process data for ease of use.
    with open(os.path.join(config.data_dir, "meta/patents_"+config.patents_year+"_meta.json"), "a") as f:
        f.write(json.dumps({"num_chunks":chunk_count,
                            "chunk_size":config.chunk_size,
                            "num_patents":patent_count
                            }))

def load_ipcr():
    # Icpr file holds detailed class information about the patents.
    # We will only investigate section column which consist of 8 distinct classes.
    # But ipc_class and sublass values are also obtained for possible future use.

    print("Ipcr data loading...")
    file_dir = os.path.join(config.data_dir,'ipcr.tsv')
    ipcr = pd.read_csv(file_dir,
        sep="\t",
        usecols=['patent_id','section','ipc_class', 'subclass'],
        dtype={'patent_id':object, 'section':object, 'ipc_class':object, 'subclass':object},
        engine='c',
        )
    ipcr.drop_duplicates(subset=['patent_id'], inplace=True, keep='first')
    print("Ipcr data loaded.")
    return ipcr
    

def get_patents_list():
    # Returns raw patent .tsv files belonging seperate years download from patentsview.
    return glob.glob(os.path.join(config.data_dir, "detail*.tsv"))
    

def get_chunks(patent_dir):
    print("\n\n ########## Processing {} with chunk size of {} ##########".format(os.path.basename(patent_dir), config.chunk_size))
    return pd.read_csv(patent_dir,
        sep="\t",
        usecols=['patent_id', 'text'],
        dtype={'patent_id':object, 'text':object},
        engine='c',
        chunksize=config.chunk_size,
        nrows=30,
        encoding='utf8',
        )


def merge_and_prepeare(chunk):
    # Combine patent with respective section info.
    chunk = chunk.merge(ipcr, how='left', on='patent_id')
    chunk.replace({'section':config.label2id}, inplace=True)
    chunk['section'].astype(int)
    return chunk



if __name__ == '__main__':

    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=config.max_length)
    ipcr = load_ipcr()

    for patent_file_counter, patent_dir in enumerate(get_patents_list()):

        chunks = get_chunks(patent_dir)
        chunk_counter = 0
        for chunk in chunks:
            chunk_counter += 1
            print("=============================== FILE {} CHUNK {} ================================".format(patent_file_counter+1, chunk_counter))
            chunk = merge_and_prepeare(chunk)

            chunk = tokenizer(chunk)
            print(chunk)

"""
    # All patents from asinge year chunked. Multiple year processing will be implemented in future.
    patents = pd.read_csv(os.path.join(config.data_dir, 'detail_desc_text_'+config.patents_year+'.tsv'),
        sep="\t",
        usecols=['patent_id', 'text'],
        dtype={'patent_id':object, 'text':object},
        engine='c',
        chunksize=config.chunk_size,
        encoding='utf8',
        )
    print("Patents data chunked with chunk_size={}.".format(config.chunk_size))

    # Drop duplicates because this table might have duplicating patent_id sharing the same section with different subclasses.
    ipcr = ipcr.drop_duplicates(subset=['patent_id'])
    print("Ipcr data de-duplicated.")

    print("\n----------\n DATA PROCESSING STARTED \n----------\n")

    pd.DataFrame({}, columns=['text', 'label']).to_csv(os.path.join(config.data_dir, 'patents_'+config.patents_year+'.csv'),
        index=False
        )
    create_data_files(patents, ipcr)

    print("\n----------\n DATA PROCESSING FINISHED \n----------\n")
"""