import os
import json
import config
import pandas as pd


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




if __name__ == '__main__':

    # Icpr file holds detailed class information about the patents.
    # We will only investigate section column which consist of 8 distinct classes.
    ipcr = pd.read_csv(os.path.join(config.data_dir, 'ipcr.tsv'),
        sep="\t",
        usecols=['patent_id','section'],
        dtype={'patent_id':object, 'section':object},
        engine='c',
        )
    print("Ipcr data loaded.")

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