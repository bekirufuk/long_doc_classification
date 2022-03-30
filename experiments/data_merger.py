import os, glob
import pandas as pd
import datetime
import config

# Merges Detailed ddescription files of patents with therir class info. Then chunks them into .csv files.
class Merger():
    def __init__(self, data_name='patentsview', chunksize=20000, engine='c', testing=False, nrows=30, skiprows=0):

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
            f.write('#### Merge Operation Finished #### chunk size:{}, end date:{}, total recorded:{} \n\n'.format(self.chunksize,
                                                                                                        datetime.datetime.now().strftime("%Y-%m-%d-%H:%M"),
                                                                                                        self.saved_record_count
                                                                                                        ))