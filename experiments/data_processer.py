import glob, os
import config
import pandas as pd
from datasets import Features, Value, ClassLabel, load_dataset
from transformers import LongformerTokenizerFast



class Processer():
    def __init__(self, data_name='patentsview', max_token=4096, label_balance=10000):

        self.data_name = data_name
        self.data_dir = os.path.expanduser('data/'+self.data_name+'/chunks')

        self.max_token = max_token
        self.label_balance = label_balance

    def get_files(self):
        return glob.glob(os.path.join(self.data_dir, "*.csv"))
    
    def batch_tokenizer(batch, tokenizer):
        return tokenizer(batch["text"], padding='max_length', truncation=True)

    def load_data(self):
        files_list = self.get_files()[:5]
        class_names = config.labels_list

        features = Features({'text': Value('string'), 'section': ClassLabel(names=class_names)})
        return load_dataset('csv', data_files=files_list,
                            features=features,
                            cache_dir=os.path.join(self.data_dir, 'cache')
                            )
    
    def clean_not_label(self):
        files_list = self.get_files()
        for i, file in enumerate(files_list):
            df = pd.read_csv(file)
            df =df[(df['section'].isin(config.labels_list)) | (df['section'].isin(['0','1','2','3','4','5','6','7']))]
            df['section'].astype(int)
            df.rename(columns = {'section':'label'}, inplace = True)
            print('File {}: {} records'.format(i, df.shape[0]))
            df.to_csv(file, index=False)

    def fix_length(self):
        files_list = self.get_files()
        for file in files_list:
            df = pd.read_csv(file)
            tokenized_data = LongformerTokenizerFast(df['section'])

