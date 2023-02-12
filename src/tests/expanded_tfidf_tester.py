import sys, os

import pandas as pd

sys.path.append(os.getcwd())
from src.data_processer.process import get_tokens
from src.utils.attention_mapper import expanded_tfidf_qual_analysis, expanded_unique_tfidf_qual_analysis

from torch.utils.data import DataLoader

# Load the sparse tfidf matrix and the feature_names(containing input_ids as words)
tfidf_sparse = pd.read_pickle('data/refined_patents/tfidf/longformer_tokenizer_no_stopwords/train_tfidf_sparse.pkl')
f_names = pd.read_pickle('data/refined_patents/tfidf/longformer_tokenizer_no_stopwords/train_f_list.pkl')

train_data = get_tokens('longformer_tokenizer_no_stopwords', test_data_only=False, train_sample_size=16)
train_data = train_data.remove_columns(['patent_id', 'ipc_class', 'subclass'])
train_data.set_format("torch")

train_dataloader = DataLoader(train_data, batch_size=8)

for batch_id, batch in enumerate(train_dataloader):
    global_attention_map = expanded_unique_tfidf_qual_analysis(tfidf_sparse[0:8], f_names, batch['input_ids'], batch['labels'])

    

