'''
    Compares the document and word counts of generated tfidf matrices with the actual data.
    prints 'FAIL - <reason>' if any found, prints 'PASS - <test_name>' else.
'''

import pandas as pd
import numpy as np
from tqdm import tqdm

from datasets import load_from_disk


# Initilize the parameters
data_name = 'refined_patents'
tokenizer = 'longformer'
partition='train'
limit=20000

# Check Label Based TFIDF

# Read the tokenized data.
print('Loading Data...')

tokenized_data = load_from_disk("../data/refined_patents/tokenized/longformer_tokenizer_no_stopwords/")
tokenized_data = tokenized_data[partition]
df = tokenized_data.remove_columns(['ipc_class', 'subclass'])


tfidf_sparse = pd.read_pickle('../data/refined_patents/tfidf/longformer_tokenizer_no_stopwords/train_tfidf_sparse.pkl')
f_names = pd.read_pickle('../data/refined_patents/tfidf/longformer_tokenizer_no_stopwords/train_f_list.pkl')

tfidf = pd.DataFrame(tfidf_sparse.toarray(), columns=f_names)

print('Tfidf word count comparison...')
err_counter=0
for i in tqdm(range(df.shape[0])):
    token_count = len(pd.Series(df[0]['input_ids']).unique())
    tfidf_count = np.count_nonzero(tfidf.loc[0])
    if token_count != tfidf_count:
        print('Word count mismatch! for doc {}. Word counts: {}-{}'.format(i,))
        err_counter += 1
if err_counter == 0:
    print('All documents matches')
else:
    print('{} mismatches found.'.format(err_counter))