'''
    Compares the document and word counts of generated tfidf matrices with the actual data.
    prints 'FAIL - <reason>' if any found, prints 'PASS - <test_name>' else.
'''

import pandas as pd
import numpy as np

from datasets import load_from_disk


# Initilize the parameters
data_name = 'refined_patents'
tokenizer = 'longformer'
partition='train'

# Check Label Based TFIDF

# Read the tokenized data.
tokenized_data = load_from_disk("data/"+data_name+"/tokenized/"+tokenizer+"_tokenizer/"+partition)

df = pd.DataFrame(columns=['text', 'labels'])
df['text'] = [' '.join([str(word) for word in doc]) for doc in tokenized_data['input_ids']]
df['labels'] = tokenized_data['labels']

print('\n### Label Based Tfidf Tests...')
# For each label group in tokenized data, compare the previously generated tfidf document and word counts. 
groups = df.groupby('labels')
for label, group in groups:

    tfidf_sparse = pd.read_pickle('data/refined_patents/tfidf/label_based/'+str(label)+'_'+partition+'_tfidf.pkl')
    feature_names = pd.read_pickle('data/refined_patents/tfidf/label_based/'+str(label)+'_'+partition+'_feature_names.pkl')

    tfidf = pd.DataFrame(tfidf_sparse.toarray(), columns=feature_names)

    if len(group) != len(tfidf):
        print('FAIL - Document count unmatched for label {}!'.format(label))
    else:
        print('PASS - Label {}'.format(label))
        for i in range (len(group)):
            tfidf_count = np.count_nonzero(np.array(list(tfidf.loc[i])))
            group_count = len(list(set(group.reset_index().loc[i]['text'].split(' '))))
            
            if tfidf_count != group_count:
                print('\t FAIL - Word Count mismatch for document {}'.format(i))

print('\n### Full Corpus Tfidf Tests...')
# Check word counts of general tfidf.
tfidf_sparse = pd.read_pickle('data/refined_patents/tfidf/'+partition+'_tfidf.pkl')
feature_names = pd.read_pickle('data/refined_patents/tfidf/'+partition+'_feature_names.pkl')

tfidf = pd.DataFrame(tfidf_sparse.toarray(), columns=feature_names)

if len(df) == len(tfidf):
    print('PASS - Document Count')
else:
    print('FAIL - Document Count Mismatch')

for i in range(len(df)):
    tfidf_count = np.count_nonzero(np.array(tfidf.loc[i]))
    df_count = len(list(set(df.loc[i]['text'].split(' '))))
    if tfidf_count != df_count:
        print('\t FAIL - Word Count mismatch for document {}'.format(i))
        break
if i == len(df)-1:
    print('PASS - Word count matches for all documents')