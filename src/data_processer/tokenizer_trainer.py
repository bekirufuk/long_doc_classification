import sys, os

from transformers import AutoTokenizer, BertTokenizerFast
from datasets import DatasetDict, Features, Value, ClassLabel, load_dataset, load_from_disk

#sys.path.append(os.getcwd())
#from src.data_processer.process import get_dataset


def get_dataset(data_name='refined_patents', seed=42, partitions={'train':0.8, 'test_validation':0.2}):
    print("Loading dataset from csv...")
    features = Features({   'patent_id': Value('string'),
                            'text': Value('string'),
                            'labels': ClassLabel(names=["A","B","C","D","E","F","G","H"]),
                            'ipc_class': Value('string'),
                            'subclass': Value('string'),
                        })
                        
    data_files = 'data/'+data_name+'/chunks/*.csv' # add preprocess forlder

    dataset = load_dataset('csv', data_files=data_files, features=features, cache_dir='data/'+data_name+'/cache')
    dataset = dataset['train'].train_test_split(test_size=partitions['test_validation'], shuffle=True, seed=seed)
    test_val = dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=seed)

    dataset = DatasetDict({
        'train': dataset['train'],
        'validation': test_val['train'],
        'test': test_val['test']
    })
    return dataset

def get_training_corpus(dataset):
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["text"]


tokenizer_bert = BertTokenizerFast.from_pretrained("bert-base-uncased", max_length=4096)

dataset = get_dataset()
training_corpus = get_training_corpus(dataset['train'])

trained_tokenizer_bert = tokenizer_bert.train_new_from_iterator(training_corpus, 50000)
trained_tokenizer_bert.save_pretrained("tokenizers/bert_trained_on_patent_data")
