import yaml
from transformers import LongformerTokenizerFast
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, load_dataset, load_from_disk


def get_config():
    config_dir = 'src/config/data.yml'
    with open(config_dir,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def get_dataset(data_name='refined_patents', seed=42, partitions={'train':0.8, 'test_validation':0.2}):
    """Obtain the refined_patents data files as a dataset.

        Parameters:
        data_name (string): Folder name of the data to be used. Default:'refined_patents' Current options: ['refined_patents', 'sample_refined_patents']

        Returns:
        dataset: A Huggingface dataset object. https://huggingface.co/docs/datasets/access 
        """
    print("Loading dataset from csv...")
    features = Features({   'patent_id': Value('string'),
                            'text': Value('string'),
                            'labels': ClassLabel(names=["A","B","C","D","E","F","G","H"]),
                            'ipc_class': Value('string'),
                            'subclass': Value('string'),
                        })
    data_files = 'data/' + data_name + '/chunks/preprocessed/*.csv'

    dataset = load_dataset('csv', data_files=data_files, features=features, cache_dir='data/'+data_name+'/cache')
    dataset = dataset['train'].train_test_split(test_size=partitions['test_validation'], shuffle=True, seed=seed)
    test_val = dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=seed)

    dataset = DatasetDict({
        'train': dataset['train'],
        'validation': test_val['train'],
        'test': test_val['test']
    })
    return dataset

def batch_tokenizer(batch, tokenizer):
    return tokenizer(batch["text"], padding='max_length', truncation=True)

def longformer_tokenizer(dataset):
    print('Tokenizing the dataset with LongformerTokenizer')
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=4096)
    tokenized_data = dataset.map(batch_tokenizer, batched=True, remove_columns=['text'], fn_kwargs= {"tokenizer":tokenizer})
    #tokenized_data.set_format("torch")
    return tokenized_data

def get_longformer_tokens(data_name='refined_patents', load_tokens=False, test_data_only=False, train_sample_size=None, validation_sample_size=None, test_sample_size=None):
    if load_tokens:
        tokenized_data = load_from_disk('data/'+data_name+'/tokenized/longformer_tokenizer_no_stopwords/')
    else:
        dataset = get_dataset()
        tokenized_data = longformer_tokenizer(dataset)
        save_longformer_tokens(tokenized_data)

    #tokenized_data.set_format("torch")
    
    if test_data_only:
        return tokenized_data['test'].select(range(test_sample_size))
    else:
        return tokenized_data['train'].select(range(train_sample_size)) #, tokenized_data['validation'].select(range(validation_sample_size))

def save_longformer_tokens(tokenized_data, data_name='refined_patents'):
    print("Saving tokenized data...")
    tokenized_data.save_to_disk('data/'+data_name+'/tokenized/longformer_tokenizer_no_stopwords/')

if __name__ == '__main__':
    dataset = get_dataset()
    tokenized_data = longformer_tokenizer(dataset)
    save_longformer_tokens(tokenized_data)