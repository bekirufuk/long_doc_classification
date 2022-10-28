import sys, os

from transformers import BertTokenizerFast

sys.path.append(os.getcwd())
from src.data_processer.process import get_dataset


def get_training_corpus(dataset):
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["text"]


tokenizer_bert = BertTokenizerFast.from_pretrained("bert-base-uncased", max_length=4096)

dataset = get_dataset()
training_corpus = get_training_corpus(dataset['train'])

trained_tokenizer_bert = tokenizer_bert.train_new_from_iterator(training_corpus, 50000)
trained_tokenizer_bert.save_pretrained("tokenizers/bert_trained_on_patent_data")
