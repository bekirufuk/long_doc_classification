import pandas as pd
import glob, os
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import re

data_name = 'refined_patents'

def preprocess_chunks(limit=None):
    files = glob.glob("data/"+data_name+"/chunks/raw/*.csv")
    if limit:
        files = files[:limit]

    for file in tqdm(files):
        chunk = pd.read_csv(file)
        # Remove every character except letters, comma(,), dot(.) and a space( )
        chunk['text'] = chunk['text'].apply(lambda x: re.sub(r"[^A-Za-z., ]", "", x))

        # chunk['text'] = chunk['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))

        chunk_dir = "data/"+data_name+"/chunks/preprocessed/"+os.path.basename(file)
        print(chunk_dir)
        chunk.to_csv(chunk_dir, index=False)


if __name__ == '__main__':
    preprocess_chunks()
