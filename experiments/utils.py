from cgi import print_arguments
from statistics import mean
import pandas as pd
import config
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(predictions, references):

    precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average='micro')
    acc = accuracy_score(references, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
#dtype=torch.long, device=device tfidf[str(x)][i]
def attention_mapper(device, global_attention_mask, f_names, tfidf):

    tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

    global_attention_mask = global_attention_mask.type(torch.float32).cpu()

    for i, item in enumerate(global_attention_mask):
        globals = item.apply_(lambda x: 0 if x == 1 else tfidf[str(int(x))][i])
        globals = torch.argsort(globals, dim=0, descending=True)[:128]
        global_attention_mask[i][globals] = 1

    global_attention_mask[:, 0] = 1
    """df = pd.DataFrame(global_attention_mask.numpy())
    print('\n ##########################')
    xx = np.count_nonzero(df, axis=1)
    print("Batch Mean: ", mean(xx))
    print(xx)
    print(labels)"""
    global_attention_mask = torch.tensor(global_attention_mask.numpy(), dtype=torch.long, device=device)
    
    return global_attention_mask



def visualize_attention_map(map):
    map_line = np.array(map)
    att_window = config.model_config['attention_window'][0]

    map = []
    for x in range(len(map_line)):
        copy_map_line = map_line.copy()

        starting_pos = int(max(x-att_window/2, 0))
        ending_pos = int(min(x+att_window/2, len(map_line)))

        copy_map_line[starting_pos:ending_pos] = 1  
        map.append(list(copy_map_line))
    plt.spy(map, precision = 0.1, markersize = 5)
    plt.show()
    return plt

if __name__ == '__main__':

    attention_map = attention_mapper('cpu',0,0,0)
