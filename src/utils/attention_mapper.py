import pandas as pd
from torch import zeros, float32

def map_tfidf(tfidf, f_names, input_ids):

    # Get ready the tfidf scores for the current batch
    tfidf = pd.DataFrame(tfidf.toarray(), columns=f_names)

    # Put the mask in cpu temporarily for .apply_() process.
    global_attention_mask = input_ids.type(float32).cpu()

    for i, item in enumerate(global_attention_mask): # For every document(item) in the batch.

        # x is 1 for padding tokens, so it would be zero, all other tokens matches to their tfidf scores.
        #i represents the document id (from zero up to batch size) and the x represents the input_id, being the column name of the tfidf DataFrame.
        globals = item.apply_(lambda x: 0 if x == 1 else tfidf.iloc[i,int(x)])
        print(globals)
        print('\n################\n')

    return tfidf
