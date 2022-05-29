from sklearn.metrics import accuracy_score

def compute_metrics(references, predictions):
    return accuracy_score(references, predictions)