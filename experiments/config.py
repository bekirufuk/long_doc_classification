import os
import json
import datetime


seed = 42

# Directory parameters
data_dir = os.path.expanduser('data/patentsview')
root_dir = os.path.expanduser("experiments")

# Model parameters.
label2id = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
id2label =  {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
labels_list = ["A","B","C","D","E","F","G","H"]
num_labels = 8
model_name="longformer_with_tfidf_balanced_nltk_definitive"

load_local_checkpoint = False

# Training parameters.
initial_step = 0 # Increase it for a checkpoint.
num_epochs = 5
batch_size = 8

downsample = True
num_train_samples = 20000
num_test_samples = 3200

num_train_batches = (num_train_samples / batch_size)
num_test_batches = (num_test_samples / batch_size)

num_train_steps = num_train_batches * num_epochs
num_test_steps = num_test_batches * num_epochs

test_split_ratio = 0.1

lr = 3e-5
weight_decay=0.01
scheduler_type = 'cosine'
num_warmup_steps = int(0.1 * num_train_steps)

global_attention_mapping = 'tfidf'

# Tokenizer parameters.
max_length = 4096 #16384
load_saved_tokens = True
save_tokens = False

# Data parameters.

upload_to_hf = False
upload_repo_name = 'ufukhaman/uspto_patents_2019'

download_from_hf = False
download_repo_name = 'ufukhaman/uspto_patents_2019'

# Logging parameters.
log_interval = int(num_train_steps/200)
date = datetime.datetime.now()
log_name = model_name + '_' + date.strftime("%Y-%m-%d-%H%M")

with open(os.path.join(root_dir, 'longformer_config.json')) as f:
    model_config = json.load(f)

wandb_config = dict(

    epochs=num_epochs,
    batch_size=batch_size,

    num_train_samples=num_train_samples,
    num_test_samples=num_test_samples,

    num_train_batches=num_train_batches,
    num_test_batches=num_test_batches,

    num_train_steps = num_train_steps,
    num_test_steps = num_test_steps,

    learning_rate = lr,
    weight_decay = weight_decay,
    scheduler_type = scheduler_type,
    num_warmup_steps = num_warmup_steps,

    global_attention_mapping = global_attention_mapping,

    model = model_name,
    dataset = 'balanced_trimmed_200K',
    input_size = max_length,
    classes = num_labels,
    log_interval = log_interval,
    model_config = model_config
)