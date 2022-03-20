import os
import json
import datetime


date = datetime.datetime.now()
seed = 42

# Directory parameters
data_dir = os.path.expanduser('data/patentsview')
root_dir = os.path.expanduser("experiments")
patents_year="2019"

# Model parameters.
label2id = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
id2label =  {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
labels_list = ["A","B","C","D","E","F","G","H"]
num_labels = 8
model_name="no_global_attention_medium_scale_0.1"

save_model = True
load_local_checkpoint = False

# Training parameters.
num_epochs = 5
batch_size = 8

num_train_samples = 16000
num_test_samples = 1600

num_train_batches = (num_train_samples / batch_size)
num_test_batches = (num_test_samples / batch_size)

num_train_steps = num_train_batches * num_epochs
num_test_steps = num_test_batches * num_epochs

lr = 3e-5
scheduler_type = 'linear'
num_warmup_steps = int(0.3 * num_train_steps)

global_attention_mapping = 'none'

# Tokenizer parameters.
max_length = 4096 #16384
load_saved_tokens = True
save_tokens = False

# Data preprocessing parameters.
chunk_size = 10000
single_chunk = False

upload_to_hf = False
upload_repo_name = 'ufukhaman/uspto_patents_2019'

download_from_hf = False
download_repo_name = 'ufukhaman/uspto_patents_2019'

# Logging parameters.
log_interval = 50

log_name = 'no_global_medium_' + date.strftime("%Y-%m-%d-%H%M")

# with open(os.path.join(root_dir, 'longformer_config.json')) as f:
#     model_config = json.load(f)
model_config = ''

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
    scheduler_type = scheduler_type,
    num_warmup_steps = num_warmup_steps,

    global_attention_mapping = global_attention_mapping,

    dataset = 'patents_'+patents_year,
    model = model_name,
    input_size = max_length,
    classes = num_labels,
    log_interval = log_interval,
    model_config = model_config
)
