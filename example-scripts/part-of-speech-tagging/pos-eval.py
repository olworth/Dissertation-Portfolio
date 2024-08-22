#!pip install transformers datasets evaluate -U accelerate -U transformers sentencepiece protobuf

# Set the name of the HuggingFace repo to download finetuned model and tokenizer from
es_checkpoint = 'homersimpson/cat-pos-es-5'

# Login to HF with token login
from huggingface_hub import login
login(token='your-token', add_to_git_credential=True)

# Import relevant dataset
from datasets import load_dataset
catalan_raw = load_dataset('universal_dependencies', 'ca_ancora')

# Initialise the list of NER labels to be used by the model
label_list = catalan_raw['train'].features[f"upos"].feature.names

# Download model tokenizer
from transformers import AutoTokenizer
es_tokenizer = AutoTokenizer.from_pretrained(es_checkpoint,add_prefix_space=True)

def es_tokenize_and_align_labels(examples):
    ''' Tokenizes and preprocesses data, mapping subword tokens to their original words and labelling only the first of these.

            Args: Batch of training dataset.

            Returns: Tokenized and preprocessed batch
    '''
    tokenized_inputs = es_tokenizer(examples["tokens"], truncation=True, max_length=512, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"upos"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i) # Map tokens to their respective word
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:   # Set the special tokens to -100
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx: # Only label the first token of a given word
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize dataset with model's tokenizer
tokenized_catalan_es = catalan_raw.map(es_tokenize_and_align_labels, batched=True)

# Load evaluation metric - in this case the sequeval implementations of F1 and accuracy
import evaluate

seqeval = evaluate.load("seqeval")
import numpy as np

# Define the evaluation function, which calculates precision, recall, F1, and accuracy of model predictions
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Create two dictionaries mapping IDs to labels and vice versa, for model purposes
id2label = {
    0:'NOUN',
    1:'PUNCT',
    2:'ADP',
    3:'NUM',
    4:'SYM',
    5:'SCONJ',
    6:'ADJ',
    7:'PART',
    8:'DET',
    9:'CCONJ',
    10:'PROPN',
    11:'PRON',
    12:'X',
    13:'_',
    14:'ADV',
    15:'INTJ',
    16:'VERB',
    17:'AUX',
}
label2id = {
    'NOUN':0,
    'PUNCT':1,
    'ADP':2,
    'NUM':3,
    'SYM':4,
    'SCONJ':5,
    'ADJ':6,
    'PART':7,
    'DET':8,
    'CCONJ':9,
    'PROPN':10,
    'PRON':11,
    'X':12,
    '_':13,
    'ADV':14,
    'INTJ':15,
    'VERB':16,
    'AUX':17,
}

# Load model
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
es_model = AutoModelForTokenClassification.from_pretrained(
    es_checkpoint, num_labels=18, id2label=id2label, label2id=label2id
)

# Load data collator using loaded tokenizer
from transformers import DataCollatorForTokenClassification
es_data_collator = DataCollatorForTokenClassification(tokenizer=es_tokenizer)

# Define training arguments for each model, create instance of Trainer class, and evaluate
es_training_args = TrainingArguments(
    output_dir="pos/es",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
    save_total_limit=4,
    push_to_hub=False,
)

es_trainer = Trainer(
    model=es_model,
    args=es_training_args,
    eval_dataset=tokenized_catalan_es["test"],
    tokenizer=es_tokenizer,
    data_collator=es_data_collator,
    compute_metrics=compute_metrics,
)

# Print results of evaluation
print(es_trainer.evaluate())
