#pip install transformers datasets evaluate -U accelerate -U transformers sentencepiece protobuf

# Set the name of the HuggingFace repo to download pretrained model and tokenizer from
checkpoint_iw = 'homersimpson/cat-sum-iw'

# Login to HF with token login
from huggingface_hub import login
login(token='your-token', add_to_git_credential=True)

# Download pretrained model
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
iw_sum_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_iw)

# Import relevant dataset
from datasets import load_dataset
catalan = load_dataset('projecte-aina/casum')

# Download model tokenizer
from transformers import AutoTokenizer
iw_sum_tokenizer = AutoTokenizer.from_pretrained(checkpoint_iw)

# Initialise prefix, necessary when working with T5-architecture models
prefix = "summarize: "

def preprocess_function_iw(examples):
    ''' Preprocesses and tokenizes data, appending the relevant prefix

            Args: Batch of training dataset

            Returns: Tokenized entry, preprocessed with the relevant prefix
    '''
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = iw_sum_tokenizer(inputs, max_length=1024, truncation=True)

    labels = iw_sum_tokenizer(text_target=examples["summary"], max_length=256, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset with model's tokenizer
tokenized_catalan_iw = catalan.map(preprocess_function_iw, batched=True)

# Load data collator using loaded tokenizer
from transformers import DataCollatorForSeq2Seq
data_collator_iw = DataCollatorForSeq2Seq(tokenizer=es_sum_tokenizer, model=iw_sum_model)

# Load evaluation metric - in this case rouge
import evaluate

rouge = evaluate.load("rouge")
import numpy as np

# Define the evaluation function, here calculating the ROUGE-1, ROUGE-2, and ROUGE-L scores of model's predictions, also returns average generation length
def compute_metrics_iw(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, iw_sum_tokenizer.pad_token_id)
    decoded_preds = iw_sum_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, iw_sum_tokenizer.pad_token_id)
    decoded_labels = iw_sum_tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != iw_sum_tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# Define training arguments for each model, create instance of Trainer class, and evaluate
training_args_iw = Seq2SeqTrainingArguments(
    output_dir="summarisation/eval/iw",
    # Saves weights every epoch
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_eval_batch_size=16,
    # Using mixed precision due to memory limitations, bf16 for T5 compatibility
    bf16 = True,
    generation_max_length=256,
    weight_decay=0.01,
    save_strategy="epoch",
    num_train_epochs=4,
    # Loads best weights checkpoint, based on ROUGE-2
    load_best_model_at_end=True,
    metric_for_best_model='eval_rouge2',
    predict_with_generate=True,
    push_to_hub=False,
)

trainer_iw = Seq2SeqTrainer(
    model=iw_sum_model,
    args=training_args_iw,
    eval_dataset=tokenized_catalan_iw["test"],
    tokenizer=iw_sum_tokenizer,
    data_collator=data_collator_iw,
    compute_metrics=compute_metrics_iw,
)

# Print results of evaluation
print(trainer_iw.evaluate())
