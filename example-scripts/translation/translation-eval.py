#pip install transformers datasets evaluate -U accelerate -U transformers sentencepiece protobuf

# Set the name of the HuggingFace repo to download finetuned model and tokenizer from
es_checkpoint = 'homersimpson/translation-es'

# Login to HF with token login
from huggingface_hub import login
login(token='your-token', add_to_git_credential=True)

# Import relevant dataset
from datasets import load_dataset
es = load_dataset('homersimpson/opensubtitles_es')

# Download model tokenizer
from transformers import AutoTokenizer
es_tokenizer = AutoTokenizer.from_pretrained(es_checkpoint)

# Initialise prefix, necessary when working with T5-architecture models
es_source_lang = "es"
es_target_lang = "ca"
es_prefix = "translate Spanish to Catalan: "

def es_preprocess_function(examples):
    ''' Preprocesses and tokenizes data, appending the relevant prefixes

            Args: Batch of training dataset

            Returns: Tokenized entry, preprocessed with the relevant prefixes
    '''
    inputs = [es_prefix + example[es_source_lang] for example in examples["translation"]]
    targets = [example[es_target_lang] for example in examples["translation"]]
    model_inputs = es_tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

# Tokenize dataset with model's tokenizer
tokenized_es = es.map(es_preprocess_function, batched=True)

# Load data collator using loaded tokenizer
from transformers import DataCollatorForSeq2Seq
es_data_collator = DataCollatorForSeq2Seq(tokenizer=es_tokenizer, model=es_checkpoint)

# Load evaluation metric - in this case the sacrebleu implementation of BLEU
import evaluate

metric = evaluate.load("sacrebleu")
import numpy as np

# Some helpful format postprocessing
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

# Define the evaluation function, here calculating the BLEU score of the model's predictions, also returns average generation length
def es_compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = es_tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, es_tokenizer.pad_token_id)
    decoded_labels = es_tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != es_tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result

# Load data collator using loaded tokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
es_model = AutoModelForSeq2SeqLM.from_pretrained(es_checkpoint)

# Define training arguments for each model, create instance of Trainer class, and evaluate
es_eval_args = Seq2SeqTrainingArguments(
    output_dir="translation/es/eval",
    evaluation_strategy="epoch",
    # Saves weights every epoch
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    num_train_epochs=4,
    # Loads best weights checkpoint, based on BLEU
    load_best_model_at_end=True,
    metric_for_best_model='eval_bleu',
    predict_with_generate=True,
    push_to_hub=False,
)

es_trainer = Seq2SeqTrainer(
    model=es_model,
    args=es_eval_args,
    eval_dataset=tokenized_es["test"],
    tokenizer=es_tokenizer,
    data_collator=es_data_collator,
    compute_metrics=es_compute_metrics,
)

# Print results of evaluation
print(es_trainer.evaluate())
