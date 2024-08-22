#pip install transformers datasets evaluate -U accelerate -U transformers sentencepiece protobuf

# Set the name of the HuggingFace repo to download finetuned model and tokenizer from
es_checkpoint = 'homersimpson/5-cat-belebele-es'

# Login to HF with token login
from huggingface_hub import login
login(token='your-token', add_to_git_credential=True)

# Download relevant dataset
from datasets import load_dataset
beletrain = load_dataset('facebook/belebele',split='cat_Latn')
test_beletrain = load_dataset('homersimpson/beletrain-ca')

# Download model tokenizer
from transformers import AutoTokenizer
es_tokenizer = AutoTokenizer.from_pretrained(es_checkpoint)

# Set dict key names for answers, in accordance with the Belebele dataset format
answer_names = ["mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]
# Also set dict key names for answers in accordance with the format of test split of training dataset
test_answer_names = ["answer1", "answer2", "answer3", "answer4"]

def es_preprocess_function(examples):
    ''' Preprocesses and tokenized data, taking each possible answer and prepending them by the relevant passage and question. 

            Args: Entry in Belebele dataset.
            
            Returns: Tokenized entry, preprocessed such that each answer is prepended by its associated passage and question.
    '''
    contexts = [[context] * 4 for context in examples["flores_passage"]]
    questions = examples["question"]
    lablist = []
    for j in examples['correct_answer_num']:
      lablist.append(int(j)-1)
    label = lablist
    answered_questions = [
        [f"{header} {examples[end][i]}" for end in answer_names] for i, header in enumerate(questions)
    ]
    
    # Flatten, tokenize, and un-flatten again
    contexts = sum(contexts, [])
    answered_questions = sum(answered_questions, [])
    examples['label']=label
    tokenized_examples = es_tokenizer(contexts, answered_questions, max_length=512, truncation=True)

    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

# Tokenize Belebele dataset with model's tokenizer
es_tokenized_beletrain = beletrain.map(es_preprocess_function, batched=True)

def test_es_preprocess_function(examples):
    ''' Preprocesses and tokenized data, taking each possible answer and prepending them by the relevant passage and question. 

            Args: Entry in test split of training dataset.
            
            Returns: Tokenized entry, preprocessed such that each answer is prepended by its associated passage and question.
    '''
    contexts = [[context] * 4 for context in examples["passage"]]
    questions = examples["question"]
    lablist = []
    for j in examples['correct_answer_num']:
      lablist.append(int(j-1))
    label = lablist
    answered_questions = [
        [f"{header} {examples[end][i]}" for end in test_answer_names] for i, header in enumerate(questions)
    ]

    # Flatten, tokenize, and un-flatten again
    contexts = sum(contexts, [])
    answered_questions = sum(answered_questions, [])
    examples['label']=label
    tokenized_examples = es_tokenizer(contexts, answered_questions, max_length=512, truncation=True)

    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

# Tokenize test split of training dataset with model's tokenizer
es_tokenized_beletrain_test = test_beletrain.map(test_es_preprocess_function, batched=True)

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

# HF has no native data collator for multiple-choice tasks, so need to manually define one
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"

        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

# Load evaluation metric - in this case accuracy
import evaluate

accuracy = evaluate.load("accuracy")
import numpy as np

# Define the evaluation function, which in this case just calculates accuracy of model's predictions
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

# Load models
es_model = AutoModelForMultipleChoice.from_pretrained(es_checkpoint)

# Define training arguments for each model, create instance of Trainer class, and evaluate
es_training_args = TrainingArguments(
    output_dir="belebele/eval/es",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    num_train_epochs=4,
    weight_decay=0.01,
    push_to_hub=False,
)

es_trainer = Trainer(
    model=es_model,
    args=es_training_args,
    eval_dataset=es_tokenized_beletrain,
    tokenizer=es_tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=es_tokenizer),
    compute_metrics=compute_metrics,
)

# Print results of Belebele evaluation
print(es_trainer.evaluate())

# Define training arguments for each model, create instance of Trainer class, and evaluate
es_test_training_args = TrainingArguments(
    output_dir="belebele/eval/es/test",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    num_train_epochs=4,
    weight_decay=0.01,
    push_to_hub=False,
)

es_test_trainer = Trainer(
    model=es_model,
    args=es_training_args,
    eval_dataset=es_tokenized_beletrain_test['test'],
    tokenizer=es_tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=es_tokenizer),
    compute_metrics=compute_metrics,
)

# Print results of evaluation on test split of training dataset
print(es_test_trainer.evaluate())
