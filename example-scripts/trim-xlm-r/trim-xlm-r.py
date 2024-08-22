# Adapted from David Dale's code at https://colab.research.google.com/drive/1f-n3zBQjmtMrp7oHzvunHPSC5aIMNe_N?usp=sharing

# pip install transformers datasets evaluate -U accelerate -U transformers sentencepiece -q

# Login to HF hub
from datasets import load_dataset
from huggingface_hub import login
login(token='your-token', add_to_git_credential=True)

# Obtain XLM-R model and tokenizer
import torch
from transformers import AutoModelForMaskedLM, XLMRobertaTokenizer
model_name = 'xlm-roberta-base'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

# Download the Leipzig corpus of every language you wish to retain model embeddings for
import re
import string
!wget http://pcai056.informatik.uni-leipzig.de/downloads/corpora/por-pt_web_2015_1M.tar.gz
!tar -xsvf por-pt_web_2015_1M.tar.gz
!wget http://pcai056.informatik.uni-leipzig.de/downloads/corpora/fra-fr_web_2013_1M.tar.gz
!tar -xsvf fra-fr_web_2013_1M.tar.gz
!wget http://pcai056.informatik.uni-leipzig.de/downloads/corpora/ita-it_web-public_2019_1M.tar.gz
!tar -xsvf ita-it_web-public_2019_1M.tar.gz
!wget http://pcai056.informatik.uni-leipzig.de/downloads/corpora/spa_web_2016_1M.tar.gz
!tar -xsvf spa_web_2016_1M.tar.gz

# Count how many tokens in the model's tokenizer are used with each language, then print
from collections import Counter
from tqdm.auto import tqdm, trange

cnt_es = Counter()
for text in tqdm(df_es.text):
    cnt_es.update(tokenizer.encode(text))

cnt_it = Counter()
for text in tqdm(df_it.text):
    cnt_it.update(tokenizer.encode(text))

cnt_fr = Counter()
for text in tqdm(df_fr.text):
    cnt_fr.update(tokenizer.encode(text))

cnt_pt = Counter()
for text in tqdm(df_pt.text):
    cnt_pt.update(tokenizer.encode(text))

print(len(cnt_es), len(cnt_es)/tokenizer.vocab_size)
print(len(cnt_it), len(cnt_it)/tokenizer.vocab_size)
print(len(cnt_fr), len(cnt_fr)/tokenizer.vocab_size)
print(len(cnt_pt), len(cnt_pt)/tokenizer.vocab_size)

# Obtain current model vocabulary
old_voc = tokenizer.get_vocab()
old_inv_voc = {v: k for k, v in old_voc.items()}

# Obtain special tokens
special_tokens = set(tokenizer.special_tokens_map.values())

# Obtain top 30K tokens of each language in tokenizer
kept_ids = []
kept_tokens = []
thirtyk_pt = []
thirtyk_fr = []
thirtyk_it = []
thirtyk_es = []

for i, (k,v) in enumerate(cnt_pt.most_common(30_000)):
  thirtyk_pt.append(tokenizer.convert_ids_to_tokens(k))
thirtyk_tokens_pt = set(thirtyk_pt)

for i, (k,v) in enumerate(cnt_fr.most_common(30_000)):
  thirtyk_fr.append(tokenizer.convert_ids_to_tokens(k))
thirtyk_tokens_fr = set(thirtyk_fr)

for i, (k,v) in enumerate(cnt_it.most_common(30_000)):
  thirtyk_it.append(tokenizer.convert_ids_to_tokens(k))
thirtyk_tokens_it = set(thirtyk_it)

for i, (k,v) in enumerate(cnt_es.most_common(30_000)):
  thirtyk_es.append(tokenizer.convert_ids_to_tokens(k))
thirtyk_tokens_es = set(thirtyk_es)

# Create the new vocabulary
for i, token in enumerate(tokenizer.convert_ids_to_tokens(range(len(tokenizer)))):
    if token in special_tokens or token in thirtyk_tokens_pt or token in thirtyk_tokens_fr or token in thirtyk_tokens_it or token in thirtyk_tokens_es:
         kept_tokens.append(token)
         kept_ids.append(i)
print(len(kept_tokens))
print(len(kept_ids))

# Create a temporary SentencePiece tokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
tokenizer.save_pretrained('tmp_tokenizer')

m = sp_pb2_model.ModelProto()
with open("tmp_tokenizer/sentencepiece.bpe.model", "rb") as f:
    m.ParseFromString(f.read())

# Update to reflect new vocabulary (pop all vocab not in that which is kept), then write this new tokenizer
from tqdm.auto import trange
TOKENS_TO_KEEP_SET = set(kept_tokens)
for i in trange(len(tokenizer)-3, 0, -1):
    if m.pieces[i].piece not in TOKENS_TO_KEEP_SET:
        m.pieces.pop(i)

with open("tmp_tokenizer/sentencepiece.bpe.model", 'wb') as f:
    f.write(m.SerializeToString())

# Check this process has worked
assert len(tokenizer_new) == len(kept_tokens)
assert kept_tokens == tokenizer_new.convert_ids_to_tokens(range(len(tokenizer_new)))

# Create a new copy of original model, and update it to reflect new vocabulary
model_new = AutoModelForMaskedLM.from_pretrained(model_name)
model_new.roberta.embeddings.word_embeddings.weight.data = model.roberta.embeddings.word_embeddings.weight.data[kept_ids]
model_new.lm_head.decoder.bias.data = model.lm_head.decoder.bias.data[kept_ids]
model_new.lm_head.decoder.bias.data = model.lm_head.decoder.bias.data[kept_ids]

# Save new model and tokenizer locally
model_new.save_pretrained("tmp2")
tokenizer_new.save_pretrained("tmp2")

# Load for posterity and then push to HF hub
model_new = AutoModelForMaskedLM.from_pretrained("tmp2")
tokenizer_new = XLMRobertaTokenizer.from_pretrained('tmp2')

model_new.push_to_hub('yourname/yourrepo')
tokenizer_new.push_to_hub('yourname/yourrepo')
