# Adapted from David Dale's tutorial at https://gist.github.com/avidale/44cd35bfcdaf8bedf51d97c468cc8001

#pip install transformers datasets evaluate sentencepiece -U accelerate -U transformers

# Login to HF hub
from datasets import load_dataset
from huggingface_hub import login
login(token='your-token', add_to_git_credential=True)

# Import base T5 model to be trimmed. 
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
model = T5ForConditionalGeneration.from_pretrained('google/mt5-base')

# Obtain original size of model
def msize(m):
    return sum(p.numel() for p in m.parameters())
original_size = msize(model)

# Download the Leipzig corpus of every language you wish to retain model embeddings for
!wget https://downloads.wortschatz-leipzig.de/corpora/spa_web_2016_1M.tar.gz
!tar -xsvf spa_web_2016_1M.tar.gz
!wget https://downloads.wortschatz-leipzig.de/corpora/fra-fr_web_2013_1M.tar.gz
!tar -xsvf fra-fr_web_2013_1M.tar.gz
!wget https://downloads.wortschatz-leipzig.de/corpora/ita-it_web-public_2019_1M.tar.gz
!tar -xsvf ita-it_web-public_2019_1M.tar.gz
!wget https://downloads.wortschatz-leipzig.de/corpora/por-pt_web_2015_1M.tar.gz
!tar -xsvf por-pt_web_2015_1M.tar.gz
!wget https://downloads.wortschatz-leipzig.de/corpora/cat_wikipedia_2021_1M.tar.gz
!tar -xsvf cat_wikipedia_2021_1M.tar.gz
!wget https://downloads.wortschatz-leipzig.de/corpora/glg_wikipedia_2021_300K.tar.gz
!tar -xsvf glg_wikipedia_2021_300K.tar.gz

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

cnt_ca = Counter()
for text in tqdm(df_ca.text):
    cnt_ca.update(tokenizer.encode(text))

cnt_gl = Counter()
for text in tqdm(df_gl.text):
    cnt_gl.update(tokenizer.encode(text))

print(len(cnt_es), len(cnt_es)/tokenizer.vocab_size)
print(len(cnt_it), len(cnt_it)/tokenizer.vocab_size)
print(len(cnt_fr), len(cnt_fr)/tokenizer.vocab_size)
print(len(cnt_pt), len(cnt_pt)/tokenizer.vocab_size)
print(len(cnt_ca), len(cnt_ca)/tokenizer.vocab_size)
print(len(cnt_gl), len(cnt_gl)/tokenizer.vocab_size)

# Count how many tokens are common to each language, and print
common = len(set(cnt_es.keys()).intersection(set(cnt_it.keys())).intersection(set(cnt_fr.keys())).intersection(set(cnt_pt.keys())).intersection(set(cnt_ca.keys())).intersection(set(cnt_gl.keys())))
print(common, common / len(cnt_es))

# Print the span of all tokens that the top 10/20/30K tokens encompass for each language
print('es')
for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_es.most_common(top)) / sum(cnt_es.values()))
print('it')
for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_it.most_common(top)) / sum(cnt_it.values()))
print('fr')
for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_fr.most_common(top)) / sum(cnt_fr.values()))
print('pt')
for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_pt.most_common(top)) / sum(cnt_pt.values()))
print('ca')
for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_ca.most_common(top)) / sum(cnt_ca.values()))
print('gl')
for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_gl.most_common(top)) / sum(cnt_gl.values()))

# Obtain current model vocabulary
old_voc = tokenizer.get_vocab()
old_inv_voc = {v: k for k, v in old_voc.items()}

# Retain top 1000 tokens of old vocabulary as a contingency
new_tokens = set(range(1000))

# Add top 30,000 tokens of each language to new vocabulary
for i, (k, v) in enumerate(cnt_es.most_common(30_000)):
    if k not in new_tokens:
        new_tokens.add(k)
print(i, 'Spanish tokens are included')
for i, (k, v) in enumerate(cnt_it.most_common(30_000)):
    if k not in new_tokens:
        new_tokens.add(k)
print(i, 'Italian tokens are included')
for i, (k, v) in enumerate(cnt_fr.most_common(30_000)):
    if k not in new_tokens:
        new_tokens.add(k)
print(i, 'French tokens are included')
for i, (k, v) in enumerate(cnt_pt.most_common(30_000)):
    if k not in new_tokens:
        new_tokens.add(k)
print(i, 'Portuguese tokens are included')
for i, (k, v) in enumerate(cnt_ca.most_common(30_000)):
    if k not in new_tokens:
        new_tokens.add(k)
print(i, 'Catalan tokens are included')
for i, (k, v) in enumerate(cnt_gl.most_common(30_000)):
    if k not in new_tokens:
        new_tokens.add(k)
print(i, 'Galician tokens are included')
for t in range(tokenizer.vocab_size - 100, tokenizer.vocab_size):
    new_tokens.add(t)

# Print size of new vocabulary, and sort them
print(len(new_tokens))
kept_ids = sorted(new_tokens)

# Update the model embeddings and LM head to match the new vocabulary
import torch
new_size = len(kept_ids)
new_emb = torch.nn.Embedding(new_size, model.shared.embedding_dim)
new_head = torch.nn.Linear(in_features=model.lm_head.in_features, out_features=new_size, bias=False)

# Takes data from old model, but only that which we have chosen to keep
for new_id, old_id in enumerate(kept_ids):
    new_emb.weight.data[new_id] = model.shared.weight.data[old_id]
    new_head.weight.data[new_id] = model.lm_head.weight.data[old_id]

model.shared.weight = new_emb.weight
model.lm_head.weight = new_head.weight

# Print size of new model
print(msize(model), msize(model) / original_size)

# Update the T5 tokenizer - need to deploy into Python manually
!wget https://raw.githubusercontent.com/google/sentencepiece/master/src/sentencepiece_model.proto
! protoc --python_out=. sentencepiece_model.proto

import sentencepiece_model_pb2 as spmp
smp = tokenizer.sp_model.serialized_model_proto()
m = spmp.ModelProto()
m.ParseFromString(smp)

new_pieces = [m.pieces[idx] for idx in kept_ids]

# Replace the content of the first 30K pieces
for i, p in enumerate(new_pieces):
    m.pieces[i].piece = p.piece
    m.pieces[i].score = p.score
    m.pieces[i].type = p.type

# Drop the remaining pieces
n = len(new_pieces)
for i in trange(len(m.pieces) - n):
    m.pieces.pop(len(m.pieces) - 1)
with open('new_sp.model', 'wb') as f:
    f.write(m.SerializeToString())

# Save new tokenizer
new_tokenizer = T5Tokenizer('new_sp.model', extra_ids=0)

# Update model configs
model.config.__dict__['vocab_size'] = new_size
model.config.__dict__['_name_or_path'] = 'subsec-t5-italo-western'

# Save model and tokenizer, load again for posterity
new_tokenizer.save_pretrained('subsec-t5-italo-western-ca-gl')
model.save_pretrained('subsec-t5-italo-western-ca-gl')
model1 = T5ForConditionalGeneration.from_pretrained('subsec-t5-italo-western-ca-gl')
tokenizer1 = T5Tokenizer.from_pretrained('subsec-t5-italo-western-ca-gl')

# Push to HF
model1.push_to_hub('yourname/yourrepo')
tokenizer1.push_to_hub('yourname/yourrepo')
