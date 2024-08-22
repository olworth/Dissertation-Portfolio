# Login to HF
from huggingface_hub import login
login(token='your-token', add_to_git_credential=True)

# Import HF dataset to be translated
from datasets import load_dataset
beletrain_en = load_dataset('yourname/yourHFdataset')

# Load CTranslate2 implementation of NLLB-200-distilled-1.2B from local
import ctranslate2
import sentencepiece as spm

ct_model_path = "ct2-nllb-200-distilled-1.2B-int8"
sp_model_path = "flores200_sacrebleu_tokenizer_spm.model"
device = "cuda"

sp = spm.SentencePieceProcessor()
sp.load(sp_model_path)

translator = ctranslate2.Translator(ct_model_path, device)

beam_size = 4

# Initialise source and target languages
src_lang = "eng_Latn"
tgt_lang = "glg_Latn"

import re

def allinone(tessents):
  ''' Function to be mapped to all entries in target dataset.
      Takes a batch of the dataset, and translates it in an efficient manner. "tessents" is derived from "test sentences"; the name stuck like glue in my head.

        Args: a portion of a dataset to be translated

        Returns: said portion, translated
  '''
  # In order to efficiently translate the (very large) dataset, it is necessary to first organise it into an array of sentences to be translated.
  # However, doing so destroys the structure of the dataset.
  # As such, I retain the structure of the dataset by creating a dictionary, mapping the index of each sentence to that of its "passage" - its key in the dataset
  map_dict = {}
  passage_index=0
  sent_index=0
  source_sentences = []

  for each in tessents['question']:
    source_sentences.append(each)
    sent_index+=1
  map_dict.update({passage_index:sent_index})
  passage_index+=1

  for each in tessents['mc_answer1']:
    source_sentences.append(each)
    sent_index+=1
  map_dict.update({passage_index:sent_index})
  passage_index+=1

  for each in tessents['mc_answer2']:
    source_sentences.append(each)
    sent_index+=1
  map_dict.update({passage_index:sent_index})
  passage_index+=1

  for each in tessents['mc_answer3']:
    source_sentences.append(each)
    sent_index+=1
  map_dict.update({passage_index:sent_index})
  passage_index+=1

  for each in tessents['mc_answer4']:
    source_sentences.append(each)
    sent_index+=1
  map_dict.update({passage_index:sent_index})
  passage_index+=1

  # Create target language prefixes, as per the model's specifications
  target_prefix = [[tgt_lang]] * len(source_sentences)

  # Preprocess and format source sentences
  source_sentences = [sent.strip() for sent in source_sentences]
  source_sents_subworded = sp.encode_as_pieces(source_sentences)
  source_sents_subworded = [[src_lang] + sent + ["</s>"] for sent in source_sents_subworded]

  # Pass preprocessed sentences to the model, recieve their translations, and unpack and decode them
  translator = ctranslate2.Translator(ct_model_path, device=device)
  translations_subworded = translator.translate_batch(source_sents_subworded, batch_type="tokens", max_batch_size=2000, beam_size=beam_size, target_prefix=target_prefix)
  translations_subworded = [translation.hypotheses[0] for translation in translations_subworded]
  for translation in translations_subworded:
    if tgt_lang in translation:
      translation.remove(tgt_lang)

  translations = sp.decode(translations_subworded)

  # Restructure the original dataset with translations, utilising the mapping dictionary created earlier
  tessents['question']=translations[:map_dict[0]]
  tessents['mc_answer1']=translations[map_dict[0]:map_dict[1]]
  tessents['mc_answer2']=translations[map_dict[1]:map_dict[2]]
  tessents['mc_answer3']=translations[map_dict[2]:map_dict[3]]
  tessents['mc_answer4']=translations[map_dict[3]:map_dict[4]]
  
  return tessents

beletrain_gl=beletrain_en.map(allinone, batched=True)

# Push new dataset to HF hub
beletrain_gl.push_to_hub('yourname/yourdataset')
