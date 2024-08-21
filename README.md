# Dissertation Portfolio

This repo serves as a portfolio for the code I wrote for my dissertation, _Exploring and Understanding Cross-Lingual Transfer to Catalan and Galician via High-Resource Typological Relatives_. 

The core of the project was an investigation into NLP tasks in Catalan and Galician, two low-resource languages that lack the vast amounts of textual data necessary to pre-train and fine-tune good LLMs. By leveraging textual data from higher-resource languages in a transfer learning paradigm, however, (for example, pre-training a model in Spanish and fine-tuning it in Catalan) performance can be improved. My results indicated that this improvement was generally greatest when data from multiple languages were used in pre-training; the best combination of languages to use, however, depended on the nature of the NLP task at hand and of each language's relationship with the target language. A particularly pertinent example is that of my Dependency Parsing models: whereas Galician models preferred a small number of closely-related languages, their Catalan equivalents benefitted substantially more from a massively multilingual pre-training, spanning over a hundred languages. I theorised that this was the result of Galician morphology and syntax sitting in more of a "sweet spot" of similarity in relation to its close typological relatives, and attempted to quantify this.

The example scripts contained in this repository are as follows:



If you would like to play around with some of the models I created, click [here](https://huggingface.co/homersimpson) for my HuggingFace account. HuggingFace's Inference API should facilitate some limited interaction with them, but if not they can be freely downloaded. If you would like to read my dissertation in its entirety, please contact me and I will provide access!
