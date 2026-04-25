# Translator
I have implemented a Transformer based English to German Translator.
The model is trained on the Tab-delimited Bilingual Sentence Pairs dataset. These are selected sentence pairs from the Tatoeba Project.
The dataset can be found at http://www.manythings.org/anki/

We have used the model architecture presented in the paper " Attention is all you need! " by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin ([Submitted on 12 Jun 2017 (v1), last revised 2 Aug 2023 (this version, v7)])



The model architecture has been built from scratch using PyTorch and trained on Kaggle GPU P100 for 10 epochs. The training time took over 5 hours 15 minutes on the above dataset of size 277451 sentence pairs.

The model achieved a BLEU score of 53.58 evaluated on a test dataset of 10k sentence pairs.

The model architecture is defined in the model.py file
The data processing is defined in the dataset.py file
The training loop is defined in the train.py file



# Translater-TRANSFORMER-L2L-
