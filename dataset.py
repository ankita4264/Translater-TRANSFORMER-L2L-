import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.utils.data import random_split
import re


def get_all_sentences(ds, lang):
    return ds[lang]

def build_or_get_tokenizer(ds, lang):
    tokenizer_path = Path(f"tokenizer_{lang}.json")

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[START]", "[END]", "[PAD]"])
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_dataset():
    file_path = "D:\\eng-ger.txt"
    dataset = pd.read_csv(file_path, delimiter='\t', usecols=[0, 1], names=["English", "German"])

    # Adding an index column
    dataset.reset_index(inplace=True)

    print(len(dataset))

    tokenizer_src = build_or_get_tokenizer(dataset, lang="English")
    tokenizer_tgt = build_or_get_tokenizer(dataset, lang="German")

    train_ds_size = int(0.75 * len(dataset))
    val_ds_size = len(dataset) - train_ds_size
    train_indices, val_indices = random_split(dataset.index.tolist(), [train_ds_size, val_ds_size])

    train_ds = TranslationDataset(dataset, train_indices, tokenizer_src, tokenizer_tgt, src_lang="English", tgt_lang="German", seq_len=115)
    val_ds = TranslationDataset(dataset, val_indices, tokenizer_src, tokenizer_tgt, src_lang="English", tgt_lang="German", seq_len=115)

    max_len_src = 0
    max_len_tgt = 0

    for idx in range(len(dataset)):
        src_text = dataset['English'].iloc[idx]
        tgt_text = dataset['German'].iloc[idx]

        src_ids = tokenizer_src.encode(src_text).ids
        tgt_ids = tokenizer_tgt.encode(tgt_text).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=32, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

class TranslationDataset(Dataset):
    def __init__(self, data, indices, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.data = data
        self.indices = indices
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[START]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[END]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.data.iloc[real_idx]
        src_text = row['English']
        tgt_text = row['German']

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
