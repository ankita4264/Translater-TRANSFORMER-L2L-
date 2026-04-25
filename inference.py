import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from pathlib import Path
from model import build_transformer  # Assuming you have a function to build your Transformer model

MAX_LENGTH = 115

# Function to load tokenizers
def load_tokenizers():
    tokenizer_src = Tokenizer.from_file("tokenizer_English.json")
    tokenizer_tgt = Tokenizer.from_file("tokenizer_German.json")
    return tokenizer_src, tokenizer_tgt

# Load tokenizers
tokenizer_src, tokenizer_tgt = load_tokenizers()

# Function to load Transformer model
def load_transformer_model(model_path):
    checkpoint = torch.load(model_path)

    # Create a new instance of the Transformer model
    transformer = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), src_seq_len=MAX_LENGTH, tgt_seq_len=MAX_LENGTH, dim_model=512)

    # Load the state dictionary into the model
    transformer.load_state_dict(checkpoint['model_state_dict'])

    # Optional: Load other components like optimizer state, epoch, etc., if needed
    epoch = checkpoint['epoch']
    print(epoch)
    global_step = checkpoint['global_step']
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return transformer, epoch, global_step

# Load trained model
model_path = "C:\\Users\\sharm\\OneDrive\\Desktop\\Translator\\translator_model"  # Adjust the path to your model file
model, epoch, global_step = load_transformer_model(model_path)

# Example English sentence for translation
english_sentence =  ""

# Function to translate a sentence
def translate_sentence(model, tokenizer_src, tokenizer_tgt, english_sentence):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    sos_token_id = tokenizer_tgt.token_to_id("[START]")
    eos_token_id = tokenizer_tgt.token_to_id("[END]")
    pad_token_id = tokenizer_tgt.token_to_id("[PAD]")

    # Encode the source sentence with the source tokenizer
    src_ids = tokenizer_src.encode(english_sentence).ids
    enc_num_padding_tokens = MAX_LENGTH - len(src_ids) - 2
    encoder_input = torch.cat([
        torch.tensor([sos_token_id], dtype=torch.int64),
        torch.tensor(src_ids, dtype=torch.int64),
        torch.tensor([eos_token_id], dtype=torch.int64),
        torch.tensor([pad_token_id] * enc_num_padding_tokens, dtype=torch.int64),
    ], dim=0).to(device)
    encoder_mask = (encoder_input != pad_token_id).unsqueeze(0).unsqueeze(0).int()

    tgt_token_ids = [sos_token_id]
    tgt_tensor = torch.tensor([tgt_token_ids], dtype=torch.int64).to(device)

    model.to(device)
    model.eval()
    
    with torch.no_grad():
        encoder_output = model.encode(encoder_input.unsqueeze(0), encoder_mask)
        for _ in range(MAX_LENGTH):
            tgt_mask = (tgt_tensor != pad_token_id).unsqueeze(0).int() & decoder_mask(tgt_tensor.size(1)).to(device)
            decoder_output = model.decode(encoder_output, encoder_mask, tgt_tensor, tgt_mask)
            projection = model.projection_layer(decoder_output)
            next_token_id = torch.argmax(projection[:, -1, :], dim=-1).item()
            tgt_token_ids.append(next_token_id)
            if next_token_id == eos_token_id:
                break
            tgt_tensor = torch.tensor([tgt_token_ids], dtype=torch.int64).to(device)

    translated_tokens = tgt_token_ids[1:]  # Remove the start token
    translated_sentence = tokenizer_tgt.decode(translated_tokens)

    return translated_sentence

def decoder_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

# Perform translation
translated_sentence = translate_sentence(model, tokenizer_src, tokenizer_tgt, english_sentence)
print(f"English: {english_sentence}")
print(f"German: {translated_sentence}")
