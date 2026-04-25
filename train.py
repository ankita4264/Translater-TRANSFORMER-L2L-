import torch
import torch.nn as nn
from model import build_transformer
import dataset
from tqdm import tqdm
from pathlib import Path

def get_model(vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, src_seq_len=100, tgt_seq_len=100, dim_model=512)
    return model

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    
    device = torch.device(device)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = dataset.get_dataset()

    model = get_model(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    global_step = 0

    for epoch in range(20):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)    # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)    # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)      # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)      # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder, and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)         # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.projection_layer(decoder_output)               # (B, seq_len, vocab_size)

            # Adjust label to match proj_output sequence length
            max_seq_len = min(proj_output.size(1), batch['label'].size(1))
            label = batch['label'][:, :max_seq_len].to(device)  # Trim to match proj_output length

            proj_output_flat = proj_output[:, :max_seq_len, :].contiguous().view(-1, tokenizer_tgt.get_vocab_size())
            label_flat = label.contiguous().view(-1)

            loss = loss_fn(proj_output_flat, label_flat)

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

    model_filename = Path("translator_model")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)

if __name__ == '__main__':
    train_model()
