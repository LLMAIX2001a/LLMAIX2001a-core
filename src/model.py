import torch
import torch.nn as nn

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_heads=8, num_layers=6, dropout=0.1, max_length=512):
        super(TransformerLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        x = x.transpose(0, 1)  # Required by PyTorch transformer (seq_len, batch_size, embed_size)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        logits = self.fc_out(x)
        return logits

