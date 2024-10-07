import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
import argparse
import json
import os

from model import TransformerLanguageModel
from tokenizer import get_tokenizer

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer()
    model = TransformerLanguageModel(vocab_size=tokenizer.vocab_size, **config['model']).to(device)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=config['training']['max_length'])
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(tokenized_datasets, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=data_collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config['training']['epochs']):
        loop = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    os.makedirs(os.path.dirname(config['training']['output_model_path']), exist_ok=True)
    torch.save(model.state_dict(), config['training']['output_model_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the LLMAIX2001a model.')
    parser.add_argument('--config', type=str, help='Path to the training config file.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    main(config)

