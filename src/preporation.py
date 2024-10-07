import os
import argparse
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

def prepare_dataset(dataset_name, tokenizer, max_length, output_dir):
    dataset = load_dataset(dataset_name, split='train')
    tokenizer.pad_token = tokenizer.eos_token
    os.makedirs(output_dir, exist_ok=True)
    tokenized_texts = []

    print("Tokenizing dataset...")
    for example in tqdm(dataset):
        tokens = tokenizer.encode(example['text'], truncation=True, max_length=max_length)
        tokenized_texts.append(tokens)

    # Save tokenized data
    tokenized_data_path = os.path.join(output_dir, 'tokenized_data.pt')
    torch.save(tokenized_texts, tokenized_data_path)
    print(f"Tokenized data saved to {tokenized_data_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for LLMAIX2001a.')
    parser.add_argument('--dataset_name', type=str, default='openwebtext', help='Name of the dataset to use.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length.')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save processed data.')
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prepare_dataset(args.dataset_name, tokenizer, args.max_length, args.output_dir)
