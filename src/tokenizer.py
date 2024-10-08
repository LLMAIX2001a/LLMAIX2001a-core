from transformers import GPT2Tokenizer

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Ensure tokenizer has a pad token
    return tokenizer

