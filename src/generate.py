import torch
import torch.nn.functional as F
import argparse

from model import TransformerLanguageModel
from tokenizer import get_tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=0, top_p=0.9):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated = input_ids
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 0] = 0
        for i in range(logits.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i, indices_to_remove] = -float('Inf')
    return logits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using the LLMAIX2001a model.')
    parser.add_argument('--model_path', type=str, help='Path to the trained model file.')
    parser.add_argument('--prompt', type=str, default='', help='Input prompt for text generation.')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text.')
    args = parser.parse_args()
    tokenizer = get_tokenizer()
    model = TransformerLanguageModel(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(args.model_path))
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    generated_text = generate_text(model, tokenizer, args.prompt, max_length=args.max_length)
    print(generated_text)

