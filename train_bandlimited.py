'''
Train hard bandlimited spectral attention with curriculum.
Goal: Train the lowpass_param to work with restricted frequency bins.
'''
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from spectral_attention import SpectralAttention, replace_attention_with_spectral, set_model_bandwidth, get_spectral_stats


CODING_SAMPLES = [
    'def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1',
    'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)',
    'def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b',
    'def is_palindrome(s):\n    cleaned = \"\".join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]',
    'def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []',
]


class CodingDataset(Dataset):
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = CODING_SAMPLES
        
    def __len__(self):
        return len(self.samples) * 20
        
    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        encoding = self.tokenizer(sample, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def train():
    print("="*60)
    print("HARD BANDLIMITED SPECTRAL TRAINING")
    print("="*60)
    
    # Config
    INITIAL_BW = 256
    FINAL_BW = 64
    EPOCHS = 3
    BATCH_SIZE = 2
    LR = 1e-3
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Replace attention
    print(f"\nReplacing attention (initial bw={INITIAL_BW})...")
    model = replace_attention_with_spectral(model, SpectralAttention, bandwidth_target=INITIAL_BW)
    
    # Freeze all except spectral params
    spectral_params = []
    for name, param in model.named_parameters():
        if 'lowpass' in name or 'sharpness' in name:
            param.requires_grad = True
            spectral_params.append(param)
        else:
            param.requires_grad = False
    
    print(f"Trainable params: {sum(p.numel() for p in spectral_params):,}")
    
    # Dataset & optimizer
    dataset = CodingDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(spectral_params, lr=LR)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=len(dataloader)*EPOCHS)
    
    # Curriculum schedule
    bw_schedule = [INITIAL_BW, 128, FINAL_BW]
    
    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        bw = bw_schedule[min(epoch, len(bw_schedule)-1)]
        set_model_bandwidth(model, bw)
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{EPOCHS} | Bandwidth: {bw}")
        print("="*60)
        
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['labels'].to("cuda")
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(spectral_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        stats = get_spectral_stats(model, bandwidth_target=FINAL_BW)
        print(f"Avg Loss: {avg_loss:.4f} | Leakage: {stats['leakage']:.4f}")
    
    # Final test
    print("\n" + "="*60)
    print("FINAL GENERATION TEST")
    print("="*60)
    model.eval()
    prompt = "def fibonacci(n):"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print(f"Prompt: {prompt}")
    print(f"Output:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    stats = get_spectral_stats(model, bandwidth_target=FINAL_BW)
    print(f"Final bandwidth: {FINAL_BW}")
    print(f"Final leakage: {stats['leakage']:.6f}")
    print("="*60)


if __name__ == "__main__":
    train()
