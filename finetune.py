import json
import os
import urllib.request
import torch
from torch.utils.data import Dataset
import tiktoken
from functools import partial
from torch.utils.data import DataLoader
from generation import generate, text_to_token_ids, token_ids_to_text
from pretrain import calc_loss_loader, train_model_simple, load_weights_into_gpt
import time
from gpt_download import download_and_load_gpt2
from model import GPTModel


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_data)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data = file.read()
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def format_input(entry):
    instruction_text=(
        f"User:\n{entry['instruction']} {(entry['input'] if entry['input'] else '')}\nAssistant:\n"
    )
    return instruction_text


def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=100,
        allowed_max_length=None,
        device='cpu'
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
                new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = entry['output']
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    file_path = 'instruction-data.json'

    url = (
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json"
    )

    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))

    train_portion = int(len(data)*0.85)
    test_portion = int(len(data)*0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion+test_portion]
    val_data = data[train_portion+test_portion:]

    print('Training set length:', len(train_data))
    print('Testing set length:', len(test_data))
    print('Validation set length:', len(val_data))

    tokenizer = tiktoken.get_encoding("gpt2")
    eos_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
    print(eos_id)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print('Device:', device)

    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    num_workers = 0
    batch_size = 4

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }

    model_configs = {
        'gpt2-small (124M)': {'emb_dim': 768, 'n_layers': 12, 'n_heads': 12},
        'gpt2-medium (355M)': {'emb_dim': 1024, 'n_layers': 24, 'n_heads': 16},
        'gpt2-large (774M)': {'emb_dim': 1280, 'n_layers': 36, 'n_heads': 20},
        'gpt2-xl (1558M)': {'emb_dim': 1600, 'n_layers': 48, 'n_heads': 25}
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip('(').rstrip(')')
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir='gpt2'
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=5
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=5
        )

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    start_time = time.time()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1
    )

    num_epochs = 2

    if input('train? ') == 'y':
        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=5, eval_iter=5,
            start_context=format_input(val_data[0]), tokenizer=tokenizer
        )
        end_time = time.time()
        execution_time_minutes = (end_time-start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")

        model.to('cpu')
        file_name_model = input('file name? ')
        torch.save(model.state_dict(), f"{file_name_model}.pth")

        from ch5 import plot_losses
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    else:
        x = input('file name? ')
        model.load_state_dict(torch.load(f"{x}.pth", weights_only=True))
        model.to(device)
        question = input('User: ')
        additional = input('User: ')
        if additional == '':
            additional = None
        entry = {
            'instruction': question,
            'input': additional,
            'output': ''
        }
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):]
        print(input_text, end='')
        print(response_text)