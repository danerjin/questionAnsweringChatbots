from model import GPTModel, replace_linear_with_lora, Embedding
from transformers import AutoTokenizer, AutoModel
from gpt_download import download_and_load_gpt2
from pretrain import load_weights_into_gpt
from finetune import format_input
import torch
from generation import text_to_token_ids, token_ids_to_text, text_to_token_ids_batch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import time
import os
from tqdm import tqdm
from datasets import load_dataset


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling_minilm(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def tokenize_dataset(ds):
    ds['sentence1'] = text_to_token_ids_batch(
        ds['sentence1'], tokenizer, BASE_CONFIG['context_length'], eos_id)
    ds['sentence2'] = text_to_token_ids_batch(
        ds['sentence2'], tokenizer, BASE_CONFIG['context_length'], eos_id)
    return ds


def calc_loss_batch(input_batch, target_batch, score_batch, model, device, eos_id, loss_calc, cos):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    score_batch = score_batch.to(device)
    att_1 = torch.where(input_batch == eos_id, 0, 1).to(device)
    att_2 = torch.where(target_batch == eos_id, 0, 1).to(device)

    input_embed = model(input_batch)
    input_embed = mean_pooling(input_embed, att_1)

    target_embed = model(target_batch)
    target_embed = mean_pooling(target_embed, att_2)

    score_calc = cos(input_embed, target_embed)
    loss = loss_calc(score_calc, score_batch)
    return loss


def evaluate_model(model, train_loader, val_loader, device, eos_id, loss_calc, cos, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, eos_id, loss_calc, cos, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, eos_id, loss_calc, cos, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


def calc_loss_loader(data_loader, model, device, eos_id, loss_calc, cos, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, entry in enumerate(data_loader):
        if i < num_batches:
            input_batch = entry['sentence1']
            target_batch = entry['sentence2']
            score_batch = entry['score']
            loss = calc_loss_batch(
                input_batch, target_batch, score_batch, model, device, eos_id, loss_calc, cos
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches



def train_embedding_simple(
        model, train_loader, val_loader, optimizer, device, eos_id, loss_calc,
        num_epochs, eval_freq, eval_iter):
    train_losses, val_losses = [], []
    examples_seen, global_step = 0, -1
    print('TRAINING START')
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for epoch in range(num_epochs):
        model.train()

        for entry in train_loader:
            input_batch = entry['sentence1']
            target_batch = entry['sentence2']
            score_batch = entry['score']
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, score_batch, model, device, eos_id, loss_calc, cos
            )
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eos_id, loss_calc, cos, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}/{len(train_loader):06d})"
                      f" Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

    return train_losses, val_losses, examples_seen


if __name__ == '__main__':
    first_few = 10

    tokenizer = tiktoken.get_encoding("gpt2")
    eos_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    print(eos_id)
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

    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model = Embedding(BASE_CONFIG)
    a = input('file name? ')
    model.load_state_dict(torch.load(f"{a}.pth", weights_only=True))

    ds = load_dataset("sentence-transformers/stsb")
    ds.set_format("torch")

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    for i in range(first_few):
        entry = ds['train'][i]
        a = text_to_token_ids(entry['sentence1'], tokenizer)
        b = text_to_token_ids(entry['sentence2'], tokenizer)
        amask = torch.where(a == eos_id, 0, 1)
        bmask = torch.where(b == eos_id, 0, 1)
        with torch.no_grad():
            x = model(a)
            y = model(b)
        x = mean_pooling(x, amask)
        y = mean_pooling(y, bmask)
        score = cos(x, y)
        print(entry['sentence1'])
        print(entry['sentence2'])
        print(score.data[0], entry['score'])
        print('--------------------------------------')

if __name__ == '__main__' and False:
    if input("use MiniLM-L6-v2? ") == 'y':
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        # Sentences we want sentence embeddings for
        sentences = [input(), input()]

        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        # print("Sentence embeddings:")
        # print(sentence_embeddings)
        print("Correlation: " + str(torch.dot(sentence_embeddings[0], sentence_embeddings[1])))
        print(sentence_embeddings[0].shape)
    else:
        tokenizer = tiktoken.get_encoding("gpt2")
        eos_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        print(eos_id)

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            device = torch.device('mps')

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

        CHOOSE_MODEL = "gpt2-small (124M)"
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip('(').rstrip(')')

        model = Embedding(BASE_CONFIG)
        is_train = input("train? ") == "y"
        if is_train:
            settings, params = download_and_load_gpt2(
                model_size=model_size,
                models_dir='gpt2'
            )
            load_weights_into_gpt(model, params)
            model.eval()

            for param in model.parameters():
                param.requires_grad = False

            for param in model.trf_blocks[-1].parameters():
                param.requires_grad = True
            for param in model.final_norm.parameters():
                param.requires_grad = True
            for param in model.out_head.parameters():
                param.requires_grad = True

            # replace_linear_with_lora(model, rank=16, alpha=16)

            model.to(device)

            ds = load_dataset("sentence-transformers/stsb")

            ds = ds.map(tokenize_dataset)
            ds.set_format("torch")
            print(ds['train']['sentence1'][0].shape)
            print(ds['train']['sentence1'][0])
            train_loader = DataLoader(ds["train"], batch_size=8)
            val_loader = DataLoader(ds["validation"], batch_size=8)
            num_epochs = 10
            start_time = time.time()
            loss_calc = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

            train_losses, val_losses, examples_seen = \
                train_embedding_simple(
                    model, train_loader, val_loader, optimizer, device, eos_id, loss_calc,
                    num_epochs=num_epochs, eval_freq=50,
                    eval_iter=5
                )

            end_time = time.time()
            execution_time_minutes = (end_time - start_time) / 60
            print(f"Training completed in {execution_time_minutes:.2f} minutes.")

            model.to('cpu')
            file_name_model = input('file name? ')
            torch.save(model.state_dict(), f"{file_name_model}.pth")
        else:
            a = input('file name? ')
            model.load_state_dict(torch.load(f"{a}.pth", weights_only=True))
            model.to(device)
        while 1:
            text1 = input()
            if text1 == 'q':
                break
            text2 = input()
            text1 = text_to_token_ids(text1, tokenizer)
            text2 = text_to_token_ids(text2, tokenizer)

            input_batch = text1.to(device)
            target_batch = text2.to(device)
            att_1 = torch.where(input_batch == eos_id, 0, 1).to(device)
            att_2 = torch.where(target_batch == eos_id, 0, 1).to(device)
            with torch.no_grad():
                input_embed = model(input_batch)
                input_embed = mean_pooling(input_embed, att_1)
                input_embed = torch.nn.functional.normalize(input_embed, p=2, dim=1)

                target_embed = model(target_batch)
                target_embed = mean_pooling(target_embed, att_2)
                target_embed = torch.nn.functional.normalize(target_embed, p=2, dim=1)

            score_calc = (input_embed * target_embed).sum(dim=1)
            print(score_calc)