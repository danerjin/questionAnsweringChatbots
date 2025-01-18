from model import GPTModel, replace_linear_with_lora, Embedding
from gpt_download import download_and_load_gpt2
from pretrain import load_weights_into_gpt
import torch
from generation import generate, generate_text_simple, text_to_token_ids, token_ids_to_text, text_to_token_ids_batch, text_to_token_ids_batch_total
from finetune import format_input
from torch.utils.data import Dataset, DataLoader
from embedding import mean_pooling, mean_pooling_minilm
from transformers import AutoTokenizer, AutoModel
import tiktoken
import time
import os


def divide_data(txt, tokenizer, max_length, stride):
    tokenized = []
    token_ids = tokenizer.encode(txt)
    for i in range(0, len(token_ids) - max_length, stride):
        input_chunk = token_ids[i:i+max_length]
        tokenized.append(input_chunk)
    return torch.tensor(tokenized)


class Retriever(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.W_key = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_query = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        with torch.no_grad():
            self.W_key.weight.fill_(0)
            self.W_key.weight.fill_diagonal_(1)
            self.W_query.weight.fill_(0)
            self.W_query.weight.fill_diagonal_(1)
        self.emb_dim = emb_dim

    def forward(self, query, documents):
        b, d_in = query.shape
        b2, num_docs, d_in2 = documents.shape
        assert b == b2 and d_in == d_in2 == self.emb_dim
        query = self.W_query(query)
        keys = self.W_key(documents)
        query = query.view(b, self.emb_dim)
        keys = keys.view(b, num_docs, self.emb_dim)

        attn_scores = torch.abs(query @ keys.transpose(1, 2))
        return attn_scores


class DocumentsDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def train_rag():
    pass


# have fun with the "virginia declaration of rights" (toy example)
if __name__ == "__main__":
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

    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    llm = GPTModel(BASE_CONFIG)
    llm.eval()
    x = input('File name for LLM chatbot? ')
    llm.load_state_dict(torch.load(f"{x}.pth", weights_only=True))
    llm.to(device)

    retriever = Retriever(384)
    divide_by_line = input("Divide by line? ") == 'y'
    with open('va_declaration.txt', 'r') as file:
        if divide_by_line:
            text_data = file.readlines()
            text_data = list(map(lambda x: x.strip('\n'), text_data))
            text_data = list(filter(lambda x: x != '', text_data))
        else:
            text_data = file.read()
    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    top_k = 1
    use_minilm = input("use MiniLM-L6-v2? ") == 'y'
    if not use_minilm:
        embed = Embedding(BASE_CONFIG)
        a = input('file name for embedding model? ')
        embed.load_state_dict(torch.load(f"{a}.pth", weights_only=True))
        embed.eval()
        # embed.to(device)
        total_characters = len(text_data)
        # text_tokens = tokenizer.encode(text_data)
        if divide_by_line:
            encoded_input = text_to_token_ids_batch_total(text_data, tokenizer, BASE_CONFIG['context_length'], eos_id)
            encoded_input = torch.tensor(encoded_input)
        else:
            encoded_input = divide_data(text_data, tokenizer, 40, 15)
        # total_tokens = len(text_tokens)
        # print("Tokens:", total_tokens)
        print("Shape of document:", encoded_input.shape)
        # print(tokenizer_embed.batch_decode(encoded_input['input_ids']))
        # print(tokenizer.decode(text_tokens))

        with torch.no_grad():
            model_output = embed(encoded_input)
        att_encoded = torch.where(encoded_input == eos_id, 0, 1)

        sentence_embeddings = mean_pooling(model_output, att_encoded)

        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        print(sentence_embeddings.shape)

        input_text = ''

        while True:
            query_text = input("User: ")
            if query_text == 'q':
                print("\n\nEXIT")
                break
            query_tok = text_to_token_ids(query_text, tokenizer)
            with torch.no_grad():
                query = embed(query_tok)
            att_query = torch.where(query_tok == eos_id, 0, 1)
            query = mean_pooling(query, att_query)
            query = torch.nn.functional.normalize(query, p=2, dim=1)
            print(query.shape)  # shape of query vector

            scores = retriever(query, sentence_embeddings.unsqueeze(0))
            document_found = torch.topk(scores, top_k, dim=-1).indices.data[0][0].item()
            if divide_by_line:
                found_doc = text_data[document_found]
            else:
                found_doc = token_ids_to_text(encoded_input.data[document_found], tokenizer)

            entry = {
                'instruction': f'Use the following context to answer a question:\n{found_doc}\n' +
                               f'Given this context, answer the following question:\n{query_text}',
                'input': '',
                'output': ''
            }
            input_text = input_text + format_input(entry)
            print(input_text)
            token_ids = generate(
                model=llm,
                idx=text_to_token_ids(input_text, tokenizer).to(device),
                max_new_tokens=256,
                context_size=BASE_CONFIG["context_length"],
                eos_id=50256
            )
            generated_text = token_ids_to_text(token_ids, tokenizer)
            response_text = generated_text[len(input_text):]
            # print(input_text, end='')
            print("Assistant: " + response_text)
            input_text = input_text + response_text
            input_text = ''

    else:
        tokenizer_embed = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        embed = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        # print("Sentence embeddings:")
        # print(sentence_embeddings)

        total_characters = len(text_data)
        # text_tokens = tokenizer.encode(text_data)
        encoded_input = tokenizer_embed(text_data, padding=True, truncation=True, return_tensors='pt')
        # total_tokens = len(text_tokens)
        # print("Tokens:", total_tokens)
        print("Shape of document:", encoded_input['input_ids'].shape)
        # print(tokenizer_embed.batch_decode(encoded_input['input_ids']))
        # print(tokenizer.decode(text_tokens))

        with torch.no_grad():
            model_output = embed(**encoded_input)

        sentence_embeddings = mean_pooling_minilm(model_output, encoded_input['attention_mask'])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        print(sentence_embeddings.shape)

        input_text = ''

        while True:
            query_text = input("User: ")
            if query_text == 'q':
                print("\n\nEXIT")
                break
            query_tok = tokenizer_embed(query_text, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                query = embed(**query_tok)

            query = mean_pooling_minilm(query, query_tok['attention_mask'])
            query = torch.nn.functional.normalize(query, p=2, dim=1)
            print(query.shape)  # shape of query vector

            scores = retriever(query, sentence_embeddings.unsqueeze(0))
            print(scores)
            document_found = torch.topk(scores, top_k, dim=-1).indices
            print(text_data[document_found])

            entry = {
                'instruction': f'Use the following context to answer a question:\n{text_data[document_found]}\n\
                Given this context, answer the following question:\n{query_text}',
                'input': '',
                'output': ''
            }
            input_text = input_text + format_input(entry)
            token_ids = generate(
                model=llm,
                idx=text_to_token_ids(input_text, tokenizer).to(device),
                max_new_tokens=256,
                context_size=BASE_CONFIG["context_length"],
                eos_id=50256
            )
            generated_text = token_ids_to_text(token_ids, tokenizer)
            response_text = generated_text[len(input_text):]
            # print(input_text, end='')
            print("Assistant: " + response_text)
            input_text = input_text + response_text
            input_text = ''
