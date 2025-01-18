import json
import os
import urllib.request
import random
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import partial
from generation import generate, text_to_token_ids, token_ids_to_text, text_to_token_ids_batch, text_to_token_ids_batch_total
from pretrain import calc_loss_loader, train_model, train_model_simple, load_weights_into_gpt
from finetune import download_and_load_file, format_input, format_input_simple, extract_response
import time
from gpt_download import download_and_load_gpt2
from tqdm import tqdm
from model import GPTModel, replace_linear_with_lora, Embedding
from rag import Retriever, divide_data
from embedding import mean_pooling, mean_pooling_minilm
from transformers import AutoTokenizer, AutoModel
from data_prep import split_by_line
from rouge import Rouge


def query_model(
        prompt,
        model="llama2",
        url="http://localhost:11434/api/chat"
):
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )

    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data


def generate_model_scores_acc(json_data, json_key, model="llama2"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input_simple(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score,"
            f"with regard to accuracy only, instead of poetic beauty or conciseness. "
            f"Respond with the integer number only, without any explanation, labels, or other text."
        )
        score = query_model(prompt, model)[-2:]
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue
    return scores


def generate_model_scores_coh(json_data, json_key, model="llama2"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input_simple(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score,"
            f"with regard to fluency (syntactically-accurate sentence formation), "
            f"relevancy to the user's query, and penalizing nonsensical responses. "
            f"Do not pay attention to accuracy. "
            f"Respond with the integer number only, without any explanation, labels, or other text."
        )
        score = query_model(prompt, model)[-2:]
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue
    return scores


def generate_model_scores_rouge(json_data):
    rouge = Rouge()
    scores_r, scores_p, scores_f = [], [], []
    for entry in tqdm(json_data, desc="Scoring entries"):
        score = rouge.get_scores(entry['model_response'].lower(), entry['output'].lower())
        score_r = score[0]['rouge-1']['r']
        score_p = score[0]['rouge-1']['p']
        score_f = score[0]['rouge-1']['f']
        scores_r.append(score_r*100)
        scores_p.append(score_p*100)
        scores_f.append(score_f*100)
    return scores_r, scores_p, scores_f


# rag eval
if __name__ == "__main__":
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

    top_k = 1
    use_minilm = input("use minilm? ") == 'y'

    if use_minilm:
        tokenizer_embed = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        embed = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    else:
        CHOOSE_MODEL = "gpt2-small (124M)"
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
        embed = Embedding(BASE_CONFIG)
        a = input('file name for embedding model? ')
        embed.load_state_dict(torch.load(f"{a}.pth", weights_only=True))
        embed.eval()
    if input("generate model response? ") == "y":
        file_path = 'rag_data.json'

        url = (
            "https://cdn-lfs-us-1.hf.co/repos/95/84/95844ab38b7a8f1bacea788114fb3c3880d280960ded9d337ea6fdcd67531cf9/707108955e1777294909e2f428b54695a6be6771ec05409bc953e0e451d747e2?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27wikipedia-dev.json%3B+filename%3D%22wikipedia-dev.json%22%3B&response-content-type=application%2Fjson&Expires=1736822911&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczNjgyMjkxMX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzk1Lzg0Lzk1ODQ0YWIzOGI3YThmMWJhY2VhNzg4MTE0ZmIzYzM4ODBkMjgwOTYwZGVkOWQzMzdlYTZmZGNkNjc1MzFjZjkvNzA3MTA4OTU1ZTE3NzcyOTQ5MDllMmY0MjhiNTQ2OTVhNmJlNjc3MWVjMDU0MDliYzk1M2UwZTQ1MWQ3NDdlMj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=sp43L1JkXOpnn60tTq3u6tEFYxPN4cC4DZVpA2kkNVOXcTlEsoVNytjnS8a6bdnFtyVdAPpAJRxFgA3G%7E%7Ep6EYlZIoZv3hB6tgvRq26DiVog6mu7y5MdX7eX334uDPWge7dH6vXtD7xjHM9tuUG8EQj6e0LXtSfkkQSSt0MFOGv88fjWy%7EG7uLkDZrtNbRKHH7B3IVcWL9QTR9pJDXVhr9DooRYGV%7EwwIHzBVu2u7oZ4L0COLjfrpza2PSrh1VHo5uPkn3GTyFW5GGJeeGIrtjghLeW6zzWlOsETAAruf4C3Wj-lYeFkVjeviMpq8T1ioUwk7UJwceqAcHB7tblSAg__&Key-Pair-Id=K24J24Z295AEI9"
        )  # validation from trivia-qa

        if not os.path.exists(file_path):
            if not os.path.exists('rag_data_raw.json'):
                with urllib.request.urlopen(url) as response:
                    text_data = response.read().decode('utf-8')
                with open('rag_data_raw.json' 'w', encoding='utf-8') as file:
                    file.write(text_data)
            with open('rag_data_raw.json', 'r', encoding='utf-8') as file:
                data = json.load(file)
            # preprocess dataset
            use_limit = 1000
            new_data = random.sample(data['data'], use_limit)
            data = []
            for entry_raw in new_data:
                dat = entry_raw['paragraphs'][0]
                try:
                    entry={
                        "context": dat['context'],
                        "instruction": dat['qas'][0]["question"],
                        "output": dat['qas'][0]['answers'][0]['text']
                    }
                except:
                    continue
                data.append(entry)
            with open("../rag_data.json", "w") as file:
                json.dump(data, file, indent=4)
        else:
            with open(file_path, 'r') as file:
                data = json.load(file)
        use_num_entries = int(input(f"how many entries to use(max {len(data)})? "))
        test_data = data[:use_num_entries]

        divide_by_line = input("Divide by line? ") == 'y'

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            device = torch.device('mps')

        CHOOSE_MODEL = "gpt2-large (774M)"
        # CHOOSE_MODEL = "gpt2-medium (355M)"
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
        llm = GPTModel(BASE_CONFIG)
        llm.eval()
        x = input('File name for LLM chatbot? ')
        llm.load_state_dict(torch.load(f"{x}.pth", weights_only=True))
        llm.to(device)

        retriever = Retriever(384)
        if use_minilm:
            for i, entry_raw in tqdm(enumerate(test_data), total=len(test_data)):
                text_data = entry_raw["context"]

                if divide_by_line:
                    text_data = split_by_line(text_data)
                    encoded_input2 = text_to_token_ids_batch_total(text_data, tokenizer, BASE_CONFIG['context_length'], eos_id)
                    encoded_input2 = torch.tensor(encoded_input2)
                else:
                    encoded_input2 = divide_data(text_data, tokenizer, 40, 5)
                    text_data = [token_ids_to_text(a, tokenizer) for a in encoded_input2.data]
                # print("Sentence embeddings:")
                # print(sentence_embeddings)

                total_characters = len(text_data)
                encoded_input = tokenizer_embed(text_data, padding=True, truncation=True, return_tensors='pt')

                with torch.no_grad():
                    model_output = embed(**encoded_input)

                sentence_embeddings = mean_pooling_minilm(model_output, encoded_input['attention_mask'])
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

                query_text = entry_raw['instruction']
                query_tok = tokenizer_embed(query_text, padding=True, truncation=True, return_tensors='pt')
                with torch.no_grad():
                    query = embed(**query_tok)

                query = mean_pooling_minilm(query, query_tok['attention_mask'])

                scores = retriever(query, sentence_embeddings.unsqueeze(0))
                document_found = torch.topk(scores, top_k, dim=-1).indices.data[0][0].item()

                found_doc = text_data[document_found]

                entry_other = {
                    'instruction': f'Use the following context to answer a question:\n{found_doc}\n' +
                                   f'Given this context, answer the following question:\n{query_text}',
                    'input': '',
                    'output': ''
                }

                input_text = format_input(entry_other)
                start = time.time()
                token_ids = generate(
                    model=llm,
                    idx=text_to_token_ids(input_text, tokenizer).to(device),
                    max_new_tokens=256,
                    context_size=BASE_CONFIG["context_length"],
                    eos_id=eos_id
                )
                generated_text = token_ids_to_text(token_ids, tokenizer)
                response_text = extract_response(input_text, generated_text)
                test_data[i]["model_response"] = response_text
                test_data[i]["time_taken"] = time.time() - start
            with open("rag-data-with-response-minilm.json", "w") as file:
                json.dump(test_data, file, indent=4)
        else:
            for i, entry_raw in tqdm(enumerate(test_data), total=len(test_data)):
                text_data = entry_raw["context"]
                if divide_by_line:
                    text_data = split_by_line(text_data)
                    encoded_input = text_to_token_ids_batch_total(text_data, tokenizer, BASE_CONFIG['context_length'], eos_id)
                    encoded_input = torch.tensor(encoded_input)
                else:
                    encoded_input = divide_data(text_data, tokenizer, 40, 5)
                with torch.no_grad():
                    model_output = embed(encoded_input)
                att_encoded = torch.where(encoded_input == eos_id, 0, 1)

                sentence_embeddings = mean_pooling(model_output, att_encoded)

                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                query_text = entry_raw['instruction']
                query_tok = text_to_token_ids(query_text, tokenizer)
                with torch.no_grad():
                    query = embed(query_tok)
                att_query = torch.where(query_tok == eos_id, 0, 1)
                query = mean_pooling(query, att_query)

                scores = retriever(query, sentence_embeddings.unsqueeze(0))
                document_found = torch.topk(scores, top_k, dim=-1).indices.data[0][0].item()
                if divide_by_line:
                    found_doc = text_data[document_found]
                else:
                    found_doc = token_ids_to_text(encoded_input.data[document_found], tokenizer)

                entry_other = {
                    'instruction': f'Use the following context to answer a question:\n{found_doc}\n' +
                                   f'Given this context, answer the following question:\n{query_text}',
                    'input': '',
                    'output': ''
                }

                input_text = format_input(entry_other)
                start = time.time()
                token_ids = generate(
                    model=llm,
                    idx=text_to_token_ids(input_text, tokenizer).to(device),
                    max_new_tokens=256,
                    context_size=BASE_CONFIG["context_length"],
                    eos_id=eos_id
                )
                generated_text = token_ids_to_text(token_ids, tokenizer)
                response_text = extract_response(input_text, generated_text)
                test_data[i]["model_response"] = response_text
                test_data[i]["time_taken"] = time.time() - start
            with open("rag-data-with-response.json", "w") as file:
                json.dump(test_data, file, indent=4)
    else:
        if use_minilm:
            with open("rag-data-with-response-minilm.json", "r") as file:
                test_data = json.load(file)
        else:
            with open("rag-data-with-response.json", "r") as file:
                test_data = json.load(file)

    scores_coh = generate_model_scores_coh(test_data, "model_response")
    print(f"Number of scores: {len(scores_coh)} of {len(test_data)}")
    print(f"Average score for coherency: {sum(scores_coh)/len(scores_coh):.2f}\n")

    scores_acc = generate_model_scores_acc(test_data, "model_response")
    print(f"Number of scores: {len(scores_acc)} of {len(test_data)}")
    print(f"Average score for accuracy: {sum(scores_acc)/len(scores_acc):.2f}\n")

    scores_r, scores_p, scores_f = generate_model_scores_rouge(test_data)
    print(f"Average score for accuracy (r): {sum(scores_r)/len(scores_r):.2f}")
    print(f"Average score for accuracy (p): {sum(scores_p)/len(scores_p):.2f}")
    print(f"Average score for accuracy (f): {sum(scores_f)/len(scores_f):.2f}\n")

    time_per_word = list(map(lambda x: x['time_taken']/len(x['model_response'].split()), test_data))
    print(f"Average time per word taken: {sum(time_per_word)/len(time_per_word):.2f}\n")
    print(f"Maximum time per word taken: {max(time_per_word):.2f}\n")


# chatbot eval
if __name__ == "chatbot":
    file_path = 'instruction-data.json'

    url = (
        "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json"
    )

    data = download_and_load_file(file_path, url)
    data = data[:int(len(data))]
    print("Number of entries:", len(data))

    train_portion = int(len(data)*0.85)
    test_portion = int(len(data)*0.1)
    val_portion = len(data) - train_portion - test_portion
    test_data = data[train_portion:train_portion+test_portion]
    val_data = data[train_portion+test_portion:]

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

    # CHOOSE_MODEL = "gpt2-large (774M)"
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    llm = GPTModel(BASE_CONFIG)
    llm.eval()
    x = input('File name for LLM chatbot? ')
    llm.load_state_dict(torch.load(f"{x}.pth", weights_only=True))
    llm.to(device)

    for entry in test_data[:3]:
        input_text = format_input(entry)
        token_ids = generate(
            model=llm,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=eos_id
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = extract_response(input_text, generated_text)
        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text}")

    if input("generate model response? ") == "y":
        for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
            input_text = format_input(entry)
            token_ids = generate(
                model=llm,
                idx=text_to_token_ids(input_text, tokenizer).to(device),
                max_new_tokens=256,
                context_size=BASE_CONFIG["context_length"],
                eos_id=eos_id
            )
            generated_text = token_ids_to_text(token_ids, tokenizer)
            response_text = extract_response(input_text, generated_text)
            test_data[i]["model_response"] = response_text

        with open("instruction-data-with-response.json", "w") as file:
            json.dump(test_data, file, indent=4)
    else:
        with open("instruction-data-with-response.json", "r") as file:
            test_data = json.load(file)

    scores_acc = generate_model_scores_acc(test_data, "model_response")
    print(f"Number of scores: {len(scores_acc)} of {len(test_data)}")
    print(f"Average score: {sum(scores_acc)/len(scores_acc):.2f}\n")

    scores_coh = generate_model_scores_coh(test_data, "model_response")
    print(f"Number of scores: {len(scores_coh)} of {len(test_data)}")
    print(f"Average score: {sum(scores_coh)/len(scores_coh):.2f}\n")

