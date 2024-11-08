### Question-Answering Chatbots
Right now, this is just running GPT2 in pure Pytorch
model.py - contains outline for GPT model
data_prep.py - helper functions for dataloader
generation.py - helper functions for text generation
pretrain.py - allows you to pretrain the model based on data, or load weights from OpenAI and do text generation with it
finetune.py - finetune the model for answering questions based on Alpaca dataset, using Phi-3 formatted prompt

TODO: add RAG
