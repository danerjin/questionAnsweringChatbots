### Question-Answering Chatbots  
FILE STRUCTURE:

    model.py - contains outline for GPT model  
    data_prep.py - helper functions for dataloader  
    generation.py - helper functions for text generation  
    pretrain.py - allows you to pretrain the model based on data, or load weights from OpenAI and do text generation with it  
    finetune.py - finetune the foundation model for answering questions based on Alpaca dataset, using Phi-3 formatted prompt  
    embedding.py - finetune the foundatino model to generate sentence embeddings
    rag.py - toy example, using the "Virginia Declaration of Rights" by George Mason
    eval.py - evaluate model, either the chatbot or the RAG system
  
In order to run this on your own computer, download the repo:   
`git clone https://github.com/danerjin/questionAnsweringChatbots.git`    
Then, download the necessary packages:    
`pip install -r requirements.txt`   
Finetune the chatbot model:    
`python finetune.py`    
Then, finetune the embedding model:  
`python embedding.py`  
Finally, evaluate the RAG system:   
`python eval.py`

# Explanation:  
We start off with GPT2, and we load pre-trained weights from OpenAI. This is our foundation model.  
We finetune foundation model to act as a chatbot -- using [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned), formatted using Phi-3 prompt template.   
We also finetune foundation model to act as embedding model using transfer learning -- replace last layer. Finetune using STSB.   
Build Retrieval System - similar to attention. Unfortunately, did not have time to train key, query matrices.  

# Flowchart (How it is calculated):   
![Flowchart](https://raw.githubusercontent.com/danerjin/questionAnsweringChatbots/refs/heads/main/flowchart.png)   

# Data:   
![Data](https://raw.githubusercontent.com/danerjin/questionAnsweringChatbots/refs/heads/main/data.png)   

# Comparison of GPT Embeddings, vs MiniLM-L6-v2 embeddings:  
![Graph](https://raw.githubusercontent.com/danerjin/questionAnsweringChatbots/refs/heads/main/graphs.png)

