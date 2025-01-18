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