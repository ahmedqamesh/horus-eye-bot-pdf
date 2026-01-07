# About the Project

<p align="center">
  <img 
    src="https://github.com/ahmedqamesh/horus-eye-bot-pdf/blob/main/assets/images/horus-eye-assistant-front.png"
    alt="horus-eye-assistant-front"
    width="500"
  />
</p>

This project implements a **local AI chatbot** that can:
- Load and index **PDF documents**
- Answer questions using **Retrieval-Augmented Generation (RAG)**
- Use **open-source LLMs** via Hugging Face
- Optionally support **audio transcription** (Whisper)

It is designed to run **locally**, using **free and open models**.
---

#### 🚀 Features

- PDF ingestion and semantic search
- Vector storage using **Chroma**
- LLM-based question answering with **LangChain**
- Hugging Face open-source models
- Optional Gradio UI for interaction

#### 🧠 Models Used
- Embeddings: **sentence-transformers/all-MiniLM-L6-v2**
- LLM:  meta-llama/Llama-2-7b-chat-hf

---

## 🧩 Environment Setup
```bash
python3.12 -m venv installations/python_envs/ml_chatbot_data_env
## Activate the environment
source /home/aq/installations/python_envs/ml_chatbot_data_env/bin/activate
```
### Install dependencies
Python Packages (installed via `requirements.txt`)
- langchain
- langchain-community
- transformers
- torch
- chromadb
- sentence-transformers
- gradio
- huggingface_hub

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Hugging Face Authentication
Some models (e.g. LLaMA-2 / LLaMA-3) are gated and require authentication.
1. Create [Hugging Face account](Go to: https://huggingface.co/join)
2. Create a Read token: 
- Go to: https://huggingface.co/settings/tokens 
3. choose  -> Access Tokens -> click on New Token (Token type: Read)
4.Login from the terminal
```bash
huggingface-cli login
```
4. Paste your token when prompted.

## ▶️ Run the Server
```bash
python3 server_chatbot.py
```
The app will be available at: http://0.0.0.0:7860


## Working with Docker
```bash
docker build --no-cache -t build_chatbot_data .
docker run -p 8000:8000 build_chatbot_data
```
### Contact Information
We welcome contributions from the community please contact : [ahmed.qamesh@gmail.com](ahmed.qamesh@gmail.com) .
