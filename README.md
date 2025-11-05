## Overview
we present a prompt-leaning framework that integrates gene expression data into large language models (LLMs) to generate low-dimensional cell embeddings, called scPT. By generating prompts from gene expression profiles and putting them into transformer layers, scPT enhances the LLM embeddings, effectively fusing expression and text identity information.
<img src="framework.png">


## Requirements
* Python==3.10

## Installation
Start by following this source codes:
```bash
conda create -n scPT python=3.10
pip install -r requirements.txt
```


## Docker package download(Optional)
```bash
docker pull closmouz/scgcm
```

Run scGCM in container
```bash
docker run -v /path/to/your/data:/apps/data/ -it closmouz/scgcm
```



## Tutorial
* Step 1: Download the corresponding nomic-ai files from https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe/tree/main.
* Step 2: Use `train.py` to train the model.
