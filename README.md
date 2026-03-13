## Overview
we present a prompt-leaning framework that integrates gene expression data into large language models (LLMs) to generate low-dimensional cell embeddings, called scPT. By generating prompts from gene expression profiles and putting them into transformer layers, scPT enhances the LLM embeddings, effectively fusing expression and text identity information.
<img src="framework.png">


## Requirements
* Python==3.10
* CUDA 12.2

## Installation
To install scPT with Nvidia GPU CUDA support, for Linux Systems:
```bash
conda create -n scPT python=3.10
conda activate scPT
pip install -r requirements.txt
```

## Data availability
* All the data can be found in the supplementary materials of the article.
* The model expects input files in `.h5ad` format. 
* `asap.py`: example script for ASAP dataset preprocessing.

## Running
```bash
python train.py
```

## Tutorial
* You can download the nomic-ai from https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe/tree/main.
* Use `train.py` to train the model, then you can obtain the data embeddings and model parameters.
* We use `result.py` to perform the final result analysis for all methods, the results of the spatial data can be found in the `spatial` folder.
