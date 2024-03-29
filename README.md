# Paper Summarizer - An MLOps project 

In the field of research, where the vast amount of scientific literature can be overwhelming, the necessity for efficient summarization tools is crucial. 
The overall goal of this project, the Paper Summarizer, is therefore to construct an MLOps workflow that can assist researchers in creating concise abstracts for their ground-breaking papers.

The state-of-the-art framework for dealing with any natural language processing (NLP) application is the `transformers` library developed by the Hugging Face group. The library allows for easy integration of many different variations of the transformer architecture. The `transformers` library allows for models build in all of the 3 major deep learning frameworks introduced in the course (PyTorch, Tensorflow and JAX) and we will use the PyTorch version of our chosen model.

## Data
We will be using the comprehensive dataset available from Hugging Face, which includes a variety of scientific articles and their corresponding abstracts. This dataset will serve as the foundation for training and fine-tuning our models, resulting in a specialized and robust tool that is tailored to the complexities of scientific language.

Find the data here: https://huggingface.co/datasets/scientific_papers?fbclid=IwAR1H4fNbqyivbE1a_L_dkbbyfIxoADfi68M5SiEJUQrtxAAeNGN2P1wfDjc

## Model
Our model of choice for the Paper Summarizer is the pre-trained BART model - [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) from Hugging Face. BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder.  By using the `transformers` library the training time can also be shortened significantly, as we will using a pretrained model, meaning the model has already learned to generate meaningful text. BART has been shown to be particularly effective when fine-tuned for text generation and we will fine-tune for the specific text generation task of text summarization. The BART version we will use has been fine-tuned on CNN Daily Mail and we plan to fine-tune it using the extensive scientific articles dataset provided, aligning our tool with the specific demands of research-oriented summarization.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── paper_summarizer  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
