# Misinformation/Disinformation Detection and Ranking Using Predictive and Generative Models  
### Chris Mo and Ryan Xavier, DSC 180A/B 2025-2026  

## Introduction
In this project, we develop AI/ML enabled ways to detect misinformation in textual data. Starting with the [Liar-PLUS dataset](https://github.com/Tariq60/LIAR-PLUS), we train predictive models on a number of factuality factors, which we define below:  
* **BERT/Sentence Transformers:** Learn patterns with Transformers. Learn the underlying pattern of the text through the use of the transformer attention mechanism.
* **Spam:** Is this spam? How does spam relate to disinformation?
* **Political Bias:** Does the text exhibit political bias? How promininent is this?
* **Sensationalism:** Is the text using sensationalist words and phrases designed to attract attention, manipulate perhaps?

We then use these factuality factors in tandem to predict the Politifact truth label of a statement using traditional classifiers.  

## Folder Structure:
```
ai4good│
└─── data                  <- folder that holds our dataset and synthetically generated data  
│        │ 
│        └───liar-plus                        <- liar-plus dataset from u/Tariq-60
│        │      │ 
│        │      └───test2.tsv
│        │      └───train2.tsv
│        │      └───val2.tsv
│
│        └───lp-clean                         <- liar-plus data with statements, veracity scores and truth labels
│        │      │ 
│        │      └───test2.tsv
│        │      └───train2.tsv
│        │      └───val2.tsv
│        │      
│        └───labeled_articles.csv             <- manually-labeled data from us and our classmates
│        └───rxavierlabeled_articles.csv
│        └───polifact.csv
│        └───test.pdf                         <- example PDF for our rag model
│        └───top_conservative_bigrams.csv     <- scraped bigrams from all text spoken on the floor of the Senate and the House of Representatitves along with speaker party (https://data.stanford.edu/congress_text)
│        └───top_liberal_bigrams.csv
│        └───vocabs.pkl                       <- author, topic list, job, affiliation vocabularies from training pipeline. Necessary for evaluating test set and other unseen data
│        
└─── notebooks             <- folder that holds our notebooks for training and running models
│        │ 
│        └───gemini_prompt_refined.ipynb      <- Pipeline integrating predictive veracity scores and generative prompting
│        └───mxnet_training.ipynb             <- Adapted from Dr. Arsanjani's approach using MXNet in PyTorch for sentence transformers
│        └───exploration_analysis.ipynb       <- Exploratory notebook examining the Liar-PLUS dataset
│        └───full_pred_model.ipynb            <- Full predictive pipeline of our 4 chosen factuality factors
│        └───ingestor.ipynb                   <- Proof-of-concept notebook showing how we use the News API to scrape listings for our dataset
│        └───prompt-refinement.ipynb          <- Notebook showing our prompt progression, and how the LLM outputs change depending on prompting strategies
│        └───rag_poc.ipynb                    <- Proof-of-concept notebook showing how we can implement a Retrieval-Augment-Generate (RAG) model from a PDF
│        └───webpage_runner.ipynb             <- Full predictive pipeline and generative prompting with Gradio website launcher
│        
└─── utils                 <- folder that holds our utility files
│        │ 
│        └───nlp_utils.py                     <- Function store for opening, cleaning, and processing Liar-PLUS data
│        └───ingestor_utils.py                <- Function store for methods outlined in ingestor.ipynb. Use in full pipeline/unseen data applications
│        └───mxnet_utils.py                   <- Function/class store for common functions used in the MXNet training and evaluation workflow
│ 
│        
└─── (checkpoints)         <- folder that holds our pretrained model
│        │ 
│        └───best.pth                         <- Pretrained MXNet model (too large to include in Github)
│
└───(.config.yaml)         <- stores gemini and nautilus API keys (described below)
└───.gitignore
└───req.txt
└───README.md

```

## Features
* **Statistical Article Vector:** Generates quantifiable statistics for any news article based on the factuality factors, including popular political phrase matching, sentiment scoring, and BERT predictions
* **Generative AI Analysis:** Utilizes a highly specialized and refined prompt to allow an LLM to deeply analyze the article based on the factuality factors and work in tandem with the statistical article vector.
* **Intuitive Website Interface:** Easy-to-use website interface that allows you to copy and paste any text to analyze.

## Requirements  
To reproduce the code, you will need to install the python packages in `req.txt` by changing directories to this repository and running in your terminal:  
```
  pip install -r req.txt
```  
**Note:** This is assuming you do not have a CUDA runtime installed, and will only utilize the CPU for computations. To install the CUDA-compatible versions, uninstall PyTorch and install the version corresponding to your CUDA version on [PyTorch's website](https://pytorch.org/get-started/locally/)  

You'll also need to download [this pretrained model](https://drive.google.com/file/d/1TE7gZblTT_pIVPH00dKf4-SrObbIcKV0/view?usp=sharing) and have it in a folder called "checkpoints." This was not included in our repository due to its size, and it would be necessary to include this folder in your `.gitignore`. Additionally, you would need a `config.yaml` file included in the main repository that follows this structure:  
```
gemini:
  api: "<your API key here>"

nautilus:
  api: "<your API key here>"
```

## Running the Web Application
* After installing all necessary packages, navigate to `notebooks/webpage_runner.ipynb`
* Run all cells
* In the last cell, a local link will appear. Follow the link to get to the demo website and paste an article in!
