# Misinformation/Disinformation Detection and Ranking Using Predictive and Generative Models  
## Chris Mo and Ryan Xavier, DSC 180A/B 2025-2026  

In this project, we develop AI/ML enabled ways to detect misinformation in textual data. Starting with the Liar-PLUS dataset, we train predictive models on a number of factuality facotors:  
* BERT/Sentence Transformers
* Spam
* Political Bias
* Sensationalism

We then use these factuality factors in tandem to predict the Politifact truth label of a statement.  

## Requirements  
To reproduce the code, you will need to install the python packages in `req.txt` by running:  
`pip install -r req.txt`  
**Note:** This is assuming you do not have a CUDA runtime installed. To install the CUDA-compatible versions, uninstall PyTorch and install the version corresponding to your CUDA version on PyTorch's website

