# Topic Modelling #

**Topic model** is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents.  It provides a convenient way to analyse big unclassified Text. The  main  importance  of  topic  modeling  is  to  discover patterns of word-use and how to connect documents that share similar  patterns.

Intuitively, we can say that if a given document is about a particular topic then one would expect particular words to appear in the document more or less frequently. (Exception might be always there)

e.g. "computer" and "technology"  will appear more in documents about computers and words "dress" "fashion" words to appear more in fashion article.  

## LDA and NMF ##

There are many ways of doing Topic Modelling. Some of them are LSA ( Latent Semantic Analysis), PLSA (Probabilistic Latent Semantic Analysis), LDA (Latent Dirichlet Allocation) and Correlated Topic Model (CTM) etc.

Here we have try LDA and NMF (Non-negative Matrix Factorization) in Python.

In Python we can use Genism or Sklearn for these algorithm. We have used Sklearn.  LDA is based on probabilistic graphical modeling while NMF relies on linear algebra.



### Steps ###

1. Collection and pre-processing of data
2. Calculate tf-IDF and Count vector for teh documents for NMF and LDA respectively.
3. NMF and LDA with Sckit Learn
4. Displaying the output



## How to use ##

`python topic-model.py nmf`  for NMF

`python topic-model.py lda` for LDA



## Output ##

To see teh output see output_lda.txt and output_nmf.txt
