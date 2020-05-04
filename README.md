# Text Classification using Different Techniques

Text Classification is one of the most common / basic NLP tasks. I have performed this task with a custom dataset using two different methods, <ol>
<li>Using BiDirectional LSTMs as pipeline for text classification with GLOVE embeddings</li>
<li>Using <a href = "https://arxiv.org/abs/1607.01759">fastText</a> Classifier as pipeline with GLOVE embeddings </li> 
</ol>

<B>Note : </B> Instead of using a simple RNN as a base model, I have directly begun with an BiLSTM, as a simple RNN, suffers from the infamous vanishing gradient problem, and LSTMs are pretty good at handling that stuff. Check <a href = "https://colah.github.io/posts/2015-08-Understanding-LSTMs/">this</a> blog to read about LSTMs and the Vanishing gradient problem with RNNs.

#### Dataset Details :
<p>Instead of using datasets from <a href = "https://pytorch.org/text/datasets.html">torchtext</a> that can be directly used to load as a torch dataset and perform operations. Therefore, I just took a fairly popular kaggle dataset, <a href="https://www.kaggle.com/bittlingmayer/amazonreviews">Amazon Reviews</a>. as my dataset for this task.</p>

Details of Amazon Reviews Dataset : <br>
<ul><li>Training reivews: 3600000 </li>
<li>Test reviews: 400000 <br></ul>

<p> As training on 3.6 million reviews is quite computationally heavy, I have taken a subset of this, by using 100,000 training reviews and 20k testing reviews. Further divided the training reviews into 80k train and 20k validation.</p>

### Preprocessing our datasets
Usually with any kind of dataset, we generally arrive at one of the following points,
<ol>
 <li>Lists of texts and labels</li>
 ```python
   TEXT = data.Field(sequential=True, 
                          tokenize='spacy', 
                          include_lengths=True, 
                          use_vocab=True)
   LABEL = data.Field(sequential=False, 
                            use_vocab=False, 
                            pad_token=None, 
                            unk_token=None)

   fields = fields = [
       ('text', TEXT), 
       ('label', LABEL)
   ]

   train_examples = [data.Example.fromlist([train_sentences[i], train_labels[i]], fields) 
                     for i in range(len(train_sentences))]
   val_examples = [data.Example.fromlist([val_sentences[i], val_labels[i]], fields) 
                     for i in range(len(val_sentences))]
   test_examples = [data.Example.fromlist([test_sentences[i], test_labels[i]], fields) 
                   for i in range(len(test_sentences))]
       
 ```
 <li>CSV / TSV / JSON file, if this is the case check this <a href = "https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/A%20-%20Using%20TorchText%20with%20Your%20Own%20Datasets.ipynb" > link </a> out</li>
</ol>

The data module provides the following:
<ol>
<li>Ability to define a preprocessing pipeline</li>
<li>Batching, padding, and numericalizing (including building a vocabulary object)</li>
<li>Wrapper for dataset splits (train, validation, test)</li>
<li>Loader for a custom NLP dataset</li>
</ol>

I have 


### BiDirectional LSTM :

### FastText Classifier :


### References :

1. References : https://github.com/bentrevett/pytorch-sentiment-analysis





