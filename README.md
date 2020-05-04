# Sentiment Analysis using Different Techniques

Sentiment Analysis is one of the most common NLP tasks. I have performed sentiment analysis using two different methods, <ol>
<li>Using BiDirectional LSTMs as pipeline for text classification with GLOVE embeddings</li>
 <li>Using <a href = "https://arxiv.org/abs/1607.01759">fastText</a> Classifier as pipeline with GLOVE embeddings </li> 
</ol>

<B>Note : </B> Instead of using a simple RNN, I have directly begun with an BiLSTM, as a simple RNN, suffers from the infamous vanishing gradient problem, and LSTMs are pretty good at handling that stuff. Check <a href = "https://colah.github.io/posts/2015-08-Understanding-LSTMs/">this</a> blog to read about LSTMs and the Vanishing gradient problem with RNNs.

#### Dataset Details :
<p>Instead of using datasets from <a href = "https://pytorch.org/text/datasets.html">torchtext</a> that can be directly used to load as a torch dataset and perform operations. Therefore, I just took a fairly popular kaggle dataset, <a href="https://www.kaggle.com/bittlingmayer/amazonreviews">Amazon Reviews</a>. as my dataset for this task.</p>

Details of Amazon Reviews Dataset : <br>
<ul><li>Training reivews: 3600000 </li>
<li>Test reviews: 400000 <br></ul>

<p> As training on 3.6 million reviews is quite computationally heavy, I have taken a subset of this, by using 100,000 training reviews and 20k testing reviews. Further divided the training reviews into 80k train and 20k validation.</p>

### Using TORCHTEXT.DATA from pytorch documentation
The data module provides the following:
<ul>
<li>Ability to define a preprocessing pipeline</li>
<li>Batching, padding, and numericalizing (including building a vocabulary object)</li>
<li>Wrapper for dataset splits (train, validation, test)</li>
<li>Loader for a custom NLP dataset</li>
</ul>
 







