# Sentiment Analysis using Different Techniques

Sentiment Analysis is one of the most common NLP tasks. 

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
 







