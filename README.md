# Text Classification using Different Techniques

Text Classification is one of the most common / basic NLP tasks. I have performed this task with a custom dataset using two different methods, <ol>
<li>Using BiDirectional LSTMs as pipeline for text classification with GLOVE embeddings</li>
<li>Using <a href = "https://arxiv.org/abs/1607.01759">fastText</a> Classifier as pipeline with GLOVE embeddings </li> 
</ol>

<B>Note : </B> Instead of using a simple RNN as a base model, I have directly begun with an BiLSTM, as a simple RNN, suffers from the infamous vanishing gradient problem, and LSTMs are pretty good at handling that stuff. Check <a href = "https://colah.github.io/posts/2015-08-Understanding-LSTMs/">this</a> blog to read about LSTMs and the Vanishing gradient problem with RNNs.

#### Dataset Details :
<p>Instead of using datasets from <a href = "https://pytorch.org/text/datasets.html">torchtext</a> that can be directly used to load as a torch dataset and perform operations. Therefore, I just took a fairly popular kaggle dataset, <a href="https://www.kaggle.com/bittlingmayer/amazonreviews">Amazon Reviews</a> as my dataset for this task.</p>

Details of Amazon Reviews Dataset : <br>
<ul><li>Training reivews: 3600000 </li>
<li>Test reviews: 400000 <br></ul>

<p> As training on 3.6 million reviews is quite computationally heavy, I have taken a subset of this, by using 100,000 training reviews and 20k testing reviews. Further divided the training reviews into 80k train and 20k validation.</p>

### Preprocessing our datasets
I will be using torchtext.Data to do all the preprocessing and to create a dataloader. The torchtext.data module provides the following:
<ol>
<li>Ability to define a preprocessing pipeline</li>
<li>Batching, padding, and numericalizing (including building a vocabulary object)</li>
<li>Wrapper for dataset splits (train, validation, test)</li>
<li>Loader for a custom NLP dataset</li>
</ol>

Usually with any kind of dataset, we generally arrive at one of the following points,
<ul><li>Lists of texts and labels</li></ul>

```python
images_AP = []
images_LAT = []
labels = []

for id in normal_ids:
    AP_image_path = os.path.join(normal_dir, id, 'AP' , 'AP.jpg')
    LAT_image_path = os.path.join(normal_dir, id, 'LAT', 'LAT.jpg')
    images_AP.append(AP_image_path)
    images_LAT.append(LAT_image_path)
    labels.append(np.array([1, 0])) # normal case

for id in damaged_ids:
    AP_image_path = os.path.join(damaged_dir, id, 'AP' , 'AP.jpg')
    LAT_image_path = os.path.join(damaged_dir, id, 'LAT', 'LAT.jpg')
    images_AP.append(AP_image_path)
    images_LAT.append(LAT_image_path)
    labels.append(np.array([0, 1])) # damaged case

data_ap = pd.DataFrame({'Images' : images_AP, 'labels' : labels})
data_lat = pd.DataFrame({'Images' : images_LAT, 'labels' : labels})

class SpineDataset(torch.utils.data.Dataset):
    def __init__(self, ap_data, lat_data, transform=None):
        self.transform = transform
        self.ap_data = ap_data
        self.lat_data = lat_data
        
    def __len__(self):
        return len(self.ap_data)
    
    def __getitem__(self, index):
        ap = Image.open(self.ap_data.loc[index]['Images'])
        lat = Image.open(self.lat_data.loc[index]['Images'])
        label = torch.from_numpy(self.ap_data.loc[index]['labels']).type(torch.LongTensor)
        if self.transform is not None:
            ap = self.transform(ap)
            lat = self.transform(lat)
        return ap, lat, label
```
 
<ul><li>CSV / TSV / JSON file, if this is the case check this <a href = "https://github.com/bentrevett/pytorch-sentimentanalysis/blob/master/A%20-%20Using%20TorchText%20with%20Your%20Own%20Datasets.ipynb" > link </a> out</li></ul>

I have 


### BiDirectional LSTM :

### FastText Classifier :


### References :

1. References : https://github.com/bentrevett/pytorch-sentiment-analysis




