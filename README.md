# Hotel-ID to Combat Human Trafficking 2021 (FGVC8) - 8th place solution
Code used for [Hotel-ID to Combat Human Trafficking 2021 - FGVC8](https://www.kaggle.com/c/hotel-id-2021-fgvc8) kaggle competition. The task was to identify hotel to which given image of a room belongs to.

Detailed description: https://www.kaggle.com/c/hotel-id-2021-fgvc8/discussion/242207


## Data
For training I used only competition data rescaled and padded to 512x512 pixels but including external data (like [Hotels-50K dataset](https://github.com/GWUvision/Hotels-50K)) can improve the score significantly.

EDA: [src/hotel-id-eda-with-plotly.ipynb](src/hotel-id-eda-with-plotly.ipynb) ([nbviewer](https://nbviewer.jupyter.org/github/michal-nahlik/kaggle-hotel-id-2021/blob/master/src/hotel-id-eda-with-plotly.ipynb))

Image preprocessing notebook: [src/hotel-id-preprocess-images.ipynb](src/hotel-id-preprocess-images.ipynb)<br>
512x512 dataset: https://www.kaggle.com/michaln/hotelid-images-512x512-padded<br>
256x256 dataset: https://www.kaggle.com/michaln/hotelid-images-256x256-padded<br>
Notebook to download Hotels-50K dataset: [src/download-hotels-50K.ipynb](/src/download-hotels-50K.ipynb)<br>


## Description
Trained 3 types of models with different backbones:<br>
ArcMargin model: [src/training/hotel-id-arcmargin-training.ipynb](src/training/hotel-id-arcmargin-training.ipynb)<br>
CosFace model: [src/training/hotel-id-cosface-training.ipynb](src/training/hotel-id-cosface-training.ipynb)<br>
Classification model: [src/training/hotel-id-classification-training.ipynb](src/training/hotel-id-classification-training.ipynb)<br>

Parameters: Lookahead (k=3) + AdamW optimizer, OneCycleLR scheduler, CrossEntropyLoss/CosFace loss

These models were then used to generate embeddings for the images which were then used to calculated cosine similarity of the test images to the train dataset. Product of similarities was used to ensemble output from different models and to find the top 5 most similar images from different hotels.

Trained models: https://www.kaggle.com/michaln/hotelid-trained-models<br>
Inference notebook: [src/hotel-id-inference.ipynb](src/hotel-id-inference.ipynb)


## Results
Evaluation metric: [Mean Average Precision @5](https://www.kaggle.com/c/hotel-id-2021-fgvc8/overview/evaluation)

| Type | Backbone | Embed size | Public LB| Private LB | Epochs | 
| --- | --- | --- | --- | --- | --- |
| ArcMargin | eca_nfnet_l0 | 1024 | 0.6564 | 0.6704 | 6/6 |
| ArcMargin | efficientnet_b1 | 4096 | 0.6780 | 0.6962 | 9/9 |
| Classification | eca_nfnet_l0 | 4096 | 0.6691 | 0.6875 | 6/9|
| CosFace | ecaresnet50d_pruned| 4096 | 0.6702 | 0.6796 | 9/9 |
| Ensemble |  |  | 0.7273 | 0.7446 | |



## Instructions
1) Prepare data: download the [preprocessed dataset](https://www.kaggle.com/michaln/hotelid-images-512x512-padded) or run [hotel-id-preprocess-images](src/hotel-id-preprocess-images.ipynb) notebook to generate images
2) Train models: run [hotel-id-arcmargin-training](src/training/hotel-id-arcmargin-training.ipynb), [hotel-id-cosface-training](src/training/hotel-id-cosface-training.ipynb), [hotel-id-classification-training](src/training/hotel-id-classification-training.ipynb) notebooks, or use [trained models](https://www.kaggle.com/michaln/hotelid-trained-models)
3) Inference: Edit models and paths in [inference](src/hotel-id-inference.ipynb) notebook and run it on Kaggle