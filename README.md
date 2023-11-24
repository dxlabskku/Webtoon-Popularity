# Predicting webtoon popularity with two multimodal fusion strategies: a business platform service perspective
This repository is to supplement the paper "Predicting webtoon popularity with two multimodal fusion strategies: a business platform service perspective".


## Abstract
Webtoon, a digital version of a cartoon, has become a prominent cultural phenomenon in South Korea. With the increasing number of mobile users, Webtoon platforms have become more accessible, allowing people to enjoy this culture regardless of their location and time. Interestingly, high-quality Webtoons have been adapted into other forms of media content, such as movies and dramas, following the concept of `one-source, multiuse'. Consequently, predicting the popularity of Webtoon content has become crucial for companies aiming to profit from various content markets. In the realm of multimedia, several scholars have explored multimodal classifiers for predicting popularity. Building upon the findings of prior research, we propose a multimodal classifier, which leverages images and text to assess the popularity of Webtoon content. In line with this, we collected and built the dataset, including webtoon thumbnails and stories, from two largest major Webtoon platforms in South Korea. The proposed model is developed and originated by an early-fusion architecture, combining long-short term memory framework with Ko-Sentence-Transformer embeddings and convolutional neural network. The model achieved impressive results, with a 94.17\% F1-score and 96.82\% accuracy in predicting the popularity of Webtoon content. The collected dataset is publicly available at https://github.com/dxlabskku/Webtoon-Popularity.


## Overview of our framework
<img alt="earlyfusion_blurred" src="https://github.com/dxlabskku/Webtoon-Popularity/assets/43632309/425f6d74-7f19-43b8-9872-05296dffb28e" width="800" height="500">
<br>
<strong>Figure 1 : Proposed model</strong>
<br>


## Dataset
We constructed webtoon dataset collected from two major webtoon platforms in South Korea, Naver Webtoon and Kakao Webtoon. The dataset contains the title, story(synopsis), and thumbnail of each webtoon. The webtoon dataset includes titles, story(synopses), and thumbnail images for 4,770 launched webtoons and 11,931 challenge webtoons.

## Model
We proposed an early fusion multimodal model using image and text features. The image model is a Convolutional neural network(CNN) based on VGG16 scratch. And text model is a Long Short Term Memory(LSTM) by Ko-Sentence-Transformer embedding vector.  


## Reference
TBD
