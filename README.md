# Predicting webtoon popularity with two multimodal fusion strategies: a business platform service perspective
This repository is to supplement the paper "Predicting webtoon popularity with two multimodal fusion strategies: a business platform service perspective".


## Abstract
Webtoon, a digital version of a cartoon, has become a prominent cultural phenomenon in South Korea. With the increasing number of mobile users, Webtoon platforms have become more accessible, allowing people to enjoy this culture regardless of their location and time. Interestingly, high-quality Webtoons have been adapted into other forms of media content, such as movies and dramas, following the concept of `one-source, multiuse'. Consequently, predicting the popularity of Webtoon content has become crucial for companies aiming to profit from various content markets. In the realm of multimedia, several scholars have explored multimodal classifiers for predicting popularity. Building upon the findings of prior research, we propose a multimodal classifier, which leverages images and text to assess the popularity of Webtoon content. In line with this, we collected and built the dataset, including webtoon thumbnails and stories, from two largest major Webtoon platforms in South Korea. The proposed model is developed and originated by an early-fusion architecture, combining long-short term memory framework with Ko-Sentence-Transformer embeddings and convolutional neural network. The model achieved impressive results, with a 94.17\% F1-score and 96.82\% accuracy in predicting the popularity of Webtoon content. The collected dataset is publicly available at https://github.com/dxlabskku/Webtoon-Popularity.


## Overview of our framework
<img alt="multimodal" src="https://github.com/dxlabskku/Webtoon-Popularity/assets/43632309/897a7ed9-d13b-4c03-8781-48d17b707bce" width="421" height="250">
<br>
<strong>Figure 1 : Proposed model</strong>
<br>


## Dataset
We constructed webtoon dataset collected from two major webtoon platforms in South Korea, Naver Webtoon and Kakao Webtoon. The dataset contains the title, story(synopsis), and thumbnail of each webtoon. The webtoon dataset includes titles, story(synopses), and thumbnail images for 4,770 launched webtoons and 11,931 challenge webtoons.


## Reference
TBD
