# melanoma-detection

A melanoma detector for [the Kaggle competition](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview).

To run training, first download the data either through the Kaggle api or the competition website. 
Just make sure that the jpegs are placed on data/jpeg/train. The install requirements.

Inside the app/ folder there's a little aws lambda style container which serves as the backend for 
the [deployed app](https://docs.dlda60ex9ihta.amplifyapp.com/). (The front end can be found in the docs branch)

While the results are very good (87 AUC), they don't match the state of the art implementation. The reason is mainly
that we used the data for the 2020 competition only instead of making use past version's data too. While the SOTA authors
also include ensembling and data augmentation, I estimate the bulk of the difference comes from the first issue. 