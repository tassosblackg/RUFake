# RUFake
Fake or Real Face Image- Autoencoder and CNN classifier

## :large_orange_diamond: Description:
We have two datasets one with 10K rgb *unlabeled* images with size *256x256* which are used to train an Autoencoder model. The goal is to save its weights and then use the Encoder part to create a classifier and fine tuning upon the second dataset which has 10K rgb images labeled as ***'Real'*** and 10K images *labeled* as ***'Fake'***. The second dataset's images are also *256x256* pixels. The *Autoencoder* was develoveped using *Google Colab* and *jupiter notebook* in order to fast prototype and evaluate a few architectures, till I chose one that suits better. Then I used the saved model weights and by keeping the *Encoder* part of the model I stacked on top of a classifier architecture [where Encoder parameters were set to non-trainable]. This is a basic concept of *Fine-Tuning* a NN model using two types of datasets (Unlabeled and Labeled).

## :large_orange_diamond: Model Info:
Both models are based on Convolutional Neural Networks a.k.a CNNs, which are implemented using **Keras API** of *Tensorflow*.

## :large_orange_diamond: Repository Structure:

 ------------------------------------------------------------------

 Name                 | Description
 :---:                | :---
 [datasets]                              | Directory with the 2 types of image datasets 
 [saved_models]                          | Includes the models' architecture, weights, etc.
 [classifier_summary]                    | Images showing classifier architecture
 [medver_classifier.py]                  | The classifier train process --run locally
 [medver_autoencoder.ipynb]              | Research & results of different architectures run on Google Colab, here is the Autoencoder notebook
 [medver_autoencoder.py]                 | Autoencoder  --raw python
 [train_classifier_log.csv]              | Training results of first classifier architecture **
 [train_classifier_log2.csv]             | Training results of second classifier architecture **

 [datasets]:                       https://github.com/tassosblackg/RUFake/datasets
 [saved_models]:                   https://github.com/tassosblackg/RUFake/saved_models
 [classifier_summary]:             https://github.com/tassosblackg/RUFake/classifier_architecture
 [medver_classifier.py]:           https://github.com/tassosblackg/RUFake/blob/main/medver_classifier.py
 [medver_autoencoder.ipynb]:       https://github.com/tassosblackg/RUFake/blob/main/mdever_autoencoder.ipynb
 [medver_autoencoder.py]:          https://github.com/tassosblackg/RUFake/blob/main/medver_autoencoder.py 
 [train_classifier_log.csv]:       https://github.com/tassosblackg/RUFake/train_classifier_log.csv
 [train_classifier_log2.csv]:      https://github.com/tassosblackg/RUFake/train_classifier_log2.csv
