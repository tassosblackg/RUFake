# RUFake
Fake or Real Face Image- Autoencoder and CNN classifier

## :large_orange_diamond: Description:
We have two datasets one with 10K rgb *unlabeled* images with size 256x256 which are used to train an Autoencoder model. The goal is to save its weights and then use the Encoder part to create a classifier and fine tuning upon the second dataset which has 10K rgb images labeled as ***'Real'*** and 10K images *labeled* as ***'Fake'***. The second dataset's images are also 256x256 pixels.

## :large_orange_diamond: Model Info:
Both models are based on Convolutional Neural Networks a.k. CNNs implemented using Keras API of Tensorflow.

## :large_orange_diamond: Repository Structure:

 ------------------------------------------------------------------

 Name                 | Description
 :---:                | :---
 [datasets]           | Directory with the 2 types of image datasets 
 [saved_models]       | Includes the models' architecture, weights, etc.
 [classifier_summary]       | Images showing classifier architecture
 [ev3_file.c]         | Module for reading/writing from/to files
 [test_count.c]       | Test to count connected sensors on EV3 brick
 [train_classifier_log.csv]             | Training results of first classifier architecture **
 [train_classifier_log2.csv]             | Training results of second classifier architecture **

 [datasets]:      https://github.com/tassosblackg/RUFake/datasets
 [saved_models]:   https://github.com/tassosblackg/RUFake/saved_models
 [classifier_summary]:  https://github.com/tassosblackg/RUFake/classifier_architecture
 [ev3_file.c]:    https://github.com/tassosblackg/RUFake
 [test_count.c]:  https://github.com/tassosblackg/RUFake
 [lfoa.c]:        https://github.com/tassosblackg/RUFake
