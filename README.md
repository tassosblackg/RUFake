# RUFake
Fake or Real Face Image- Autoencoder and CNN classifier

## Description:
We have two datasets one with 10K rgb *unlabeled* images with size 256x256 which are used to train an Autoencoder model. The goal is to save its weights and then use the Encoder part to create a classifier and fine tuning upon the second dataset which has 10K rgb images labeled as 'Real' and 10K images labeled as 'Fake'. The second dataset's images are also 256x256 pixels.

## Model Info:
Both models are based on Convolutional Neural Networks a.k. CNNs implemented using Keras API of Tensorflow.

## Repository Structure:
