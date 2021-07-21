from tensorflow.keras import layers, utils, optimizers
from tensorflow.keras.models import Sequential, Model, load_model, model_from_json
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from skimage.io import imread
from sklearn.metrics import confusion_matrix
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf


# Assure Reproducibility
from tensorflow import random

np.random.seed(1337)
random.set_seed(1337)

# tf.debugging.set_log_device_placement(True)

# Generators for batching data
def image_gen(imgs_paths):
    # Iterate over all the image paths
    for image_file in imgs_paths:

        # Load the image and mask, and normalize it to 0-1 range
        img = imread(image_file) / 255.0

        # Yield the image mask pair
        yield img


# pass as input a lsit with images paths a list with integer numbers per class, and a batch size
# imgs_paths must have the same length with labels
def image_batch_generator(imgs_paths, labels, batchsize=32):
    while True:
        ig = image_gen(imgs_paths)
        batch_img, batch_labels = [], []

        for img, label in zip(ig, labels):
            # Add the image and mask to the batch
            batch_img.append(img)
            batch_labels.append(label)
            # If we've reached our batchsize, yield the batch and reset
            if len(batch_img) == batchsize:
                yield np.stack(batch_img, axis=0), np.stack(
                    utils.to_categorical(batch_labels), axis=0
                )
                batch_img, batch_labels = [], []

        # If we have an nonempty batch left, yield it out and reset
        if len(batch_img) != 0:
            yield np.stack(batch_img, axis=0), np.stack(
                utils.to_categorical(batch_labels), axis=0
            )
            batch_img, batch_labels = [], []


# load model
# with open("model_arch_autoen.json", "r") as f:
#     jf = f.read()
my_model = model_from_json(open("model_arch_autoen.json").read())
print("Original AutoEncoder Model summary as two Sequentials \n")
my_model.summary()
my_model.load_weights("cnn3_autoenc_weights.h5")  # load weights

# Create model from using encoder only
encoder = Sequential(
    [layers.InputLayer([256, 256, 3]), my_model.get_layer("sequential_13")]
)

print("\n Create again encoder model with the same weights as trained \n")
encoder.summary()
encoder_outshape = encoder.layers[-1].output_shape

# Add extra layers
"""#**Fully Connected Block** {Classifier}"""


num_class = 2  # real or fake


fc_block = Sequential(
    [
        layers.InputLayer(encoder_outshape[1:]),
        layers.Conv2D(
            filters=64, kernel_size=3, padding="same"
        ),  # Learn 2D Representations
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(80, activation="relu"),
        layers.Dropout(0.2),
        # layers.Dense(256,activation='relu'),
        # layers.Dropout(0.2),
        layers.Dense(30, activation="relu"),
        # layers.Dropout(0.2),
        # keras.layers.Dense(64,activation='relu'),
        layers.Dense(num_class, activation="softmax"),
    ]
)

fc_block.summary()
utils.plot_model(fc_block, show_shapes=True)


# Read data
# load path of real images data
folderReal = "dataset2/real"
train_img_paths = [
    os.path.join(folderReal, filename) for filename in os.listdir(folderReal)
]
labels = [1] * len(train_img_paths)  # 1 is for Real image label
print(train_img_paths[:10])
print(labels)
print(len(labels))

# load paths of fake ones
folderFake = "dataset2/fake"
train_img_paths_fake = [
    os.path.join(folderFake, filename) for filename in os.listdir(folderFake)
]

train_img_paths.extend(train_img_paths_fake)  # add fake images paths
labels.extend([0] * len(train_img_paths_fake))  # 0 zero is for Fake image label
print("Fake images number", len(train_img_paths_fake))

print("\n |> Whole images fakes and real ones, ", train_img_paths, "\n ", labels)
print(
    "\n |> Size of all images fake and real img_paths_size =",
    len(train_img_paths),
    "Labels size =",
    len(labels),
)

# ohe_labels =utils.to_categorical(labels) # label 1 is converted to [0. 1.]--Real and 0 is converted to [1. 0.] --Fake
# print('\n Ohe Labels \n',type(ohe_labels),ohe_labels)

# img=imread(train_img_paths[0])
# print(img.shape)
# # Split the data into a train and validation set
train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(
    train_img_paths, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Check if set are splitted correctly
print("\n Train set Labels ", train_labels)
print("\n Size of training data ", len(train_labels), len(train_img_paths))
print("\n Training Images paths ", train_img_paths)

print("\n Number of Real Images ", len([x for x in train_img_paths if "real" in x]))
print(
    "\n Number of Real Images counting labels ",
    len([x for x in train_labels if x == 1]),
)

# Split valid set into half and create valid and test set
test_img_paths, val_img_paths, test_labels, val_labels = train_test_split(
    val_img_paths, val_labels, test_size=0.5, shuffle=True, stratify=val_labels
)
print(
    "\n *Size of each test and validation sets 1% of the total(20k)",
    len(test_img_paths),
    len(val_img_paths),
)


# TRAIN
BATCHSIZE = 128
# Create the train and validation generators
traingen = image_batch_generator(train_img_paths, train_labels, batchsize=BATCHSIZE)
valgen = image_batch_generator(val_img_paths, val_labels, batchsize=BATCHSIZE)


def calc_steps(data_len, batchsize):
    return (data_len + batchsize - 1) // batchsize


# Calculate the steps per epoch
train_steps = calc_steps(len(train_img_paths), BATCHSIZE)
val_steps = calc_steps(len(val_img_paths), BATCHSIZE)

encoder.trainable = False
stacked_classifier = Sequential([encoder, fc_block])

opt = optimizers.Adam(learning_rate=0.01)
csv_logger = CSVLogger("train_classifier_log2.csv", append=True, separator=",")
# Compile the stacked model and train with adam
stacked_classifier.compile(
    loss="binary_crossentropy",
    optimizer=opt,
    metrics=[
        "accuracy",
        AUC(),
        Precision(),
        Recall(),
    ],
)
# Train the model
history_classifier = stacked_classifier.fit(
    traingen,
    steps_per_epoch=train_steps,
    epochs=70,  # Change this to a larger number to train for longer
    validation_data=valgen,
    validation_steps=val_steps,
    verbose=1,
    callbacks=[csv_logger],
    max_queue_size=10,  # Change this number based on memory restrictions
)


# Save models
stacked_classifier.save("save_points/cnn_classifierB70ep.h5")
stacked_classifier.save_weights("save_points/cnn_classifier_weightsB70ep.h5")
json_arch = stacked_classifier.to_json()
jsonFile = open("save_points/model_arch_classifier.json", "w")
jsonFile.write(json_arch)
jsonFile.close()


# PLOTS
# plt.plot(history_classifier.history["accuracy"])
# plt.plot(history_classifier.history["val_accuracy"])
# plt.title("model accuracy")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.legend(["train", "val"], loc="upper left")
# plt.show()
#
# plt.plot(history_classifier.history["loss"])
# plt.plot(history_classifier.history["val_loss"])
# plt.title("model loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.legend(["train", "val"], loc="upper left")
# plt.show()
#
# plt.plot(history_classifier.history["auc"])
# plt.plot(history_classifier.history["val_auc"])
# plt.title("model AUC")
# plt.ylabel("TP")
# plt.xlabel("FP")
# plt.legend(["train_auc", "val_auc"], loc="upper left")
# plt.show()
