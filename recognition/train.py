import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io

from pyimagesearch.trafficsignnet import TrafficSignNet

# backend matplotlib
matplotlib.use("Agg")


def load_split(base_path, csv_path):
    # inicjalizacja listy danych i nazw
    data = []
    labels = []

    # ladowanie pliku csv (bez naglowka) i mieszanie wierszy
    rows = open(csv_path).read().strip().split("\n")[1:]
    random.shuffle(rows)

    # iteracja po wieszach pliku csv
    for (i, row) in enumerate(rows):
        # sprawdzanie czy wypisac kolejne info
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {} total images".format(i))

        # rozdzielanie wiersza csv i przypisywanie labela do sciezki
        (label, image_path) = row.strip().split(",")[-2:]

        # ladowanie zdjecia po pelnej sciezce
        image_path = os.path.sep.join([base_path, image_path])
        image = io.imread(image_path)

        # skalowanie zdjecia na 32x32 i robienie dziwnych obliczen
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)

        # aktualizowanie zawartosci list z poczatku funkcji
        data.append(image)
        labels.append(int(label))

    # konwertowanie list na tablicy numpy
    data = np.array(data)
    labels = np.array(labels)

    # funkcja zwraca tuple danych i nazw
    return data, labels


# parser argumentow bo ten program dziala z cmd (za dlugo chodzi)
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input GTSRB")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to training history plot")
args = vars(ap.parse_args())

# todo comments
# initialize the number of epochs to train for, base learning rate,
# and batch size
# inicjalizacja liczby epochs, wspolczynnika i batch size
NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 64

# ladowanie nazw z csv
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [label.split(",")[1] for label in labelNames]

# inicjalizacja sciezek
trainPath = os.path.sep.join([args["dataset"], "Train.csv"])
testPath = os.path.sep.join([args["dataset"], "Test.csv"])

# ladowanie danych do nauki i testowania
print("[INFO] loading training and testing data...")
(trainX, trainY) = load_split(args["dataset"], trainPath)
(testX, testY) = load_split(args["dataset"], testPath)

# skalowanie danych do przedzialu (0,1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# enkodowanie labeli
numLabels = len(np.unique(trainY))
trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

classTotals = trainY.sum(axis=0)
classWeight = classTotals.max() / classTotals

# generator do podmiany plikow aby byly lepsze wyniki
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

# inicjalizacja modelu
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = TrafficSignNet.build(width=32, height=32, depth=3,
                             classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# po prostu nauka sieci z modelu
print("[INFO] training network...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=trainX.shape[0] // BS,
    epochs=NUM_EPOCHS,
    class_weight=classWeight,
    verbose=1)

# wypisywanie raportu do terminala
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=labelNames))

# zapisywanie modelu
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# wykresik ze strat i dokladnosci
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
