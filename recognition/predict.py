# USAGE
# python recognition/predict.py --model output/trafficsignnet.model
#                               --images gtsrb-german-traffic-sign/Test
#                               --examples examples

import os
import random
import cv2
import imutils
import numpy as np
from imutils import paths
from skimage import exposure
from skimage import io
from skimage import transform
from tensorflow.keras.models import load_model


class Predict:
    def __init__(self, model, examples, csv, images):
        self.model = model
        self.examples = examples
        self.csv = csv
        self.images = images

    def predict(self):
        # ladowanie folderu .model
        print("[INFO] loading model...")
        model = load_model(self.model)

        # ladowanie nazw z pliku csv do zmiennej
        label_names = open(self.csv).read().strip().split("\n")[1:]
        label_names = [label.split(",")[1] for label in label_names]

        # mieszanie sciezek zdjec wejsciowych
        print("[INFO] predicting...")
        image_paths = list(paths.list_images(self.images))
        random.shuffle(image_paths)
        image_paths = image_paths[:25]

        # petla po sciezkach zdjec
        for (i, imagePath) in enumerate(image_paths):
            # ladowanie zdjecia i przeskalowanie go na 32x32
            image = io.imread(imagePath)
            image = transform.resize(image, (32, 32))
            image = exposure.equalize_adapthist(image, clip_limit=0.1)

            # skalowanie zdjecia
            image = image.astype("float32") / 255.0
            image = np.expand_dims(image, axis=0)

            # przewidywanie na podstawie modelu
            preds = model.predict(image)
            j = preds.argmax(axis=1)[0]
            label = label_names[j]

            # ladowanie i rysowanie nazwy na zdjeciu
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=128)
            cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 2)

            # zapisywanie zdjecia na dysku
            p = os.path.sep.join([self.examples, "{}{}.png".format(label, i)])
            cv2.imwrite(p, image)
        print("END")
