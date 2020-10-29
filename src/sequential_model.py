try:
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import os
except ModuleNotFoundError:
    print("Program requires python module to be installed")
    exit(1)


class SequentialModel:
    rgbaNormalization = 255
    items = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    def __init__(self):
        self.model = tf.keras.models.load_model("my_model.h5")
        self.setTrainTestData()

    def setTrainTestData(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (
            self.test_images,
            self.test_labels,
        ) = fashion_mnist.load_data()
        self.train_images = self.train_images / SequentialModel.rgbaNormalization
        self.test_images = self.test_images / SequentialModel.rgbaNormalization

    def updateTrainData(self, newImage, imageNumber):
        self.train_images = np.concatenate((self.train_images, newImage))
        self.train_labels = np.append(self.train_labels, imageNumber)

    def resetModel(self):
        self.model = tf.keras.Sequential()
        self.buildModel()
        self.compileModel()
        self.trainModel()
        self.evaluateModel()

    def buildModel(self):
        imageShape = 28
        classNumber = 10
        nodesNumber = 128
        self.model.add(keras.layers.Flatten(input_shape=(imageShape, imageShape)))
        self.model.add(keras.layers.Dense(nodesNumber, activation="relu"))
        self.model.add(keras.layers.Dense(classNumber, activation="softmax"))

    def compileModel(self):
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def trainModel(self):
        self.model.fit(self.train_images, self.train_labels, epochs=10)

    def evaluateModel(self):
        loss, acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print("\nTest accuracy:", acc)

    def saveModel(self):
        self.model.save("my_model.h5")

    def runAndSavePrediction(self, image, name):
        imgVector = np.expand_dims(image, 0)
        singlePrediction = self.model.predict(imgVector)
        fig = plt.figure()
        X = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
        Y = singlePrediction[0]
        plt.plot(X, Y, "ro")
        plt.ylabel("Probability(items)")
        plt.xlabel("items")
        plt.grid()
        fig.savefig(fr"./results/{name}")
        plt.close(fig)
        _ = np.argmax(Y)
        print(f"{name} with {round(Y[_]*100)}% is {SequentialModel.items[_]}")
