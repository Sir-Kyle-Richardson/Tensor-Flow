try:
    import sequential_model as sm
    import tensorflow as tf
    import numpy as np
except ModuleNotFoundError:
    print('Program requires python module to be installed')
    exit(1)


def runNameMode(nameOfImage):
    model = sm.SequentialModel()
    image = sm.ClassificationImage(nameOfImage)
    pred = model.runAndSavePrediction(image.getImage(), nameOfImage)


def runNamePlusMode(nameOfImage, numberOfRetrainImage):
    if numberOfRetrainImage not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print("Wrong image indicator")
        exit(1)
    model = sm.SequentialModel()
    image = sm.ClassificationImage(nameOfImage)
    shape = sm.ClassificationImage.standardShape
    retrainPattern = np.ones((1, shape, shape))
    retrainPattern[0] = image.getImage()
    model.updateTrainData(retrainPattern, numberOfRetrainImage)
    model.resetModel()
    model.saveModel()


def runSeriesMode(numberOfImages):
    model = sm.SequentialModel()
    integerList = [i for i in range(1, numberOfImages+1)]
    listOfNames = [f"image_{i}.png" for i in integerList]
    for i in listOfNames:
        image = sm.ClassificationImage(i)
        pred = model.runAndSavePrediction(image.getImage(), i)


def runResetMode():
    model = sm.SequentialModel()
    model.resetModel()
    model.saveModel()
