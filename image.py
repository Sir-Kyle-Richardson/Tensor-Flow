import tensorflow as tf             # library for numerical computing created by Google, e.g. to create deep learning models
print("Tensor Flow " + tf.__version__)
from tensorflow import keras        # open-source neural-network library, set of images
import numpy as np                  # adding support for large, multi-dimensional arrays and matrices
print("Numpy " + np.__version__)
import matplotlib.pyplot as plt     # Python 2D plotting library
from PIL import Image               # Python Imaging Library
print("PIL " + Image.__version__)
import os                           # to change path os.chdir('...')


model = tf.keras.models.load_model('my_model.h5')   # loading the trained model
fashion_mnist = keras.datasets.fashion_mnist        # training and test dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()    # loading training and test dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255
test_images = test_images/255

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


def prediction_imag(img, name):                 # prediction for one image, image as a pixels table
    img = (np.expand_dims(img, 0))              # change to vector
    predictions_single = model.predict(img)     # prediction for function argument
    fig = plt.figure()
    X = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    Y = predictions_single[0]                   # probability of each node
    plt.plot(X, Y, 'ro')
    plt.ylabel('Probability(items)')
    plt.xlabel('items')
    plt.grid()
    os.chdir('F:/Mente/tensor_flow/results')
    fig.savefig(str(name))
    plt.close(fig)
    tmp = np.argmax(Y)
    return [tmp, Y[tmp]]


def plot_matrix(img, name):     # save the resize image
    os.chdir('F:/Mente/tensor_flow/28x28')
    fig = plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.colorbar()
    fig.savefig(str(name))
    plt.close(fig)


def train_model():
    model.fit(train_images, train_labels,
        epochs=10)


def compile_model():
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),     # transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels)
        keras.layers.Dense(128, activation='relu'),     # the first dense layer has 128 nodes
        keras.layers.Dense(10, activation='softmax')    # the second layer is a 10-node softmax layer that returns an array of 10 probability scores that sum to 1
    ])                                                  # Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
    return model


def evaluate_model():
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)


"USER PART"
print(items)
print('\n')
print('There is two option of loading image:')
print('0. Exit')
print('1. By give a name of image')
print('2. Series, by give a amount of images called: picture_i.png')
print('3. By give a name of image and re - train this photo')
print('4. Reset')
mode = input('Enter 0, 1, 2, 3 or 4: ')
mode = int(mode)
if mode == 2:
    n = input('How many pictures is in folder pictures: i = ')
    n = int(n)


"SIMULATION"

ii = 0
state = True
os.chdir('F:/Mente/tensor_flow/results')  # path to the results
f = open("results.dat", "w+")

while state:
    name = ''
    if mode == 0:
        break
    if mode == 1:
        name = input("Give a name of file to retrain the program or 0, if exit: ")
        if name == '0':
            break
    if mode == 2:
        ii = ii + 1
        if ii > n:
            break
        name = 'picture_' + str(ii) + '.png'
        print(name)
    if mode == 3:
        name = input("Give a name of file or 0, if exit: ")
        if name == '0':
            break
    if mode == 4:   # LOAD ORIGINAL DATA TEST
        fashion_mnist = keras.datasets.fashion_mnist  # training and test dataset
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # loading training and test dataset
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        train_images = train_images/255
        test_images = test_images/255
        model = build_model()
        compile_model()
        train_model()
        os.chdir('F:/Mente/tensor_flow')
        model.save('my_model.h5')  # save new model
        break
    os.chdir('F:/Mente/tensor_flow/pictures')     # path to the images
    img = Image.open(name)

    if img.size[0] != img.size[1]:      # Does the photo have N x X pixels ?
        x = img.size[0]
        y = img.size[1]
        if abs(x-y)/x < 0.1 and abs(x-y)/y < 0.1:       # auto - correction if photo is close to square
            if x > y:
                tmp = int((x-y)/2)
                cropped = img.crop((tmp, 1, y+tmp-1, y))      # crop((left, top, right, bottom))
                cropped.save(name)
                img = cropped
            if y > x:
                tmp = int((y-x)/2)
                cropped = img.crop((1, tmp, x, x+tmp-1))      # crop((left, top, right, bottom))
                cropped.save(name)
                img = cropped
        else:
            print('The image is not a square (auto-correction for max 10% difference: x-y)')

    img.thumbnail((28, 28), Image.ANTIALIAS)    # resizes image for a neural - network set
    img = np.asarray(img)[:, :, 0]

    img = img/255       # color -> black and white
    i = 0
    j = 0
    for i in range(28):
      for j in range(28):
            img[i, j] = abs(1-img[i, j])

    plot_matrix(img, name)                # save prepared image
    pred = prediction_imag(img, name)     # prediction for loaded image
    print(name + ' with ' + str(round(pred[1]*100, 1)) + '% is a ' + str(items[pred[0]]))
    f.write(name + ' with ' + str(round(pred[1]*100, 1)) + '% is a ' + str(items[pred[0]]) + '\n')

    if pred[1] < 0.5 or mode == 3:
        print('Do you want retrain neural-network for ' + name)
        choice = input('Enter \'yes\' or anything else, if no: ')
        if choice == 'yes':
            print(items)
            number = input("Put 0,1...9 to point right class: ")
            tmp1 = np.ones((1, 28, 28))
            tmp1[0] = img
            tmp2 = int(number)
            train_images = np.concatenate((train_images, tmp1))
            train_labels = np.append(train_labels, tmp2)
            model = build_model()
            compile_model()
            train_model()
            os.chdir('F:/Mente/tensor_flow')
            model.save('my_model.h5')  # save new model
f.close()