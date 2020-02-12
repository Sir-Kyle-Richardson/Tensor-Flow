# IMPORT OF LIBRARIES
try:
    import tensorflow as tf             # library for numerical computing created by Google, e.g. to create deep learning models
    print("Tensor Flow " + tf.__version__)
except:
    lib_1 = 'Tensor Flow'
    print('Program requires ' + lib_1 + ' python module to be installed')
    print('Recommended' + lib_1 + '2.0.0')
    exit(1)
from tensorflow import keras            # open-source neural-network library, set of images
try:
    import numpy as np                  # adding support for large, multi-dimensional arrays and matrices
    print("Numpy " + np.__version__)
except:
    lib_2 = 'Numpy'
    print('Program requires ' + lib_2 + ' python module to be installed')
    print('Recommended' + lib_2 + '1.17.4')
    exit(1)
try:
    import matplotlib.pyplot as plt     # Python 2D plotting library
except:
    lib_3 = 'Matplotlib'
    print('Program requires ' + lib_3 + ' python module to be installed')
    exit(1)
try:
    from PIL import Image                # Python Imaging Library
    print("PIL " + Image.__version__)
except:
    lib_4 = 'PIL'
    print('Program requires ' + lib_4 + ' python module to be installed')
    print('Recommended' + lib_4 + '6.2.1')
    exit(1)
try:
    import argparse
except:
    lib_5 = 'Argparse'
    print('Program requires ' + lib_5 + ' python module to be installed')
    exit(1)
import os                                # to change path os.chdir('...')

# ARGPARSE, module make command-line interfaces, generates a help option
parser = argparse.ArgumentParser(description='Neural network for image recognition')                # initialise the parser
parser.add_argument('-m', '--mode', type=str,
                    help='mode for program run: \'name\' or \'name+\' or \'series\' or \'reset\'')  # adding parameters
parser.add_argument('-n', '--number', type=int,
                    help='number of pictures named \'picture_i.png\' for series mode')
group = parser.add_mutually_exclusive_group()
group.add_argument('-ns', '--not_save', action='store_true',
                   help='not save prediction images to file, e.g. in case of a lot of pictures')
args = parser.parse_args()      # PARSE

# GLOBAL VARIABLES
main_folder = os.getcwd()                           # path to main folder with image.py
rgba = 255                                          # max value for color matrix
shape = 28                                          # number of pixels for network image
class_number = 10                                   # number of classes, f({0,1...9}) = 10
node_number = 128                                   # number of nodes for first dense layer of neural network
verb = 2                                            # value for the look of model in evaluate,
                                                    # if 0 = silent, 1 = partial progress, 2 = see progress
max_variation = 0.1                                 # max image variation from a square, 0.1 <-> 10%
model = tf.keras.models.load_model('my_model.h5')   # loading the trained model
fashion_mnist = keras.datasets.fashion_mnist        # training and test dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()    # loading training and test dataset
train_images = train_images/rgba
test_images = test_images/rgba
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
items = {               # dictionary of items
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


# FUNCTIONS
def prediction_imag(img, name):                                 # prediction for one image, image as a pixels table
    img = (np.expand_dims(img, 0))                              # change to vector
    predictions_single = model.predict(img)                     # prediction for function argument
    fig = plt.figure()
    X = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')      # there is 9 class for items
    Y = predictions_single[0]                                   # probability of each node
    plt.plot(X, Y, 'ro')
    plt.ylabel('Probability(items)')
    plt.xlabel('items')
    plt.grid()
    os.chdir(main_folder + '/results')
    if args.not_save != True:                                   # save if there is no command 'not save'
        fig.savefig(str(name))
    plt.close(fig)
    tmp = np.argmax(Y)
    return [tmp, Y[tmp]]


def plot_matrix(img, name):             # save the resize image
    os.chdir(main_folder + '/28x28')
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
        keras.layers.Flatten(input_shape=(shape, shape)),       # transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels)
        keras.layers.Dense(node_number, activation='relu'),     # the first dense layer has 128 nodes
        keras.layers.Dense(class_number, activation='softmax')  # the second layer is a 10-node softmax layer that returns an array of 10 probability scores that sum to 1
    ])                                                          # Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
    return model


def evaluate_model():
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=verb)
    print('\nTest accuracy:', test_acc)


"USER PART"
print(items)
print('\n')
if args.mode != None:
    mode = args.mode
    if (mode == 'name' or mode == 'series' or
        mode == 'name+' or mode == 'reset' or mode == 'exit') == 0:   # instead of NOR;
        print('Wrong command option for mode')                        # the OR gate does not suit here
        exit(1)
else:
    print('Options of loading image:')
    print('1. By give a name of image, command \'name\'')
    print('2. By give a name of image and re-train this photo, command \'name+\'')
    print('3. By give a amount of images called: picture_i.png, command \'series\'')
    print('4. command \'reset\' (to initial settings)')
    print('5. command \'exit\'')
    while True:     # loading options by user
        mode = input('Choose option: ')
        if mode == 'name' or mode == 'series' or mode == 'name+' or mode == 'reset' or mode == 'exit':
            break
        else:
            print('Wrong command option for mode')

if mode == 'series':
    ii = 0                      # counter for series mode
    if args.number != None:
        n = args.number
    else:
        n = input('How many pictures is in folder pictures: i = ')
        try:
            n = int(n)
        except:
            print('Number of pictures for series mode is not a \'int\' type')
            exit(1)


"SIMULATION"
state = True
os.chdir(main_folder + '/results')  # path to the results
f = open("results.dat", "w+")

while state:
    name = ''                       # name of the picture
    if mode == 'name':
        name = input("Give a \'name\' of file or \'exit\', if exit: ")
        if name == 'exit':
            break
    if mode == 'name+':
        name = input("Give a \'name\' of file to retrain the program or \'exit\', if exit: ")
        if name == 'exit':
            break
    if mode == 'series':
        ii = ii + 1
        if ii > n:
            break
        name = 'picture_' + str(ii) + '.png'
        print(name)
    if mode == 'reset':   # LOAD ORIGINAL DATA TEST
        fashion_mnist = keras.datasets.fashion_mnist  # training and test dataset
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # loading training and test dataset
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        train_images = train_images/rgba
        test_images = test_images/rgba
        model = build_model()
        compile_model()
        train_model()
        os.chdir(main_folder)
        model.save('my_model.h5')  # save new model
        break
    if mode == 'exit':
        break
    os.chdir(main_folder + '/pictures')     # path to the images
    try:
        img = Image.open(name)
    except:
        print('Wrong file name')
        exit(1)

    if img.size[0] != img.size[1]:                                       # Does the photo have N x X pixels ?
        x = img.size[0]                                                  # pixels number of image from OX
        y = img.size[1]                                                  # pixels number of image from OY
        if abs(x-y)/x < max_variation and abs(x-y)/y < max_variation:    # auto-correction if photo is close to square
            if x > y:                                                    # choose longer side
                tmp = int((x-y)/2)                                       # temporary variable to symmetric image cut
                cropped = img.crop((tmp, 1, y+tmp-1, y))                 # crop((left, top, right, bottom)), cut for size: y x y
                img = cropped
            if y > x:
                tmp = int((y-x)/2)
                cropped = img.crop((1, tmp, x, x+tmp-1))                 # crop((left, top, right, bottom)), cut for size: X x X
                img = cropped
        else:
            print('The image is not a square (auto-correction for max 10% difference: x-y)')

    img.thumbnail((shape, shape), Image.ANTIALIAS)          # resize image for a neural - network set
    img = np.asarray(img)[:, :, 0]

    img = img/rgba                          # color -> black and white
    i = 0
    j = 0
    for i in range(shape):
      for j in range(shape):
            img[i, j] = abs(1-img[i, j])    # color correction -> white background of image
    if args.not_save != True:
        plot_matrix(img, name)              # save prepared image if there is no command to not
    pred = prediction_imag(img, name)       # prediction for loaded and prepared image
    print(name + ' with ' + str(round(pred[1]*100, 1)) + '% is a ' + str(items[pred[0]]))
    f.write(name + ' with ' + str(round(pred[1]*100, 1)) + '% is a ' + str(items[pred[0]]) + '\n')

    if pred[1] < 0.5 or mode == 'name+':        # if prediction smaller than 50% or user wish: re-train neural-network
        print('Do you want retrain neural-network for ' + name + ' ?')
        choice = input('Enter \'yes\' or anything else if no: ')
        if choice == 'yes':
            print(items)
            while True:
                try:
                    tmp2 = input("Put \'0,1...9\' to point right class: ")
                    tmp2 = int(tmp2)
                    break
                except:
                    print('Number of item is not a \'int\' type')
            tmp1 = np.ones((1, shape, shape))
            tmp1[0] = img
            train_images = np.concatenate((train_images, tmp1))
            train_labels = np.append(train_labels, tmp2)
            model = build_model()
            compile_model()
            train_model()
            os.chdir(main_folder)
            model.save('my_model.h5')  # save new model
f.close()