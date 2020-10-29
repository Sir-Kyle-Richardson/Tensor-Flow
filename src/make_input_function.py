import tensorflow as tf

def makeInputFn(dataDf, labelDf, epochs = 10, shuffle = True, batchSize = 32): 
    def inputFn(): # inner function returned
        ds = tf.data.Dataset.from_tensor_slices((dict(dataDf), labelDf))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batchSize).repeat(epochs)
        return ds
    return inputFn