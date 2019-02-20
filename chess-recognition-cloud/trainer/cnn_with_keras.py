from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import pickle
import keras.backend as K
from keras import callbacks
import tensorflow as tf
import time

K.set_image_data_format('channels_last')
import argparse
from tensorflow.python.lib.io import file_io


def main(job_dir,**args):

    ##Setting up the path for saving logs
    logs_path = job_dir + 'logs/tensorboard-{}'.format(int(time.time()))

    logs_dir = job_dir + 'logs/tensorboard/'

    ##Using the GPU
    with tf.device('/device:GPU:0'):

        with file_io.FileIO(job_dir + 'dataset/train_data.pickle', mode='rb') as file:
            train_set = pickle.load(file)

        with file_io.FileIO(job_dir + 'dataset/train_label.pickle', mode='rb') as file:
            train_label = pickle.load(file)

        dense_layers = [0, 1, 2]
        layer_sizes = [8, 16, 32, 64]
        conv_layers = [1, 2, 3]

        for dense_layer in dense_layers:
            for layer_size in layer_sizes:
                for conv_layer in conv_layers:
                    NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))

                    model = Sequential()

                    model.add(Conv2D(layer_size, (3, 3), input_shape=(64, 64, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                    for l in range(conv_layer - 1):
                        model.add(Conv2D(layer_size, (3, 3)))
                        model.add(Activation('relu'))
                        model.add(MaxPooling2D(pool_size=(2, 2)))

                    model.add(Flatten())
                    for _ in range(dense_layer):
                        model.add(Dense(layer_size))
                        model.add(Activation('relu'))

                    model.add(Dense(13))
                    model.add(Activation('sigmoid'))

                    tensorboard = callbacks.TensorBoard(log_dir=logs_dir + "{}".format(NAME))

                    # checkpoint = ModelCheckpoint('model-checkpoint-{}.h5'.format(NAME),
                    #                              monitor='val_loss', verbose=1,
                    #                              save_best_only=True, mode='min')

                    model.compile(loss='categorical_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'],
                                  )

                    model.fit(train_set, train_label,
                              batch_size=40,
                              epochs=15,
                              validation_split=0.3,
                              callbacks=[tensorboard])

                    model.save('model-{}.h5'.format(NAME))
                    with file_io.FileIO('model-{}.h5'.format(NAME), mode='r') as input_f:
                        with file_io.FileIO(job_dir + 'model/' + 'model-{}.h5'.format(NAME), mode='w+') as output_f:
                            output_f.write(input_f.read())




        # classifier = get_classifier()
        # # Compiling the CNN
        # classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        ## Adding the callback for TensorBoard and History
        # tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
        # checkpoint = ModelCheckpoint(job_dir + 'model/chess-rec-model-checkpoint.h5', monitor='val_acc', verbose=1,
        #                              save_best_only=True, mode='max')

        # train_set = data_set[:2*(len(data_set)/3)]
        # train_label = data_label[:2*(len(data_set)/3)]
        #
        # validation_set = data_set[2*(len(data_set)/3):]
        # validation_label = data_set[2 * (len(data_set) / 3):]

        # classifier.fit(train_set,
        #                train_label,
        #                batch_size=40,
        #                epochs=30,
        #                callbacks=[tensorboard],
        #                validation_split=0.3)

        ##fitting the model
        # Model.fit(x = train_data, y = train_labels, epochs = 50,verbose = 1, batch_size=100, callbacks=[tensorboard], validation_data=(eval_data,eval_labels) )

        # Save model.h5 on to google storage
        # classifier.save('chess-rec-model3.h5')
        # with file_io.FileIO('chess-rec-model3.h5', mode='r') as input_f:
        #     with file_io.FileIO(job_dir + 'model/chess-rec-model3.h5', mode='w+') as output_f:
        #         output_f.write(input_f.read())


##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
