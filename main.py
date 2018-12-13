import os
import numpy as np

from data_helpers import load_data
from keras import callbacks
from keras import backend as K
from capsule_net import CapsNet


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))


def train(model, train, test, save_directory):
    (X_train, Y_train)= train
    (X_test, Y_test) = test

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(filepath=save_directory + '/weights-improvement-{epoch:02d}.hdf5',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    # compile the model
    model.compile(optimizer='adam',
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.0005],
                  metrics={'out_caps': 'accuracy'})

    model.fit([X_train, Y_train], [Y_train, X_train], batch_size=200, epochs=20,
              validation_data=[[X_test, Y_test], [Y_test, X_test]], callbacks=[lr_decay, checkpoint], verbose=1)


if __name__ == "__main__":
    if not os.path.exists("./result"):
        os.makedirs("./result")

    databases = ["CR", "MR", "SST-1", "SST-2", "SUBJ", "TREC", "IMDB",]

    # Train
    for d in databases:
        print(d)
        # Load data
        (x_train, y_train), (x_test, y_test), vocab_size, max_len = load_data(d)

        model = CapsNet(input_shape=x_train.shape[1:],
                        n_class=len(np.unique(np.argmax(y_train, 1))),
                        num_routing=3,
                        vocab_size=vocab_size,
                        embed_dim=50,
                        max_len=max_len
                        )

        model.summary()

        train(model,
              train=(x_train, y_train),
              test=(x_test, y_test),
              save_directory="./result"
              )