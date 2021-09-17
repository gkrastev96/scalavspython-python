# deeper cnn model for mnist
from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD


# data order is important here, consider sorting the columns to ensure that
# you get the same result every time. (problem arises when you want to sort as
# the columns are string and you want to sort by their integer values)
def transform_to_tensors(row, dim1, dim2):
    return row.reshape((dim1, dim2, 1))


# load train and test dataset
def load_transform_encode_dataset(dir, train_name, test_name):
    train = pd.read_csv(dir + train_name)
    test = pd.read_csv(dir + test_name)
    train_output = to_categorical(train["label"])
    test_output = to_categorical(test["label"])
    apply_config = {
        "func1d": transform_to_tensors,
        "axis": 1,
        "dim1": 28,
        "dim2": 28,
    }
    train_input = np.apply_along_axis(
        arr=train.drop("label", axis=1).to_numpy(),
        **apply_config
    )
    test_input = np.apply_along_axis(
        arr=test.drop("label", axis=1).to_numpy(),
        **apply_config
    )
    # one hot encode target values
    return train_input, train_output, test_input, test_output


def normalize_data(train, test):
    # normalize to range 0-1
    train_norm = train / 255.0
    test_norm = test / 255.0
    # return normalized images
    return train_norm, test_norm


def makeLeNet5():
    model = Sequential()
    model.add(layers.Conv2D(
        6, kernel_size=(5, 5),
        strides=(1, 1), activation='tanh',
        input_shape=(28, 28, 1))
    )
    model.add(layers.AveragePooling2D(
        pool_size=(2, 2), strides=(2, 2), padding='valid')
    )
    model.add(layers.Conv2D(
        16, kernel_size=(5, 5), strides=(1, 1), activation='tanh')
    )
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='tanh'))
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(10, activation='softmax'))

    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    # compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Train model using train test split for validation while training.
def train_model(model, data_input, data_output, test_size):
    # prepare cross validation
    train_input, test_input, train_output, test_output = train_test_split(
        data_input, data_output, test_size=test_size, random_state=5
    )
    return model.fit(
        train_input,
        train_output,
        epochs=10,
        batch_size=32,
        validation_data=(test_input, test_output),
        verbose=0
    )


# run the test harness for evaluating a model
def run_test_harness(dir, train_name, test_name):
    # load dataset
    train_input, train_output, test_input, test_output = load_transform_encode_dataset(dir, train_name, test_name)
    # prepare pixel data
    train_input, test_input = normalize_data(train_input, test_input)
    # # evaluate model
    model = makeLeNet5()
    train_model(model, train_input, train_output, test_size=0.2)
    scores = model.evaluate(test_input, test_output, return_dict=True)
    print("Evaluation results:")
    pprint(scores)


if __name__ == '__main__':
    # entry point, run the test harness
    run_test_harness(
        "/home/joreto360/Workspace/motion-blogpost/mnist_data/mnist_csv/",
        "mnist_train.csv",
        "mnist_test.csv",
    )
