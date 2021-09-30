
import os
import cv2
import yaml
import datetime
import numpy as np
import pandas as pd
from tensorflow import keras
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def get_data_list(element=None):
    # signal_datetime list
    signal_datetime_arr = element.get("signal_datetime")
    # start_gap list
    start_gap_arr = element.get("start_gap")
    # end_gap list
    end_gap_arr = element.get("end_gap")
    # day_shift
    day_shift_arr = element.get("day_shift")
    # zero_ratio
    zero_ratio_arr = element.get("zero_ratio")

    # result data list
    data_list = []

    # make loop
    for i in range(len(signal_datetime_arr)):

        signal_datetime = signal_datetime_arr[i]
        start_gap = start_gap_arr[i]
        end_gap = end_gap_arr[i]
        day_shift = day_shift_arr[i]
        zero_ratio = zero_ratio_arr[i]

        # element data info
        dataInfo = DataInfo(signal_datetime, start_gap, end_gap, day_shift, zero_ratio)

        # data_list append
        data_list.append(dataInfo)

    return data_list


def make_csv_dataset_file_format(data_row_arr, window_size, target_val, zero_ratio):

    # Make a data list for making csv file
    row_list_for_csv = []

    # loop for making csv dataset file format
    for i, window in enumerate(data_row_arr):

        # cut and make windows data
        row_data_arr_tmp = data_row_arr[i:i + window_size]

        row_data_arr_tmp_list = row_data_arr_tmp.tolist()
        # if the number of the value '0' is over the ratio specified in config file
        if row_data_arr_tmp_list.count(0) > window_size * zero_ratio:
            continue

        if row_data_arr_tmp_list.count(0) < window_size * zero_ratio:
            print(row_data_arr_tmp_list.count(0)/window_size)

        # if length of array is less than window_size, execute break
        if len(row_data_arr_tmp) < window_size:
            break

        # add target_val value as a column of target value locating index 0
        # reshape row data
        np_row_data_arr_tmp = np.insert(np.array(row_data_arr_tmp), 0, target_val).reshape(1, window_size+1)

        # convert numpy array to list
        lw44_row_tmp_list = np_row_data_arr_tmp.tolist()

        # append rows into list for creating csv file
        row_list_for_csv.append(lw44_row_tmp_list[0])

    return row_list_for_csv



def make_csv_dataset(signal_log_list, data_columns, column_name, target_val, window_size, csv_dataset_file_name, zero_ratio):

    # Dataframe 생성
    df = DataFrame(signal_log_list, columns=data_columns)

    # Save csv dataset file
    save_csv_file_with_column_name(df, column_name, window_size, target_val, csv_dataset_file_name, zero_ratio)



def make_predict_dataset(signal_log_list, data_columns, column_name, window_size):

    # Dataframe 생성
    df = DataFrame(signal_log_list, columns=data_columns)

    # get column data array
    column_data_arr = df[column_name]

    # reshape column data to row data
    np_row_data_arr_return = np.array(column_data_arr, dtype=np.float64).reshape(1, window_size)

    return np_row_data_arr_return

'''
    Function: Read csv dataset file
    INPUT
        file_name: dataset file name
    OUTPUT
        x: x data array
        y: y data array
'''


def read_csv_dataset(file_name):
    # data = np.loadtxt(filename, delimiter="\t")
    data = np.loadtxt(file_name, delimiter=",")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


'''
    file 존재 체크
'''


def is_file(file_path):
    return_val = os.path.isfile(file_path)
    return return_val


'''
    delete file
'''


def del_file(file_path):
    os.remove(file_path)


'''
    Function: make model
    INPUT
        input_shape: x
        num_classes: number of unique classes
    OUTPUT
        model: keras model
'''


def make_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def get_train_test_dataset(folder_name, dataset_file_name):
    # get csv dataset file name
    csv_dataset_file_name = "{}/{}.csv".format(folder_name, dataset_file_name)

    # csv dataset file
    x, y = read_csv_dataset(csv_dataset_file_name)

    # split data into train, test dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=4)

    print(x_train)
    print(y_train)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    return x_train, x_test, y_train, y_test



'''
    Function: train model
    INPUT
        model:
        dataset_file_name:
        x_train:
        y_train:
    OUTPUT
        history: the result of the train
'''
def train_model(model, folder_name, dataset_file_name, x_train, y_train):
    epochs = 500
    batch_size = 32

    h5_file_path = "{}/{}.h5".format(folder_name, dataset_file_name)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            h5_file_path, save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    return history


