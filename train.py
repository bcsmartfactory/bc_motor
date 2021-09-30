

from pandas import DataFrame
import numpy as np
import logging
import sys

from tensorflow import keras
import numpy as np
from utils import utils as utils


# data columns
data_columns = ['result', for i in range(1,600)]
    # --------------------------------------------------
    # get train and test dataset
    # --------------------------------------------------
    x_train, x_test, y_train, y_test = utils.get_train_test_dataset(folder_name, dataset_file_name)

    # --------------------------------------------------
    # make model
    # --------------------------------------------------
    # number of unique classes
    num_classes = len(np.unique(y_train))

    # make model
    model = utils.make_model(input_shape=x_train.shape[1:], num_classes=num_classes)

    """
    ## Train the model
    """
    # --------------------------------------------------
    # Train the model
    # --------------------------------------------------
    history = utils.train_model(model, folder_name, dataset_file_name, x_train, y_train)

