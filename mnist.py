from tensorflow.keras.datasets import mnist
# Importing the four most important things lol
from model.aggregator import Aggregator
from model.logger import Log
from model.round import Round
from model.user import User
# for shuffling purposes
import random

# first, load the data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# define important variables
NUM_USERS = 10

# Randomly associating each user with a subset of training data id
uid_to_data_ids = {}
ids, last_id_inclusive = list(range(X_train.shape[0])), 0
interval = int(X_train.shape[0]/NUM_USERS)
# shuffle ids
random.shuffle(ids)

for u in range(NUM_USERS):
    uid_to_data_ids[u] = ids[last_id_inclusive:last_id_inclusive+interval]
    last_id_inclusive += interval

