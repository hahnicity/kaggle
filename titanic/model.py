from os.path import abspath, dirname, join

import numpy
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier


def get_training_data(data_set):
    data_set = get_data(data_set)
    y = data_set["Survived"]
    del data_set["Survived"]
    return data_set, y


def get_data(data_set):
    data_set["Sex"] = data_set["Sex"] == "male"
    data_set = data_set[numpy.isfinite(data_set.Age)]  # XXX Don't filter out NaNs
    data_set.index = range(len(data_set))
    del data_set["Name"]
    del data_set["PassengerId"]
    del data_set["Fare"]
    del data_set["Ticket"]
    # This is a questionable deletion. But do it for now
    #
    # I guess the only thing that would make it relevant is if you
    # could get some map of the ship
    del data_set["Cabin"]
    del data_set["Embarked"]
    return data_set


def main():
    working_dir = dirname(abspath(__file__))
    training_path = join(working_dir, "train.csv")
    training_set, y = get_training_data(read_csv(training_path))
    classifier = RandomForestClassifier()
    classifier.fit(training_set, y)
    testing_path = join(working_dir, "test.csv")
    testing_set = get_data(read_csv(testing_path))
    predictions = classifier.predict(testing_set)
    import pdb; pdb.set_trace()
