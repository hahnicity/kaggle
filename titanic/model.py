import csv
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
    data_set.Age = data_set.Age.map(lambda x: -1 if numpy.isnan(x) else x)
    data_set.Fare = data_set.Fare.map(lambda x: -1 if numpy.isnan(x) else x)
    del data_set["Name"]
    del data_set["PassengerId"]
    del data_set["Ticket"]
    del data_set["Cabin"]
    del data_set["Embarked"]
    return data_set


def write_output(predictions):
    # passenger id is essentially the DataFrame id plus 892
    with open("titanic-results.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["PassengerId", "Survived"])
        for idx, val in enumerate(predictions):  # predictions should be an array
            writer.writerow([idx + 892, val])


def main():
    working_dir = dirname(abspath(__file__))
    training_path = join(working_dir, "train.csv")
    training_set, y = get_training_data(read_csv(training_path))
    classifier = RandomForestClassifier()
    classifier.fit(training_set, y)
    testing_path = join(working_dir, "test.csv")
    testing_set = get_data(read_csv(testing_path))
    predictions = classifier.predict(testing_set)
    write_output(predictions)
