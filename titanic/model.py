from argparse import ArgumentParser
import csv
from os.path import abspath, dirname, join

import numpy
from pandas import read_csv

from titanic import svm
from titanic import rf


def perform_feature_scaling(data_set):
    for col in data_set.columns:
        scaled = (data_set[col] - data_set[col].min()) / (data_set[col].max() - data_set[col].min())
        data_set[col] = scaled
    return data_set


def write_output(predictions):
    # passenger id is essentially the DataFrame id plus 892
    with open("titanic-results.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["PassengerId", "Survived"])
        for idx, val in enumerate(predictions):  # predictions should be an array
            writer.writerow([idx + 892, int(val)])


def get_training_data(data_set):
    data_set = get_data(data_set)
    y = data_set["Survived"]
    del data_set["Survived"]
    return data_set, y


def get_data(data_set):
    data_set["Sex"] = data_set.Sex.map(lambda x: 1 if x == "female" else 0)
    mean_age = data_set.Age.mean()
    mean_fare = data_set.Fare.mean()
    # So from the data it looks like age is only a good predictor if we ask if the
    # person is 10 or less (from rough binning). So just do that on a boolean
    data_set.Age = data_set.Age.map(lambda x: mean_age if numpy.isnan(x) else x)
    data_set.Age = data_set.Age.map(lambda x: 1 if x <= 10.0 else 0)
    data_set.Fare = data_set.Fare.map(lambda x: mean_fare if numpy.isnan(x) else x)
    data_set.Cabin = data_set.Cabin.map(lambda x: 1 if isinstance(x, str) else 0)
    del data_set["Name"]
    del data_set["PassengerId"]
    del data_set["Ticket"]
    del data_set["Embarked"]
    #del data_set["Fare"]  # XXX comment if you want
    #del data_set["Parch"]  # XXX comment if you want
    #del data_set["SibSp"]  # XXX comment if you want
    data_set = perform_feature_scaling(data_set)
    return data_set


def validate_classifiers(x_train, y_train, x_cv, y_cv):
    svm.svc_classifier(x_train, y_train, x_cv, y_cv)
    rf.random_forest_classifier(x_train, y_train, x_cv, y_cv)


def main():
    parser = ArgumentParser()
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--predict-model", choices=("svm", "rf"), default="svm")
    args = parser.parse_args()
    working_dir = dirname(abspath(__file__))
    training_path = join(working_dir, "train.csv")
    x, y = get_training_data(read_csv(training_path))
    two_thirds = int(len(x) * (2.0 / 3))
    x_train = x[:two_thirds]
    y_train = y[:two_thirds]
    x_cv = x[two_thirds:]
    y_cv = y[two_thirds:]
    if not args.predict:
        validate_classifiers(x_train, y_train, x_cv, y_cv)
    else:
        testing_path = join(working_dir, "test.csv")
        testing_set = get_data(read_csv(testing_path))
        predictor = {"svm": svm.predictor, "rf": rf.predictor}[args.predict_model]
        predictions = predictor(x_train, y_train, testing_set)
        write_output(predictions)
