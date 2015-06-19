from argparse import ArgumentParser
import csv
from os.path import abspath, dirname, join

import numpy
from pandas import read_csv

from titanic import ada, bagging, dtree, log_reg, rf, svm
from titanic.common import predictor


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
    data_set["Sex"] = data_set.Sex.map({"male": 0, "female": 1})
    median_age = data_set.Age.median()
    fare_mapping = {
        1: data_set[data_set.Pclass == 1].Fare.median(),
        2: data_set[data_set.Pclass == 2].Fare.median(),
        3: data_set[data_set.Pclass == 3].Fare.median()
    }
    # So from the data it looks like age is only a good predictor if we ask if the
    # person is 10 or less (from rough binning). So just do that on a boolean
    #
    # In the future instead of replacing everyone with the mean age replace them
    # with a proper age distribution...maybe according to bayes??
    data_set.Age = data_set.Age.map(lambda x: median_age if numpy.isnan(x) else x)
    data_set.Age = data_set.Age.map(lambda x: 1 if x <= 10.0 else 0)
    # In the future instead of replacing everyone with the mean age replace them
    # with a proper age distribution...maybe according to bayes??
    for i in range(len(data_set)):
        if numpy.isnan(data_set.iloc[i].Fare):
            data_set.loc[i, "Fare"] = fare_mapping[data_set.iloc[i].Pclass]
    #data_set.Fare = data_set.Fare.map(lambda x: 1 if x > 50 else 0)
    #
    # The following transformation increases the accuracy of the random forest at the expense of
    # the SVM. Makes sense to me why, we sacrifice some accuracy on fare to more easily
    # bin out fares into a tree structure.
    #fare_bins = [(0, 10), (10, 45), (45, 60), (60, 100), (100, 600)]
    #for i in range(len(data_set)):
    #    for idx, bin_ in enumerate(fare_bins):
    #        low, high = bin_
    #        if data_set.iloc[i].Fare >= low and data_set.iloc[i].Fare < high:
    #            data_set.loc[i, "Fare"] = idx
    data_set["Embarked"] = data_set.Embarked.map({numpy.nan: -1, "S": 0, "Q": 1, "C": 2})
    data_set.Cabin = data_set.Cabin.map(lambda x: 1 if isinstance(x, str) else 0)
    del data_set["Name"]
    del data_set["PassengerId"]
    del data_set["Ticket"]
    #del data_set["Fare"]  # XXX comment if you want
    #del data_set["Parch"]  # XXX comment if you want
    #del data_set["SibSp"]  # XXX comment if you want
    #data_set = perform_feature_scaling(data_set)
    return data_set


def validate_classifiers(x_train, y_train, x_cv, y_cv):
    best = {}
    candidates = []
    #ada.adaboost_classifier(x_train, y_train, x_cv, y_cv)
    #candidates.append(bagging.bagging_classifier(x_train, y_train, x_cv, y_cv))
    #candidates.append(dtree.dtree_classifier(x_train, y_train, x_cv, y_cv))
    #candidates.append(rf.random_forest_classifier(x_train, y_train, x_cv, y_cv))
    candidates.append(svm.svc_classifier(x_train, y_train, x_cv, y_cv))
    candidates.append(log_reg.logistic_regression_classifier(x_train, y_train, x_cv, y_cv))
    for cand in candidates:
        try:
            if cand["value"] > best["value"]:
                best = cand
        except KeyError:
            best = cand
    return best, candidates


def main():
    parser = ArgumentParser()
    parser.add_argument("--predict", action="store_true")
    parser.add_argument(
        "--predict-model",
        choices=("bagging", "svm", "rf", "dtree", "log_reg"),
        default="svm"
    )
    args = parser.parse_args()
    working_dir = dirname(abspath(__file__))
    training_path = join(working_dir, "train.csv")
    x, y = get_training_data(read_csv(training_path))
    cv_frac = int(len(x) * (3.0 / 4))
    x_train = x[:cv_frac]
    y_train = y[:cv_frac]
    x_cv = x[cv_frac:]
    y_cv = y[cv_frac:]
    best, candidates = validate_classifiers(x_train, y_train, x_cv, y_cv)
    import pdb; pdb.set_trace()
    if args.predict:
        testing_path = join(working_dir, "test.csv")
        testing_set = get_data(read_csv(testing_path))
        if args.predict_model:
            candidate = [cand for cand in candidates if cand["type"] == args.predict_model]
            predictions = predictor(candidate[0]["classifier"], testing_set)
        else:
            predictions = predictor(best["classifier"], testing_set)
        write_output(predictions)
