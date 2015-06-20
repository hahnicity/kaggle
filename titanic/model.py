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
    fare_mapping = {
        1: data_set[data_set.Pclass == 1].Fare.mean(),
        2: data_set[data_set.Pclass == 2].Fare.mean(),
        3: data_set[data_set.Pclass == 3].Fare.mean()
    }
    titles = data_set.Name.map(lambda x: x.split(",")[-1].split(".")[0].strip())
    mean_ages_by_title = {title: data_set[titles == title].Age.mean() for title in titles.unique()}
    # Ensure no nan's sneak into the mapping
    for k, v in mean_ages_by_title.items():
        if numpy.isnan(v):
            # This is a rare case, so just set it to mean age
            mean_ages_by_title[k] = data_set.Age.mean()
    for i in data_set[data_set.Age.isnull()].index:
        title = titles.iloc[i]
        data_set.loc[i, "Age"] = mean_ages_by_title[title]
    for i in data_set[(data_set.Fare.isnull()) | (data_set.Fare == 0.0)].index:
        data_set.loc[i, "Fare"] = fare_mapping[data_set.iloc[i].Pclass]
    data_set["LifeboatPriority"] = (
        ((data_set.Pclass == 1) | (data_set.Pclass == 2)) &
        ((data_set.Sex == 1) | (data_set.Age <= 15))
    ).map({False: 0, True: 1})
    data_set["Embarked"] = data_set.Embarked.map({numpy.nan: -1, "S": 0, "Q": 1, "C": 2})
    data_set.Cabin = data_set.Cabin.map(lambda x: 1 if isinstance(x, str) else 0)
    del data_set["Name"]
    del data_set["PassengerId"]
    del data_set["Ticket"]
    data_set = perform_feature_scaling(data_set)
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
    if args.predict:
        testing_path = join(working_dir, "test.csv")
        testing_set = get_data(read_csv(testing_path))
        if args.predict_model:
            candidate = [cand for cand in candidates if cand["type"] == args.predict_model]
            predictions = predictor(candidate[0]["classifier"], testing_set)
        else:
            predictions = predictor(best["classifier"], testing_set)
        write_output(predictions)
