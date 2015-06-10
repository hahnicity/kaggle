import csv
from os.path import abspath, dirname, join

import numpy
from pandas import read_csv
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def perform_feature_scaling(data_set):
    for col in data_set.columns:
        scaled = (data_set[col] - data_set[col].min()) / (data_set[col].max() - data_set[col].min())
        data_set[col] = scaled
    return data_set


def get_training_data(data_set):
    data_set = get_data(data_set)
    y = data_set["Survived"]
    del data_set["Survived"]
    return data_set, y


def get_data(data_set):
    data_set["Sex"] = data_set.Sex.map(lambda x: 1 if x == "female" else 0)
    mean_age = data_set.Age.mean()
    mean_fare = data_set.Fare.mean()
    data_set.Age = data_set.Age.map(lambda x: mean_age if numpy.isnan(x) else x)
    data_set.Fare = data_set.Fare.map(lambda x: mean_fare if numpy.isnan(x) else x)
    data_set.Cabin = data_set.Cabin.map(lambda x: 1 if isinstance(x, str) else 0)
    del data_set["Name"]
    del data_set["PassengerId"]
    del data_set["Ticket"]
    del data_set["Embarked"]
    data_set = perform_feature_scaling(data_set)
    return data_set


def write_output(predictions):
    # passenger id is essentially the DataFrame id plus 892
    with open("titanic-results.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["PassengerId", "Survived"])
        for idx, val in enumerate(predictions):  # predictions should be an array
            writer.writerow([idx + 892, val])


def validate_classifiers(x_train, y_train, x_cv, y_cv):
    classifiers = {
        "svc": svm.SVC,
        #"svr": svm.SVR,
        #"nusvc": svm.NuSVC,
        #"linsvc": svm.LinearSVC,
        #"rf": RandomForestClassifier,
        #"dtree": DecisionTreeClassifier,
    }
    results = {key: None for key in classifiers.keys()}
    for type_, Classifier in classifiers.iteritems():
        for c_exp in range(-5, 17, 2):
            for gamma_exp in range(-15, 5, 2):
                classifier = Classifier(C=2 ** c_exp, gamma=2 ** gamma_exp)
                classifier.fit(x_train, y_train)
                results["{} C:{} gamma:{}".format(type_, c_exp, gamma_exp)] = classifier.score(x_cv, y_cv)
    return results


def main():
    working_dir = dirname(abspath(__file__))
    training_path = join(working_dir, "train.csv")
    x, y = get_training_data(read_csv(training_path))
    two_thirds = int(len(x) * (2.0 / 3))
    x_cv = x[two_thirds:]
    y_cv = y[two_thirds:]
    x_train = x[:two_thirds]
    y_train = y[:two_thirds]
    results = validate_classifiers(x_train, y_train, x_cv, y_cv)
    from pprint import pprint
    pprint(results)
#    testing_path = join(working_dir, "test.csv")
#    testing_set = get_data(read_csv(testing_path))
#    predictions = classifier.predict(testing_set)
#    write_output(predictions)
