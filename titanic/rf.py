from sklearn.ensemble import RandomForestClassifier

from titanic.common import get_best


def random_forest_classifier(x_train, y_train, x_cv, y_cv):
    results = {}
    for estimator in range(5, 50):
        for max_features in ["auto", "sqrt", "log2", None]:
            classifier = RandomForestClassifier(n_estimators=estimator, max_features=max_features)
            classifier.fit(x_train, y_train)
            results["rf estimators: {}, mf: {}".format(estimator, max_features)] = classifier.score(x_cv, y_cv)
    get_best(results)


def predictor(x_train, y_train, x_test):
    # represents the best model I have available to me right now
    classifier = RandomForestClassifier(n_estimators=11, max_features=None)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    return predictions
