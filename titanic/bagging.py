from sklearn.ensemble import BaggingClassifier

from titanic.common import get_best


def bagging_classifier(x_train, y_train, x_cv, y_cv):
    results = []
    for estimator in range(5, 50, 5):
        for features in range(x_train.shape[1] - 5, x_train.shape[1] + 1):
            classifier = BaggingClassifier(n_estimators=estimator, max_features=features)
            classifier.fit(x_train, y_train)
            run_results = {
                "classifier": classifier,
                "type": "bagging",
                "value": classifier.score(x_cv, y_cv),
            }
            results.append(run_results)
    best = get_best(results)
    return best
