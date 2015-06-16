from sklearn.ensemble import AdaBoostClassifier

from titanic.common import get_best


def adaboost_classifier(x_train, y_train, x_cv, y_cv):
    results = []
    for estimator in range(100, 200, 20):
        for irate in range(-5, 5):
            rate = 2.0 ** irate
            classifier = AdaBoostClassifier(n_estimators=estimator, learning_rate=rate)
            classifier.fit(x_train, y_train)
            run_results = {
                "classifier": classifier,
                "type": "ada",
                "value": classifier.score(x_cv, y_cv),
            }
            results.append(run_results)
    best = get_best(results)
    return best
