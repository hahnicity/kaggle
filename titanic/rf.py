from sklearn.ensemble import RandomForestClassifier

from titanic.common import get_best


def random_forest_classifier(x_train, y_train, x_cv, y_cv):
    results = []
    for estimator in range(5, 50):
        for max_features in ["auto", "sqrt", "log2", None]:
            for random_state in [1, None]:
                classifier = RandomForestClassifier(
                    n_estimators=estimator, max_features=max_features, random_state=random_state
                )
                classifier.fit(x_train, y_train)
                run_results = {
                    "classifier": classifier,
                    "type": "rf",
                    "value": classifier.score(x_cv, y_cv),
                }
                results.append(run_results)
    best = get_best(results)
    return best
