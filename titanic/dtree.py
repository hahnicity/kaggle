from sklearn.tree import DecisionTreeClassifier

from titanic.common import get_best


def dtree_classifier(x_train, y_train, x_cv, y_cv):
    results = []
    for random_state in [1, None]:
        for max_features in ["auto", "sqrt", "log2", None]:
            classifier = DecisionTreeClassifier(
                random_state=random_state, max_features=max_features
            )
            classifier.fit(x_train, y_train)
            run_results = {
                "classifier": classifier,
                "type": "dtree",
                "value": classifier.score(x_cv, y_cv),
            }
            results.append(run_results)
    best = get_best(results)
    return best
