from sklearn.linear_model import LogisticRegression

from titanic.common import get_best


def logistic_regression_classifier(x_train, y_train, x_cv, y_cv):
    results = []
    for c_exp in range(-12, 2):
        for tol_exp in (-6, 1):
            for random_state in [None, 1]:
                for scaling_exp in (-4, 4):
                    classifier = LogisticRegression(
                        C=2 ** c_exp,
                        tol=10 ** tol_exp,
                        random_state=random_state,
                        intercept_scaling=2 ** scaling_exp,
                        max_iter=1000,
                    )
                    classifier.fit(x_train, y_train)
                    run_results = {
                        "classifier": classifier,
                        "type": "log_reg",
                        "value": classifier.score(x_cv, y_cv),
                    }
            results.append(run_results)
    best = get_best(results)
    return best
