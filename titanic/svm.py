from sklearn import svm

from titanic.common import get_best


def svc_classifier(x_train, y_train, x_cv, y_cv):
    results = []
    for c_exp in range(-5, 17, 2):
        for gamma_exp in range(-15, 5, 2):
            classifier = svm.SVC(C=2 ** c_exp, gamma=2 ** gamma_exp)
            classifier.fit(x_train, y_train)
            run_results = {
                "args": {
                    "C": c_exp, "gamma": gamma_exp,
                },
                "classifier": classifier,
                "type": "svm",
                "value": classifier.score(x_cv, y_cv),
            }
            results.append(run_results)
    best = get_best(results)
    return best
