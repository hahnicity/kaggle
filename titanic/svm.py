from sklearn import svm

from titanic.common import get_best


def svc_classifier(x_train, y_train, x_cv, y_cv):
    results = {}
    for c_exp in range(-5, 17, 2):
        for gamma_exp in range(-15, 5, 2):
            classifier = svm.SVC(C=2 ** c_exp, gamma=2 ** gamma_exp)
            classifier.fit(x_train, y_train)
            results["svc C:{} gamma:{}".format(c_exp, gamma_exp)] = classifier.score(x_cv, y_cv)
    get_best(results)


def predictor(x_train, y_train, x_test):
    # represents the best model I have available to me right now
    classifier = svm.SVC(C=2 ** 3, gamma=2 ** 1)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    return predictions
