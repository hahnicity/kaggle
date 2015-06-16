def get_best(results):
    best = {}
    for item in results:
        try:
            if item["value"] > best["value"]:
                best = item
        except KeyError:
            best = item

    print best
    return best


def predictor(classifier, x_test):
    # represents the best model I have available to me right now
    predictions = classifier.predict(x_test)
    return predictions
