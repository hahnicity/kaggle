def get_best(results):
    best = ("", 0.0)
    for key, val in results.iteritems():
        if val > best[1]:
            best = (key, val)
    print best
