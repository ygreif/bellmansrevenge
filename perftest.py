import worker

if __name__ == '__main__':
    import timeit
    for i in range(1, 8):
        print "On worker", i
        print i, timeit.timeit("worker.runworkers(16, 100, 1, " + str(i) + ", {'k': .2, 'z': 1}, default_params=worker.default_params)", setup="import worker", number=5)
