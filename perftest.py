import worker

if __name__ == '__main__':
    import timeit
    print timeit.timeit("worker.runworkers(1, 100, 1, 1, {'k': .2, 'z': 1}, default_params=worker.default_params)", setup="import worker", number=10)
