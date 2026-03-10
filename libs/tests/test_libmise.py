# import time
import numpy as np

from .. import MISE


def test_mise():
    # t0 = time.perf_counter()
    extractor = MISE(1, 2, 0.0)
    print(extractor.resolution_0, extractor.resolution)

    p = extractor.query()
    print(extractor.resolution)
    i = 0

    while p.shape[0] != 0:
        # PointCloud(p).show()
        v = 2 * (p.sum(axis=-1) > 2).astype(np.float64) - 1
        extractor.update(p, v)
        p = extractor.query()
        i += 1
        if i >= 8:
            break

    # print(extractor.to_dense())
    # p, v = extractor.get_points()
    # print(p)
    # print(v)
    # print('Total time: %f' % (time.perf_counter() - t0))
