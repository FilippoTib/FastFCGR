from fastfcgr import FastFCGR
import numpy as np

# def test_basic_fcgr():
#     fcgr = FastFCGR()
#     fcgr.initialize(2)
#     fcgr.set_sequence("AAAAAAAACCTTG")
#     expected = np.array([[7, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0]]).astype(np.uint)
#     actual = fcgr.get_matrix()
#     res = expected.all() == actual.all()
#     assert res
