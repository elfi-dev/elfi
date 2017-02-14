import numpy as np
from elfi.bo.gpy_model import GPyModel

class Test_GPyModel():

    def test_default_init(self):
        gp = GPyModel(noise_var=0.)
        assert gp.n_observations == 0
        assert gp.evaluate(np.random.uniform(0.0, 1.0, (1,))) == (0.0, 0.0, 0.0)

    def test_one_1d_sample(self):
        bounds = ((0, 1), )
        X = np.atleast_2d([0.5])
        Y = np.atleast_2d([1.0])
        gp = GPyModel(bounds=bounds, noise_var=0.)
        gp.update(X, Y)
        assert gp.n_observations == 1
        # at observation:
        pred = gp.evaluate(np.array([0.5]))
        target = (1.0, 0.0, 0.0)
        np.testing.assert_allclose(pred, target, atol=1e-3)
        # symmetric estimate:
        d = np.random.uniform(0.01, 0.5)
        pred1 = gp.evaluate(np.array([0.0+d]))
        pred2 = gp.evaluate(np.array([1.0-d]))
        np.testing.assert_allclose(pred1, pred2, atol=1e-3)

    def test_four_2d_samples(self):
        bounds = ((0, 1), (1, 2))
        d = np.random.uniform(0.01, 0.5)
        X = np.atleast_2d([[0.5-d, 1.5-d],
                           [0.5-d, 1.5+d],
                           [0.5+d, 1.5-d],
                           [0.5+d, 1.5+d]])
        v = np.random.uniform(-1.0, 1.0)
        Y = np.atleast_2d([[v], [-v], [v], [-v]])
        gp = GPyModel(input_dim=2, bounds=bounds, noise_var=0.)
        gp.update(X[0:2], Y[0:2])
        assert gp.n_observations == 2
        gp.update(X[2:4], Y[2:4])
        assert gp.n_observations == 4
        # at observations:
        i = np.random.randint(4)
        pred = gp.evaluate(X[i])
        target = (Y[i][0], 0.0, 0.0)
        np.testing.assert_allclose(pred, target, atol=1e-3)
        # circular-symmetric estimate:
        d = np.random.uniform(0.01, 0.5)
        phi = np.random.uniform(0.0, 2.0*np.pi)
        pred1 = gp.evaluate(np.array([0.5+d*np.sin(phi), 1.5+d*np.cos(phi)]))
        pred2 = gp.evaluate(np.array([0.5-d*np.sin(phi), 1.5-d*np.cos(phi)]))
        assert abs(pred1[0] + pred2[0]) < 1e-3
        np.testing.assert_allclose(pred1[1:2], pred2[1:2], atol=1e-3)

    def test_copy_of_gp_gives_same_results(self):
        n = 10
        X = np.random.uniform(size=(n, 2))
        Y = np.random.uniform(size=(n, 1))
        gp1 = GPyModel(input_dim=2, bounds=((0,1),(0,1)))
        gp1.update(X, Y)
        gp2 = gp1.copy()
        loc = np.random.uniform(size=(n, 2))
        for i in range(n):
            m, s, s2 = gp1.evaluate(loc[i][None,:])
            m_, s_, s2_ = gp2.evaluate(loc[i][None,:])
            assert np.abs(m - m_) < 1e-5
            assert np.abs(s - s_) < 1e-5
            assert np.abs(s2 - s2_) < 1e-5

    def test_update_order_is_irrelevant_for_end_result(self):
        n = 10
        X = np.random.uniform(size=(n, 2))
        Y = np.random.uniform(size=(n, 1))
        order1 = np.random.permutation(n)
        order2 = np.random.permutation(n)
        gp1 = GPyModel(input_dim=2, bounds=((0,1),(0,1)))
        gp2 = GPyModel(input_dim=2, bounds=((0,1),(0,1)))
        for i in order1:
            gp1.update(X[i][None,:], Y[i][None,:])
        for i in order2:
            gp2.update(X[i][None,:], Y[i][None,:])
        loc = np.random.uniform(size=(n, 2))
        for i in range(n):
            m, s, s2 = gp1.evaluate(loc[i][None,:])
            m_, s_, s2_ = gp2.evaluate(loc[i][None,:])
            assert np.abs(m - m_) < 1e-5
            assert np.abs(s - s_) < 1e-5
            assert np.abs(s2 - s2_) < 1e-5


    # FIXME
    # def test_change_kernel(self):
    #     bounds = ((0, 1), )
    #     X = np.atleast_2d([0.5])
    #     Y = np.atleast_2d([1.0])
    #     le = np.random.uniform(0.01, 1.0)
    #     va = np.random.uniform(0.01, 1.0)
    #     gp = GPyModel(bounds=bounds, kernel_scale=le, kernel_var=va, noise_var=0.)
    #     gp.update(X, Y)
    #     d = np.random.uniform(-0.5, 0.5)
    #     x = np.array([0.5 + d])
    #     pred1 = gp.evaluate(x)
    #     # change variance
    #     c = np.random.uniform(0.01, 2.0)
    #     gp.set_kernel(kernel_var=va*c)
    #     pred2 = gp.evaluate(x)
    #     assert abs(pred1[0] - pred2[0]) < 1e-3
    #     assert abs(pred1[1] * c - pred2[1]) < 1e-3
    #     assert abs(pred1[2] * np.sqrt(c) - pred2[2]) < 1e-3
    #     # change lengthscale
    #     gp.set_kernel(kernel_var=va, kernel_scale=le*c)
    #     x2 = np.array([0.5 + d*c])
    #     pred3 = gp.evaluate(x2)
    #     np.testing.assert_allclose(pred1, pred3, atol=1e-3)
