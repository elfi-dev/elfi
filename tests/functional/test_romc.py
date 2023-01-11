import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats as ss

import elfi
from elfi.examples import ma2
from elfi.methods.inference.romc import RegionConstructor, RomcOptimisationResult, OptimisationProblem, NDimBoundingBox
from elfi.methods.posteriors import RomcPosterior


def test_ndim_bounding_box1():
    """Test 2-dimensional bounding box around (5,-5) rotated by 45 degrees
    """
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array(((c, -s), (s, c)))
    center = np.array([5, -5.])
    limits = np.array([[-1, 1], [-2, 2]])
    bb = NDimBoundingBox(rotation, center, limits)
    assert np.allclose(bb.volume, 8.)
    assert np.allclose(bb.pdf(np.array([5, -5.])), 1 / 8)
    assert np.allclose(bb.pdf(np.array([3, -3.])), 0.)


def test_ndim_bounding_box2():
    """Test 4-dimensional bounding box rotated so that the 1-st dimension becomes the 4-th,
    the 2-nd becomes the 3-rh etc.
    """
    rotation = np.array([[0.,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
    center = np.array([0.,0,0,0])
    limits = np.array([[-1., 1.],
                       [-2., 2.],
                       [-3., 3.],
                       [-4., 4.]])
    bb = NDimBoundingBox(rotation, center, limits)
    assert np.equal(bb.volume, 2*4*6*8)
    assert np.allclose(bb.pdf(np.array([4, 3, 2, 1])), 1/bb.volume)
    assert np.allclose(bb.pdf(np.array([4.1, 3, 2, 1])), 0)


def test_ndim_bounding_box3():
    """Test 2-dimensional bounding box with very zero limits
    """
    rotation = np.eye(2)
    center = np.array([0, 0])
    limits = np.array([[0, 0], [0, 0]])
    bb = NDimBoundingBox(rotation, center, limits)
    assert bb.volume > 0


def test_region_constructor1():
    """Test for squeezed Gaussian."""
    # Create Gaussian with rotation
    mean = np.array([0., 0.])
    hess = np.array([[1.0, .7], [.7, 1.]])

    def f(x):
        rv = ss.multivariate_normal(mean, hess)
        return - rv.pdf(x)

    opt_res = RomcOptimisationResult(x_min=mean, f_min=f(mean), hess_appr=hess)
    lim = 20
    step = .1
    K = 10
    eta = 1
    eps_region = -.025
    constr = RegionConstructor(opt_res, f, dim=2,
                               eps_region=eps_region,
                               K=K, eta=eta)

    prep_region = constr.build()
    limits_pred = prep_region[0].limits
    limits_gt = np.array([[-2.7265, 2.7265], [-1.1445, 1.1445]])

    plt.figure()
    x, y = np.mgrid[-4:4:.01, -4:4:.01]
    pos = np.dstack((x, y))
    plt.contourf(x, y, f(pos))
    plt.colorbar()
    x = prep_region[0].sample(300)
    plt.plot(x[:,0], x[:,1], 'ro')
    plt.show(block=False)

    assert np.allclose(limits_pred, limits_gt, atol=1e-2)


def test_region_constructor2():
    """Test for square proposal region"""

    # boundaries
    x1_neg = -.1
    x1_pos = 1
    x2_neg = -.2
    x2_pos = 2

    center = np.array([0, 0])

    hess = np.eye(2)
    def f(x):
        if (x1_neg <= x[0] <= x1_pos) and (x2_neg <= x[1] <= x2_pos):
            y = -1
        else:
            y = 1
        return y

    # compute proposal region
    opt_res = RomcOptimisationResult(x_min=center,
                                     f_min=f(center),
                                     hess_appr=hess)
    lim = 20
    step = .1
    K = 10
    eta = 1
    eps_region = 0.
    constr = RegionConstructor(opt_res, f, dim=2,
                               eps_region=eps_region,
                               K=K, eta=eta)
    proposal_region = constr.build()[0]
    # compare limits
    limits_pred = proposal_region.limits
    limits_gt = np.array([[x1_neg, x1_pos], [x2_neg, x2_pos]])
    assert np.allclose(limits_pred, limits_gt, atol=eta/2**(K-1))

    # compare volume
    assert np.allclose((x1_pos - x1_neg)*(x2_pos - x2_neg),
                       proposal_region.volume, atol=.1)


def test_region_constructor3():
    """Test for very tight edge case"""

    # boundaries
    x1_neg = -.0001
    x1_pos = .0001
    x2_neg = -.0001
    x2_pos = .0001

    center = np.array([0, 0])

    hess = np.eye(2)
    def f(x):
        if (x1_neg <= x[0] <= x1_pos) and (x2_neg <= x[1] <= x2_pos):
            y = -1
        else:
            y = 1
        return y

    # compute proposal region
    opt_res = RomcOptimisationResult(x_min=center,
                                     f_min=f(center),
                                     hess_appr=hess)
    lim = 20
    step = .1
    K = 5
    eta = 1
    eps_region = 0.
    constr = RegionConstructor(opt_res, f, dim=2,
                               eps_region=eps_region,
                               K=K, eta=eta)
    proposal_region = constr.build()[0]

    # compare limits
    limits_pred = proposal_region.limits
    limits_gt = np.array([[x1_neg, x1_pos], [x2_neg, x2_pos]])
    assert np.allclose(limits_pred, limits_gt, atol=eta/2**(K-1))

    # compare volume
    assert np.allclose((x1_pos - x1_neg)*(x2_pos - x2_neg),
                       proposal_region.volume, atol=.1)


def test_region_constructor4():
    """Test when boundary is limitless"""

    # boundaries
    rep_lim = 300
    eta = 1
    x1_neg = -rep_lim*eta
    x1_pos = rep_lim*eta
    x2_neg = -rep_lim*eta
    x2_pos = rep_lim*eta

    center = np.array([0, 0])

    hess = np.eye(2)
    def f(x):
        return 0

    # compute proposal region
    opt_res = RomcOptimisationResult(x_min=center,
                                     f_min=f(center),
                                     hess_appr=hess)
    lim = 20
    K = 5
    eps_region = 0.1
    constr = RegionConstructor(opt_res, f, dim=2, eps_region=eps_region,
                               K=K, eta=eta, rep_lim=rep_lim)
    proposal_region = constr.build()[0]

    # compare limits
    limits_pred = proposal_region.limits
    limits_gt = np.array([[x1_neg, x1_pos], [x2_neg, x2_pos]])
    assert np.allclose(limits_pred, limits_gt, atol=.1)

    # compare volume
    assert np.allclose((x1_pos - x1_neg)*(x2_pos - x2_neg),
                       proposal_region.volume, atol=.1)


def test_optimisation_problem1():
    ind = 1
    nuisance = 1
    parameter_names = ["x1", "x2"]
    target_name = "y"
    dim = 2
    n1 = 20
    bounds = [(-10, 10), (-10, 10)]

    def f(x):
        y = np.array([x[0], x[1]**2])
        return y

    def objective(x):
        y1 = f(x)
        return np.sqrt((y1[0] - 1)**2 + (y1[1] - 4)**2)

    # Create Gaussian with rotation
    mean = np.array([0., 0.])
    hess = np.array([[1.0, .7], [.7, 1.]])
    prior = ss.multivariate_normal(mean, hess)

    opt_prob = OptimisationProblem(ind, nuisance, parameter_names, target_name,
                                   objective, dim, prior, n1, bounds)

    x0 = np.array([-10, -10])
    solved = opt_prob.solve_gradients(x0=x0)

    assert solved
    assert (np.allclose(opt_prob.result.x_min, np.array([1, 2]), atol=.1) or np.allclose(opt_prob.result.x_min, np.array([1, -2]), atol=.1))

    opt_prob.build_region(eps_region=0.2)
    opt_prob.visualize_region()


def test_optimisation_problem2():
    ind = 1
    nuisance = 1
    parameter_names = ["x1", "x2"]
    target_name = "y"
    dim = 2
    n1 = 20
    bounds = [(-10, 10), (-10, 10)]

    def f(x):
        y = np.array([x[0], x[1]])
        return y

    def objective(x):
        y1 = f(x)
        return np.sqrt((y1[0] - 1)**2 + (y1[1] - 4)**2)

    # Create Gaussian with rotation
    mean = np.array([0., 0.])
    hess = np.array([[1.0, .7], [.7, 1.]])
    prior = ss.multivariate_normal(mean, hess)

    opt_prob = OptimisationProblem(ind, nuisance, parameter_names, target_name,
                                   objective, dim, prior, n1, bounds)

    x0 = np.array([-10, -10])
    solved = opt_prob.solve_gradients(x0=x0)

    assert solved
    assert np.allclose(opt_prob.result.x_min, np.array([1, 4]), atol=.1)

    opt_prob.build_region(eps_region=0.2)
    opt_prob.visualize_region()


def test_optimisation_problem3():
    ind = 1
    nuisance = 1
    parameter_names = ["x1", "x2"]
    target_name = "y"
    dim = 2
    n1 = 20
    bounds = [(-10, 10), (-10, 10)]

    def f(x):
        y = np.array([x[0], x[1]])
        return y

    # Create Gaussian with rotation
    mean = np.array([0., 0.])
    hess = np.array([[1.0, .7], [.7, 1.]])
    prior = ss.multivariate_normal(mean, hess)

    def objective(x):
        return - prior.pdf(x)

    opt_prob = OptimisationProblem(ind, nuisance, parameter_names, target_name,
                                   objective, dim, prior, n1, bounds)

    x0 = np.array([-10, -10])
    solved = opt_prob.solve_gradients(x0=x0)
    assert solved
    assert np.allclose(opt_prob.result.x_min, np.array([0., 0.]), atol=.1)
    opt_prob.build_region(eps_region=-0.1)
    opt_prob.visualize_region()

    solved = opt_prob.solve_bo(x0=x0)
    assert solved
    assert np.allclose(opt_prob.result.x_min, np.array([0., 0.]), atol=1.)
    opt_prob.build_region(eps_region=-0.1, use_surrogate=False)
    opt_prob.visualize_region(force_objective=False)
    opt_prob.visualize_region(force_objective=True)


def test_optimisation_problem4():
    ind = 1
    nuisance = 1
    parameter_names = ["x1", "x2"]
    target_name = "y"
    dim = 2
    n1 = 20
    bounds = [(-10, 10), (-10, 10)]

    def f(x):
        y = np.array([x[0], x[1]])
        return y

    # Create Gaussian with rotation
    mean = np.array([0., 0.])
    hess = np.array([[1.0, .7], [.7, 1.]])
    prior = ss.multivariate_normal(mean, hess)

    def objective(x):
        return - prior.pdf(x)

    opt_prob = OptimisationProblem(ind, nuisance, parameter_names, target_name,
                                   objective, dim, prior, n1, bounds)

    solved = opt_prob.solve_bo()

    assert solved
    assert np.allclose(opt_prob.result.x_min, np.array([0., 0.]), atol=1)

    opt_prob.build_region(eps_region=-0.1)

    opt_prob.visualize_region(force_objective=False)
    opt_prob.visualize_region(force_objective=True)


def test_romc_posterior1():
    """Test for ROMC posterior."""
    # f(x) is -1 inside the box 2x4, and 1 outside
    x1_neg = -1
    x1_pos = 1.
    x2_neg = -2
    x2_pos = 2.
    center = np.array([0, 0])
    hess = np.eye(2)

    # define the deterministic simulator d
    def f(x):
        if (x1_neg <= x[0] <= x1_pos) and (x2_neg <= x[1] <= x2_pos):
            y = -1
        else:
            y = 1
        return y

    # define the prior class
    class Prior:
        def __init__(self, ):
            self.dim = 2
            return

        @staticmethod
        def pdf(x):
            if (-1 <= x[0, 0] <= 1) and (-1 <= x[0, 1] <= 1):
                return np.array([1.])
            else:
                return np.array([0.])

    # obtain optimisation result
    opt_res = RomcOptimisationResult(x_min=center,
                                     f_min=f(center),
                                     hess_appr=hess)

    # construct bounding box region
    K = 10
    eta = 1
    eps_region = 0.
    constr = RegionConstructor(opt_res, f, dim=2,
                               eps_region=eps_region,
                               K=K, eta=eta)
    proposal_region = constr.build()[0]

    post = RomcPosterior(proposal_region, [f], [f], [f], [f], [0],
                         False,
                         Prior(),
                         np.array([-1., -1.]),
                         np.array([1., 1.]),
                         eps_filter=eps_region,
                         eps_region=eps_region,
                         eps_cutoff=eps_region)

    assert np.array_equal(np.array([1.]), post.pdf_unnorm_batched(np.array([[.1, .2]])))
    assert np.array_equal(np.array([0.25]), post.pdf(np.array([[.1, .2]])))


@pytest.mark.slowtest
def test_romc1():
    """Test ROMC at the simple 1D example."""
    # the prior distribution
    class Prior:
        def rvs(self, size=None, random_state=None):
            # size from (BS,) -> (BS,1)
            if size is not None:
                size = np.concatenate((size, [1]))
            return ss.uniform(loc=-2.5, scale=5).rvs(size=size, random_state=random_state)

        def pdf(self, theta):
            return ss.uniform(loc=-2.5, scale=5).pdf(theta)

        def logpdf(self, theta):
            return ss.uniform(loc=-2.5, scale=5).logpdf(theta)

    # function for sampling from the likelihood
    def likelihood_sample(theta, seed=None):
        """Vectorized sampling from likelihood."""
        assert isinstance(theta, np.ndarray)
        theta = theta.astype(float)
        samples = np.empty_like(theta)

        c = 0.5 - 0.5 ** 4

        tmp_theta = theta[theta <= -0.5]
        samples[theta <= -0.5] = ss.norm(loc=-tmp_theta - c, scale=1).rvs(random_state=seed)
        theta[theta <= -0.5] = np.inf

        tmp_theta = theta[theta <= 0.5]
        samples[theta <= 0.5] = ss.norm(loc=tmp_theta ** 4, scale=1).rvs(random_state=seed)
        theta[theta <= 0.5] = np.inf

        tmp_theta = theta[theta < np.inf]
        samples[theta < np.inf] = ss.norm(loc=tmp_theta - c, scale=1).rvs(random_state=seed)
        theta[theta < np.inf] = np.inf

        assert np.allclose(theta, np.inf)
        return samples

    # define the simulator
    def simulator(theta, dim, batch_size=10000, random_state=None):
        theta = np.repeat(theta, dim, -1)
        return likelihood_sample(theta, seed=random_state)

    data = np.array([0.])
    dim = data.shape[0]

    # Define ELFI model
    elfi.new_model("1D_example")
    elfi_prior = elfi.Prior(Prior(), name="theta")
    elfi_simulator = elfi.Simulator(simulator, elfi_prior, dim, observed=np.expand_dims(data, 0),
                                    name="simulator")
    dist = elfi.Distance('euclidean', elfi_simulator, name="dist")

    # Define ROMC inference method
    bounds = [(-2.5, 2.5)]
    romc = elfi.ROMC(dist, bounds)

    # Gradients-Based solution
    n1 = 100
    seed = 21
    optimizer_args = {}
    use_bo = False
    romc.solve_problems(n1=n1, seed=seed, use_bo=use_bo, optimizer_args=optimizer_args)

    # Estimate Regions
    eps_filter = .75
    fit_models = True
    fit_models_args = {"nof_points": 30}
    romc.estimate_regions(eps_filter=eps_filter, fit_models=fit_models,
                          fit_models_args=fit_models_args)

    # Sample from the approximate posterior
    n2 = 30
    romc.sample(n2=n2)

    # assert summary statistics of samples match the ground truth
    assert np.allclose(romc.compute_expectation(h=lambda x: np.squeeze(x)), 0, atol=.4)
    assert np.allclose(romc.compute_expectation(h=lambda x: np.squeeze(x) ** 2), 1.1, atol=.4)


@pytest.mark.slowtest
def test_romc2():
    """Test ROMC at the simple 1D example."""
    # the prior distribution
    class Prior:
        def rvs(self, size=None, random_state=None):
            # size from (BS,) -> (BS,1)
            if size is not None:
                size = np.concatenate((size, [1]))
            return ss.uniform(loc=-2.5, scale=5).rvs(size=size, random_state=random_state)

        def pdf(self, theta):
            return ss.uniform(loc=-2.5, scale=5).pdf(theta)

        def logpdf(self, theta):
            return ss.uniform(loc=-2.5, scale=5).logpdf(theta)

    # function for sampling from the likelihood
    def likelihood_sample(theta, seed=None):
        """Vectorized sampling from likelihood."""
        assert isinstance(theta, np.ndarray)
        theta = theta.astype(float)
        samples = np.empty_like(theta)

        c = 0.5 - 0.5 ** 4

        tmp_theta = theta[theta <= -0.5]
        samples[theta <= -0.5] = ss.norm(loc=-tmp_theta - c, scale=1).rvs(random_state=seed)
        theta[theta <= -0.5] = np.inf

        tmp_theta = theta[theta <= 0.5]
        samples[theta <= 0.5] = ss.norm(loc=tmp_theta ** 4, scale=1).rvs(random_state=seed)
        theta[theta <= 0.5] = np.inf

        tmp_theta = theta[theta < np.inf]
        samples[theta < np.inf] = ss.norm(loc=tmp_theta - c, scale=1).rvs(random_state=seed)
        theta[theta < np.inf] = np.inf

        assert np.allclose(theta, np.inf)
        return samples

    # define the simulator
    def simulator(theta, dim, batch_size=10000, random_state=None):
        theta = np.repeat(theta, dim, -1)
        return likelihood_sample(theta, seed=random_state)

    data = np.array([0.])
    dim = data.shape[0]

    # Define ELFI model
    elfi.new_model("1D_example")
    elfi_prior = elfi.Prior(Prior(), name="theta")
    elfi_simulator = elfi.Simulator(simulator, elfi_prior, dim, observed=np.expand_dims(data, 0),
                                    name="simulator")
    dist = elfi.Distance('euclidean', elfi_simulator, name="dist")

    # Define ROMC inference method
    bounds = [(-2.5, 2.5)]
    romc = elfi.ROMC(dist, bounds)

    # Bayesian Optimisation solution part
    n1 = 50
    seed = 21
    optimizer_args = {}
    use_bo = True
    romc.solve_problems(n1=n1, seed=seed, use_bo=use_bo, optimizer_args=optimizer_args)

    eps_filter = .75
    fit_models = True
    use_surrogate = True
    fit_models_args = {"nof_points": 30}
    romc.estimate_regions(eps_filter=eps_filter, use_surrogate=use_surrogate,
                          fit_models=fit_models, fit_models_args=fit_models_args)

    n2 = 100
    romc.sample(n2=n2)

    # assert summary statistics of samples match the ground truth
    assert np.allclose(romc.compute_expectation(h=lambda x: np.squeeze(x)), 0, atol=.4)
    assert np.allclose(romc.compute_expectation(h=lambda x: np.squeeze(x) ** 2), 1.1, atol=.4)
