import numpy as np

from elfi.methods.classifier import GPClassifier, LogisticRegression


def make_train_data(seed=270622):
    random_state = np.random.RandomState(seed=seed)
    nsim = 100
    sample_1 = random_state.normal(0.5, 0.1, size=nsim)
    sample_2 = random_state.normal(0.2, 0.1, size=nsim)
    features = np.concatenate((sample_1, sample_2)).reshape(-1, 1)
    labels = np.concatenate((-1 * np.ones(nsim), np.ones(nsim)))
    return features, labels


def test_logistic_regression():
    train_x, train_y = make_train_data()
    cls = LogisticRegression()
    cls.fit(train_x, train_y)
    test_x = np.array([0.2, 0.3, 0.5, 0.7]).reshape(-1, 1)
    test_ratios = [5.012608, 1.906086, -4.306957, -10.520000]
    pred_ratios = cls.predict_log_likelihood_ratio(test_x)
    assert np.allclose(pred_ratios, test_ratios, atol=0.001)


def test_GP_classifier():
    train_x, train_y = make_train_data()
    cls = GPClassifier()
    cls.fit(train_x, train_y)
    test_x = np.array([0.2, 0.3, 0.5, 0.7]).reshape(-1, 1)
    test_ratios = [3.433998, 1.427316, -2.947594, -3.268294]
    pred_ratios = cls.predict_log_likelihood_ratio(test_x)
    assert np.allclose(pred_ratios, test_ratios, atol=0.001)
