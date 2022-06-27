import numpy as np

from elfi.classifiers.classifier import GPClassifier, LogisticRegression


def make_train_data():
    nsim = 100
    sample_1 = np.random.normal(0.5, 0.1, size=nsim)
    sample_2 = np.random.normal(0.2, 0.1, size=nsim)
    features = np.concatenate((sample_1, sample_2)).reshape(-1, 1)
    labels = np.concatenate((-1 * np.ones(nsim), np.ones(nsim)))
    return features, labels


def test_logistic_regression():
    np.random.seed(270622)
    train_x, train_y = make_train_data()
    cls = LogisticRegression()
    cls.fit(train_x, train_y)
    test_x = np.array([0.2, 0.3, 0.5, 0.7]).reshape(-1, 1)
    test_ratios = [5.01260789, 1.90608623, -4.30695707, -10.52000038]
    pred_ratios = cls.predict_log_likelihood_ratio(test_x)
    assert np.allclose(pred_ratios, test_ratios, atol=0.001)


def test_GP_classifier():
    np.random.seed(270622)
    train_x, train_y = make_train_data()
    cls = GPClassifier()
    cls.fit(train_x, train_y)
    test_x = np.array([0.2, 0.3, 0.5, 0.7]).reshape(-1, 1)
    test_ratios = [3.43399806, 1.4273162, -2.94759437, -3.26829432]
    pred_ratios = cls.predict_log_likelihood_ratio(test_x)
    assert np.allclose(pred_ratios, test_ratios, atol=0.001)
