import elfi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from elfi import BOLFI
from matplotlib.animation import FuncAnimation, FFMpegWriter

seed = 0

# Number of minimum samples.
n_low = 1
# Number of maximum samples.
n = 5
np.random.seed(seed)
gauss_samples = ss.norm().rvs(size=n)
"""
t_plus_range = [0.36, 0.44]
t_plus = np.linspace(*t_plus_range, 200)
model = lambda t_plus: t_plus**2
x_range = [0.36, 0.44]
y_range = [-0.8, 1.5]
canvas = np.meshgrid(np.linspace(*x_range, 200),
                     np.linspace(*reversed(y_range), 200))
t_plus_samples = t_plus_range.copy()

fit_spline = []
fit_spline_uncertainty = []
fit_acq = []
fit_discrepancy = []
fit_acq_t_plus = []

prior = elfi.Prior('norm', 0.4, 0.02**2, name='t_+')
elfi_simulator = elfi.Simulator(
    lambda *args, **_: model(args[0]) + ss.norm().rvs(size=1)[0],
    prior, observed=model(0.415)
)
processed_data = elfi.Summary(lambda data: data, elfi_simulator)
distance = elfi.Distance('euclidean', processed_data)
log_distance = elfi.Operation(np.log, distance)
bolfi = elfi.BOLFI(
    log_distance, initial_evidence=10, bounds={'t_+': t_plus_range}, seed=seed
)
for i in range(20):
    bolfi.fit(n_evidence=10 + i + 1, bar=True)
    #plt.plot(t_plus, bolfi.target_model.predict(t_plus, noiseless=True)[0])
    #plt.show()
"""

t_plus_range = [0.36, 0.44]
t_plus = np.linspace(*t_plus_range, 200)


def model(t_plus):
    return t_plus**2


x_range = [0.358, 0.442]
t_plus_eval = np.linspace(*x_range, 200)
y_range = [-0.012, 0.052]
canvas = np.meshgrid(np.linspace(*x_range, 200),
                     np.linspace(*reversed(y_range), 200))
t_plus_samples = t_plus_range.copy()

fit_spline = []
fit_spline_uncertainty = []
fit_acq = []
fit_discrepancy = []
fit_acq_t_plus = []

prior = elfi.Prior('norm', 0.4, 0.02, name='t_+')
elfi_simulator = elfi.Simulator(
    lambda *args, **_: model(args[0]) + 0.001 * ss.norm().rvs(size=1)[0],
    prior, observed=model(0.415)
)
processed_data = elfi.Summary(lambda data: data, elfi_simulator)
distance = elfi.Distance('euclidean', processed_data)
# log_distance = elfi.Operation(np.log, distance)


class Animate_GP:
    def __init__(self, fig, ax, initial_evidence=0, frames=None):
        self.bolfi = elfi.BOLFI(
            distance,
            initial_evidence=initial_evidence,
            bounds={'t_+': t_plus_range},
            seed=seed,
            batch_size=10,
        )
        self.ax = ax
        self.ax.set_xlim(*x_range)
        self.ax.set_ylim(*y_range)
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])
        self.ax.set_xlabel("model parameter")
        self.ax.set_ylabel("discrepancy")
        self.initial_evidence = initial_evidence
        self.line, = self.ax.plot([], [])
        self.acq, = self.ax.plot([], [])
        self.img = self.ax.imshow(
            0 * canvas[0],
            cmap='gist_yarg',
            vmin=0,
            vmax=1,
            aspect='auto',
            extent=[*x_range, *y_range]
        )
        self.pts, = self.ax.plot(
            [], [], lw=0, marker='o',
            color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        )
        trans = matplotlib.transforms.ScaledTranslation(
            240/72, -10/72, fig.dpi_scale_trans
        )
        self.txt = self.ax.text(
            0.0, 1.0, str(self.initial_evidence) + " samples    ",
            transform=ax.transAxes + trans, verticalalignment='top'
        )
        self.frames = frames

    def __call__(self, i):
        if self.frames is not None:
            n_evidence = self.initial_evidence + self.frames[i]
        else:
            n_evidence = self.initial_evidence + i
        self.bolfi.fit(n_evidence=n_evidence)
        mean, var = self.bolfi.target_model.predict(
            t_plus_eval, noiseless=True
        )
        self.line.set_data(t_plus_eval, mean)
        eta_squared = 2 * np.log(
            n_evidence**4 * np.pi**2 / 0.3
        )
        self.acq.set_data(t_plus_eval, mean - np.sqrt(eta_squared * var))
        unc = (
            np.exp(-(canvas[1] - mean.T)**2 / (2 * var.T))
            / np.sqrt(2 * np.pi * var.T)
        )
        self.ax.imshow(
            unc,
            cmap='gist_yarg',
            vmin=0,
            vmax=np.max(unc),
            aspect='auto',
            extent=[*x_range, *y_range]
        )
        self.pts.set_data(
            self.bolfi.target_model._gp.X, self.bolfi.target_model._gp.Y
        )
        self.txt.set_text(str(n_evidence) + " samples")
        return self.line, self.acq, self.img, self.pts, self.txt


initial_evidence = 3
frames = np.array([3, 4, 13, 14]) - initial_evidence

fig, ax = plt.subplots(figsize=(6, 4))
anim_gp = Animate_GP(fig, ax, initial_evidence=initial_evidence, frames=frames)
anim = FuncAnimation(
    fig, anim_gp, frames=4, interval=200, blit=False, repeat=False
)

bolfi = elfi.BOLFI(
    distance,
    initial_evidence=initial_evidence,
    bounds={'t_+': t_plus_range},
    seed=seed,
    batch_size=10,
)

likelihood_range = [0.405, 0.425]
likelihood_eval = np.linspace(*likelihood_range, 200)
gps_at_each_frame = []

fig_static, axes = plt.subplots(figsize=(6, 4), nrows=2, ncols=2)

for i, ax in enumerate(axes.flatten()):
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    if frames is not None:
        n_evidence = initial_evidence + frames[i]
    else:
        n_evidence = initial_evidence + i
    bolfi.fit(n_evidence=n_evidence)
    trans = matplotlib.transforms.ScaledTranslation(
        126/72, -5/72, fig.dpi_scale_trans
    )
    mean, var = bolfi.target_model.predict(t_plus_eval, noiseless=True)
    gps_at_each_frame.append(
        bolfi.target_model.predict(likelihood_eval, noiseless=True)
    )
    ax.plot(t_plus_eval, mean)
    eta_squared = 2 * np.log(
        n_evidence**4 * np.pi**2 / 0.3
    )
    ax.plot(t_plus_eval, mean - np.sqrt(eta_squared * var))
    unc = (
        np.exp(-(canvas[1] - mean.T)**2 / (2 * var.T))
        / np.sqrt(2 * np.pi * var.T)
    )
    ax.imshow(
        unc,
        cmap='gist_yarg',
        vmin=0,
        vmax=np.max(unc),
        aspect='auto',
        extent=[*x_range, *y_range]
    )
    ax.scatter(
        bolfi.target_model._gp.X, bolfi.target_model._gp.Y
    )
    ax.text(
        0.0, 1.0, str(n_evidence) + " samples    ",
        transform=ax.transAxes + trans,
        verticalalignment='top', horizontalalignment='right'
    )
fig_static.supxlabel(r"Model parameter $\theta$")
fig_static.supylabel("Model-data distance")
fig_static.tight_layout()

posterior = bolfi.extract_posterior()
threshold = posterior.threshold
likelihood_yrange = [-0.0012, 0.0082]
likelihood_canvas = np.meshgrid(
    np.linspace(*likelihood_range, 200),
    np.linspace(*reversed(likelihood_yrange), 200)
)
mean, var = bolfi.target_model.predict(likelihood_eval, noiseless=True)
likelihood = ss.norm().cdf((threshold - mean) / np.sqrt(var))
normalizing = np.sum(
    0.5
    * (likelihood[:-1] + likelihood[1:])
    * (t_plus_eval[1:] - t_plus_eval[:-1])
)

fig_int, (ax_gp, ax_int) = plt.subplots(figsize=(6, 4), nrows=2)
ax_gp.set_xlim(*likelihood_range)
ax_int.set_xlim(*likelihood_range)
ax_gp.xaxis.set_ticks([])
ax_int.xaxis.set_ticks([])
ax_gp.yaxis.set_ticks([])
ax_int.yaxis.set_ticks([])
ax_gp.set_ylabel(r"$\log||y_i(\theta)-y_i^\star||$")
ax_int.set_ylabel(r"$L_K(\theta)$")
ax_int.set_xlabel(r"Model parameter $\theta$")
trans_int = matplotlib.transforms.ScaledTranslation(
    10/72, -5/72, fig_int.dpi_scale_trans
)
ax_gp.text(0.0, 1.0, '(a)', transform=ax_gp.transAxes + trans_int,
           verticalalignment='top', fontsize=20)
ax_int.text(0.0, 1.0, '(b)', transform=ax_int.transAxes + trans_int,
            verticalalignment='top', fontsize=20)
trans_samples = matplotlib.transforms.ScaledTranslation(
    162/72, -10/72, fig.dpi_scale_trans
)
ax_gp.text(
    0.0, 1.0, str(n_evidence) + " samples    ",
    transform=ax_gp.transAxes + trans_samples, verticalalignment='top'
)
ax_gp.plot(likelihood_eval, mean)
ax_gp.plot(likelihood_eval, mean - np.sqrt(eta_squared * var))
ax_gp.plot(likelihood_eval, [threshold] * len(likelihood_eval))
ax_gp.scatter(
    bolfi.target_model._gp.X, bolfi.target_model._gp.Y
)
truncated_unc = (
    np.exp(-(likelihood_canvas[1] - mean.T)**2 / (2 * var.T))
    / np.sqrt(2 * np.pi * var.T)
) / normalizing * (likelihood_canvas[1] < threshold)
ax_gp.imshow(
    truncated_unc,
    cmap='gist_yarg',
    vmin=0,
    vmax=np.max(truncated_unc),
    aspect='auto',
    extent=[*likelihood_range, *likelihood_yrange]
)
ax_int.plot(
    likelihood_eval,
    likelihood / normalizing,
    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][4]
)

fig_int.tight_layout()

eps_min = -0.005
eps_max = 0.04
threshold_eval = np.linspace(eps_min, eps_max, 200)
norm = matplotlib.colors.Normalize(eps_min, eps_max)
cmap = plt.get_cmap('viridis')
fig_eps, ax_eps = plt.subplots(
    figsize=(4 * 2**0.5, 4), constrained_layout=True)
# mean, var = gps_at_each_frame[2]
for eps in threshold_eval:
    likelihood_eps = ss.norm().cdf((eps - mean) / np.sqrt(var))
    normalizing_eps = np.sum(
        0.5
        * (likelihood_eps[:-1] + likelihood_eps[1:])
        * (t_plus_eval[1:] - t_plus_eval[:-1])
    )
    ax_eps.plot(
        likelihood_eval,
        likelihood_eps / normalizing_eps,
        color=cmap(norm(eps))
    )
ax_eps.set_title("Normalized Likelihoods")
fig_eps.colorbar(matplotlib.cm.ScalarMappable(
    norm=norm, cmap=cmap
), ax=ax_eps, label="threshold")

plt.show()
