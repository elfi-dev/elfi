import elfi
from elfi.examples import ma2

# load the model from elfi.examples
model = ma2.get_model()

# setup and run rejection sampling
rej = elfi.Rejection(model['d'], batch_size=10000)
result = rej.sample(1000, quantile=0.01)

# show summary of results on stdout
result.summary()
