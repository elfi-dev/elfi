Implementing a new method
=========================

In this tutorial we lay out the basics for implement a custom parameter inference method
using ELFI. ELFI provides many features out of the box, such as parallelization or random
state handling. In a typical case these happen "automatically" behind the scenes when
the algorithms are built on top of the provided interface classes.

The base class for parameter inference classes is the `ParameterInference`_ interface
which is found from the ``elfi.methods.parameter_inference`` module. Among the methods in
the interface, those that must be implemented raise a ``NotImplementedError``. In
addition, you probably also want to override at least the ``update`` and ``__init__``
methods.

Let's create an empty skeleton for a custom method that includes just the minimal set of
methods to create a working algorithm in ELFI::

    from elfi.methods.parameter_inference import ParameterInference
    from elfi.methods.result import Result

    class CustomMethod(ParameterInference):

        def __init__(self, model, output_names, **kwargs):
            super(CustomMethod, self).__init__(model, output_names, **kwargs)

        def set_objective(self, *args, **kwargs):
            pass

        def extract_result(self):
            return self.state

The method ``extract_result`` is called by ELFI in the end of inference and should return
a ``ParameterInferenceResult`` object (``elfi.methods.result`` module). For now we will
however return the member ``state`` dictionary that stores all the current state
information of the inference. Let's make now an instance of our method and run it::

    import elfi.examples.ma2

    # Get a ready made MA2 model to test our inference method with
    m = ma2.get_model()

    # We want the outputs for node 'd' from the model `m`
    custom_method = CustomMethod(m, ['d'])

    # Run the inference
    custom_method.infer()

Running the above returns the state dictionary. We will find a few keys in it that track
some basic properties of the state, such as the ``n_batches`` telling how many batches has
been processed and ``n_sim`` that tells the number of total simulations contained in those
batches. The algoritm however does nothing else at this point.

.. note:: A batch in ELFI is a dictionary that maps names of nodes of the generative model
   to their outputs. An output in the batch consists of one or more runs of it's operation
   stored to a numpy array. Each batch has an index, and the outputs in the same batch are
   guaranteed to be the same if you recompute the batch.

To have our algorithm do something useful, we first need to have an *objective* for it.
Every `ParameterInference`_ instance has a Python dictionary `objective` in it that is a
counterpart to the ``state`` dictionary. The objective defines the conditions when the
inference is finished. The default controlling key in that dictionary is the string
``n_batches`` that tells ELFI how many batches we need to generate in total from the
provided generative ``ElfiModel`` model. The generation of batches is automatically
parallelized in the background, so we don't have to worry about it.

Let say we want five batches::

    class CustomMethod(ParameterInference):
        ...

        def set_objective():
            self.objective['n_batches'] = 5


    # Set logging to show INFO level messages
    import logging
    logging.basicConfig(level=logging.INFO)

    ...

    # Run the inference
    custom_method.infer()

Running this you should see 5 log messages each telling about a received batch. Also you
will see that the ``n_batches`` in the state dictionary is now 5. The ``n_sim`` should be
five times the current ``batch_size`` (see ``custom_method.batch_size``). Now this is
still not very useful, we should do something with those 5 batches.

To use the batches, we shall override the ``update`` method that ELFI calls every time a
new batch is received. Let's filter all parameters whose distance is over a threshold, say
0.1. We shall redefine some of the earlier methods to add this functionality::

    class CustomMethod(ParameterInference):
        def __init__(self, model, discrepancy_name, **kwargs):
            # Create a name list of nodes whose outputs we wish to receive
            output_names = [discrepancy_name] + model.parameter_names
            super(CustomMethod, self).__init__(model, output_names, **kwargs)

            self.discrepancy_name = discrepancy_name

            # Prepare lists to push the filtered outputs into
            self.state['filtered_outputs'] = {name: [] for name in output_names}

        ...

        def update(self, batch, batch_index):
            super(CustomMethod, self).update(batch, batch_index)

            # Make a filter mask (logical numpy array) from the distance array
            filter_mask = batch[self.discrepancy_name] <= .1

            for name in self.output_names:
                # Take the output values from the batch
                values = batch[name]
                # Add the filtered values to its list
                self.state['filtered_outputs'][name].append(values[filter_mask])

    # Run it
    custom_method = CustomMethod(m, 'd', batch_size=1000)
    custom_method.infer()

.. note:: The reason for the imposed structure in ``ParameterInference`` is to encourage a
   design where one can advance the inference iteratively, that is, to stop at any point,
   check the current state and to be able to continue. This makes it possible to effectively
   stop early and tune the inference as there are usually many moving parts, such as summary
   statistic choices or deciding the best discrepancy function.

Calling the inference method now returns the state dictionary that has the filtered
parameters in it from each of the batches. The last step to complete the algorithm is to
convert the state dict to a more user friendly format in the ``extract_result`` method.
We will convert the result to a ``elfi.methods.result.Result`` object and return it. Below
is the final implementation of our inference method class::

    fefe

Where to go from here
.....................

When implementing your own method it is advisable to read the documentation of the
`ParameterInference`_ class. In addition we recommend reading the ``Rejection``, ``SMC``
and/or ``BayesianOptimization`` class implementations from the source to get you going
faster. These methods feature e.g. how to inject values from outside into the ELFI
model (acquisition functions in BayesianOptimization), how to modify the user provided
model to get e.g. the pdf:s of the parameters (SMC) and so forth.

Good to know
------------

ELFI guarantees that computing a batch with the same index will always produce the same
output given the same model and ``ComputationContext`` object. The ``ComputationContext``
object holds the batch size, seed for the PRNG, the pool object of precomputed batches
of nodes.

If you want to provide values for outputs of certain nodes from outside the generative
model, you can return them from ``prepare_new_batch`` method. They will replace any
default value or operation in that node. This is used e.g. in ``BOLFI`` where values from
the acquisition function replace values coming from the prior in the Bayesian optimization
phase.

The `ParameterInference`_ instance also the following helper classes:

``BatchHandler``
................

`ParameterInference`_ class instantiates a ``elfi.client.BatchHandler`` helper class that
is set as the ``self.batches`` member variable. This object is in essence a wrapper to the
``Client`` interface making it easier to work with batches that are in computation. Some
of the duties of ``BatchHandler`` is to keep track of the current batch_index and of the
status of the batches that have been submitted. You often don't need to interact with it
directly.

``OutputPool``
..............

``elfi.store.OutputPool`` serves a dual purpose:
1. It stores all the computed outputs of selected nodes
2. It provides those outputs when a batch is recomputed saving the need to recompute them.

Note however that reusing the values is not always possible. In sequential algorithms that
decide their next parameter values based on earlier results, modifications to the ELFI
model will invalidate the earlier data. On the other hand, Rejection sampling for instance
allows changing any of the summaries or distances and still reuse e.g. the simulations.
This is because all the parameter values will still come from the same priors.

.. _`ParameterInference`: `Parameter inference base class`_

Parameter inference base class
------------------------------

.. autoclass:: elfi.methods.parameter_inference.ParameterInference
   :members:
   :inherited-members: