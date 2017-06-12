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

        def __init__(self, model, outputs, **kwargs):
            super(CustomMethod, self).__init__(model, outputs, **kwargs)

        def set_objective(self, *args, **kwargs):
            pass

        def extract_result(self):
            pass

We can now make an instance of it and run it::

    import elfi.examples.ma2

    # Get a ready made MA2 model to test our inference method with
    m = ma2.get_model()

    # We want the outputs for node 'd' from the model `m`
    custom_method = CustomMethod(ma2, ['d'])

    # Run the inference
    custom_method.infer()

Running the above does nothing. To have our algorithm do something useful, we first need
to have an *objective* for it. Every `ParameterInference`_ instance has a Python
dictionary `objective` in it that defines the conditions when the inference is finished.
The default controlling key in that dictionary is the string ``n_batches`` that tells ELFI
how many batches we need to generate in total from the provided generative model
``ElfiModel``. The generation of batches is automatically parallelized in the background,
so we don't have to worry about it.

.. note:: A batch in ELFI is a dictionary that maps names of nodes of the generative model
   to their outputs. An output in the batch consists of one or more runs of it's operation
   stored to a numpy array. Each batch has an index, and the outputs in the same batch are
   guaranteed to be the same if you recompute the batch.

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

Running this you should see 5 log messages each telling about a received batch. Now this
is still not very useful, we should do something with the batches.

To use the batches, we shall override the ``update`` method that ELFI calls every time a
new batch is received. Let's filter all parameters whose distance is over a threshold, say
0.5. We shall redefine some of the earlier methods to add this functionality::

    class CustomMethod(ParameterInference):
        def __init__(model, distance_name):
            # Create a name list of nodes whose outputs we wish to receive
            outputs = [distance_name] + model.parameters
            super(CustomMethod, self).__init__(model, outputs)

            self.distance_name = distance_name

            # Prepare lists to push the filtered outputs into
            self.state['filtered_outputs'] = {name: [] for name in outputs}

        def update(batch, batch_index):
            super(CustomMethod, self).update(batch, batch_index)

            # Make a filter mask (logical numpy array) from the distance array
            filter_mask = batch[self.distance_name] < .5

            for name in self.outputs:
                # Take the output values from the batch
                values = batch[name]
                # Add the filtered values to its list
                self.state['filtered_outputs'][name].append(values[filter_mask])


Above we used member variable ``self.state``. This is a counterpart to the
``self.objective`` dictionary. The ``state`` dictionary stores all the state information
of the inference. If you investigate this dictionary, you will see that in the end of the
inference, it has an ``n_batches`` key and the ``filtered_outputs`` key that we defined.
The ``n_batches`` was added there by ELFI in the ``super`` of ``update`` and allowed ELFI
to keep track when to stop generating new batches.

.. note:: The reason for the imposed structure in ``ParameterInference`` is to encourage a
   design where one can advance the inference iteratively, that is, to stop at any point,
   check the current state and to be able to continue. This makes it possible to effectively
   stop early and tune the inference as there are usually many moving parts, such as summary
   statistic choices or deciding the best discrepancy function.

Calling the inference still doesn't return anything. This is because we still have to
extract the result out of the ``state`` dictionary and return it.


ELFI guarantees that computing a batch with the same index will always produce the same
output given the same model and ``ComputationContext`` object. The ``ComputationContext``
object holds the batch size, seed for the PRNG, and a pool of precomputed batches of nodes
and the observed values of the nodes.

When a new ``ParameterInference`` is constructed, it will make a copy of the user provided
``ElfiModel`` and make a new ``ComputationContext`` object for it. The user's model will stay
intact and the algorithm is free to modify it's copy as it needs to.


Implementing the ``__init__`` method
------------------------------------

You will need to call the ``InferenceMethod.__init__`` with a list of outputs, e.g. names of
nodes that you need the data for in each batch. For example, the rejection algorithm needs
the parameters and the discrepancy node output.

The first parameter to your ``__init__`` can be either the ElfiModel object or directly a
"target" node, e.g. discrepancy in rejection sampling. Assuming your ``__init__`` takes an
optional discrepancy parameter, you can detect which one was passed by using
``_resolve_model`` method::

   def __init__(model, discrepancy, ...):
      model, discrepancy = self._resolve_model(model, discrepancy)

In case you need multiple target nodes, you will need to write your own resolver.


Explanations for some members of ``ParameterInference``
-------------------------------------------------------

- objective : dict
    Holds the data for the algorithm to internally determine how many batches are still
    needed. You must have a key ``n_batches`` here. This information is used to determine
    when the algorithm is finished.

- state : dict
    Stores any temporal data related to achieving the objective. Must include a key
    ``n_batches`` for determining when the inference is finished.


Good to know
------------

``BatchHandler``
................

``ParameterInference`` class instantiates a ``elfi.client.BatchHandler`` helper class for you and
assigns it to ``self.batches``. This object is in essence a wrapper to the ``Client``
interface making it easier to work with batches that are in computation. Some of the
duties of ``BatchHandler`` is to keep track of the current batch_index and of the status of
the batches that have been submitted. You may however may not need to interact with it
directly.

``OutputPool``
..............

``elfi.store.OutputPool`` serves a dual purpose:
1. It stores the computed outputs of selected nodes
2. It provides those outputs when a batch is recomputed saving the need to recompute them.

If you want to provide values for outputs of certain nodes from outside the generative
model, you can return then in ``prepare_new_batch`` method. They will be inserted into to
the ``OutputPool`` and will replace any value that would have otherwise been generated from
the node. This is used e.g. in ``BOLFI`` where values from the acquisition function replace
values coming from the prior in the Bayesian optimization phase.


.. hint:: For more advanced algorithms, it may be beneficial to read the ``Rejection``,
   ``SMC`` and/or ``BayesianOptimization`` class implementations to get you going faster.


.. _`ParameterInference`: `Parameter inference base class`_

Parameter inference base class
------------------------------

.. autoclass:: elfi.methods.parameter_inference.ParameterInference
   :members:
   :inherited-members: