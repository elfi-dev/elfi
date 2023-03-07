Implementing a new inference method
===================================

This tutorial provides the fundamentals for implementing custom parameter inference
methods using ELFI. ELFI provides many features out of the box, such as parallelization or
random state handling. In a typical case these happen "automatically" behind the scenes
when the algorithms are built on top of the provided interface classes.

The base class for parameter inference classes is the `ParameterInference`_ interface
which is found from the ``elfi.methods.inference.parameter_inference`` module. Among the methods in
the interface, those that must be implemented raise a ``NotImplementedError``. In
addition, you probably also want to override at least the ``update`` and ``__init__``
methods.

Let's create an empty skeleton for a custom method that includes just the minimal set of
methods to create a working algorithm in ELFI::

    from elfi.methods.inference.parameter_inference import ParameterInference

    class CustomMethod(ParameterInference):

        def __init__(self, model, output_names, **kwargs):
            super(CustomMethod, self).__init__(model, output_names, **kwargs)

        def set_objective(self):
            # Request 3 batches to be generated
            self.objective['n_batches'] = 3

        def extract_result(self):
            return self.state

The method ``extract_result`` is called by ELFI in the end of inference and should return
a ``ParameterInferenceResult`` object (``elfi.methods.result`` module). For illustration
we will however begin by returning the member ``state`` dictionary. It stores all the
current state information of the inference. Let's make an instance of our method and run
it::

    import elfi.examples.ma2 as ma2

    # Get a ready made MA2 model to test our inference method with
    m = ma2.get_model()

    # We want the outputs from node 'd' of the model `m` to be available
    custom_method = CustomMethod(m, ['d'])

    # Run the inference
    custom_method.infer()  # {'n_batches': 3, 'n_sim': 3000}

Running the above returns the state dictionary. We will find a few keys in it that track
some basic properties of the state, such as the ``n_batches`` telling how many batches has
been generated and ``n_sim`` that tells the number of total simulations contained in those
batches. It should be ``n_batches`` times the current batch size
(``custom_method.batch_size`` which was 1000 here by default).

You will find that the ``n_batches`` in the state dictionary had a value 3. This is
because in our ``CustomMethod.set_objective`` method, we set the ``n_batches`` key of the
objective dictionary to that value. Every `ParameterInference`_ instance has a Python
dictionary called ``objective`` that is a counterpart to the ``state`` dictionary. The
objective defines the conditions when the inference is finished. The default controlling
key in that dictionary is the string ``n_batches`` whose value tells ELFI how many batches
we need to generate in total from the provided generative ``ElfiModel`` model. Inference
is considered finished when the ``n_batches`` in the ``state`` matches or exceeds that in
the ``objective``. The generation of batches is automatically parallelized in the
background, so we don't have to worry about it.

.. note:: A batch in ELFI is a dictionary that maps names of nodes of the generative model
   to their outputs. An output in the batch consists of one or more runs of it's operation
   stored to a numpy array. Each batch has an index, and the outputs in the same batch are
   guaranteed to be the same if you recompute the batch.

The algorithm, however, does nothing else at this point besides generating the 3 batches.
To actually do something with the batches, we can add the ``update`` method that allows us
to update the state dictionary of the inference with any custom values. It takes in the
generated ``batch`` dictionary and it's index and is called by ELFI every time a new batch
is received. Let's say we wish to filter parameters by a threshold (as in ABC Rejection
sampling) from the total number of simulations::

    class CustomMethod(ParameterInference):
        def __init__(self, model, output_names, **kwargs):
            super(CustomMethod, self).__init__(model, output_names, **kwargs)

            # Hard code a threshold and discrepancy node name for now
            self.threshold = .1
            self.discrepancy_name = output_names[0]

            # Prepare lists to push the filtered outputs into
            self.state['filtered_outputs'] = {name: [] for name in output_names}

        def update(self, batch, batch_index):
            super(CustomMethod, self).update(batch, batch_index)

            # Make a filter mask (logical numpy array) from the distance array
            filter_mask = batch[self.discrepancy_name] <= self.threshold

            # Append the filtered parameters to their lists
            for name in self.output_names:
                values = batch[name]
                self.state['filtered_outputs'][name].append(values[filter_mask])

        ... # other methods as before

    m = ma2.get_model()
    custom_method = CustomMethod(m, ['d'])
    custom_method.infer()  # {'n_batches': 3, 'n_sim': 3000, 'filtered_outputs': ...}

After running this you should have in the returned state dictionary the
``filtered_outputs`` key containing filtered distances for node ``d`` from the 3 batches.

.. note:: The reason for the imposed structure in ``ParameterInference`` is to encourage a
   design where one can advance the inference iteratively using the ``iterate`` method.
   This makes it possible to stop at any point, check the current state and to be able to
   continue. This is important as there are usually many moving parts, such as summary
   statistic choices or deciding a good discrepancy function.

Now to be useful, we should allow the user to set the different options - the 3 batches is
not going to take her very far. The user also probably thinks in terms of simulations
rather than batches. ELFI allows you to replace the ``n_batches`` with ``n_sim`` key
in the objective to spare you from turning ``n_sim`` to ``n_batches`` in the code. Just
note that the ``n_sim`` in the state will always be in multiples of the ``batch_size``.

Let's modify the algorithm so, that the user can pass the threshold, the name of the
discrepancy node and the number of simulations. And let's also add the parameters to the
outputs::

    class CustomMethod(ParameterInference):
        def __init__(self, model, discrepancy_name, threshold, **kwargs):
            # Create a name list of nodes whose outputs we wish to receive
            output_names = [discrepancy_name] + model.parameter_names
            super(CustomMethod, self).__init__(model, output_names, **kwargs)

            self.threshold = threshold
            self.discrepancy_name = discrepancy_name

            # Prepare lists to push the filtered outputs into
            self.state['filtered_outputs'] = {name: [] for name in output_names}

        def set_objective(self, n_sim):
            self.objective['n_sim'] = n_sim

        ... # other methods as before

    # Run it
    custom_method = CustomMethod(m, 'd', threshold=.1, batch_size=1000)
    custom_method.infer(n_sim=2000)  # {'n_batches': 2, 'n_sim': 2000, 'filtered_outputs': ...}

Calling the inference method now returns the state dictionary that has also the filtered
parameters in it from each of the batches. Note that any arguments given to the ``infer``
method are passed to the ``set_objective`` method.

Now due to the structure of the algorithm the user can immediately continue from this
state::

    # Continue inference from the previous state (with n_sim=2000)
    custom_method.infer(n_sim=4000) # {'n_batches': 4, 'n_sim': 4000, 'filtered_outputs': ...}

    # Or use it iteratively
    custom_method.set_objective(n_sim=6000)

    custom_method.iterate()
    assert custom_method.finished == False
    # Investigate the current state
    custom_method.extract_result()  # {'n_batches': 5, 'n_sim': 5000, 'filtered_outputs': ...}

    self.iterate()
    assert custom_method.finished
    custom_method.extract_result()  # {'n_batches': 6, 'n_sim': 6000, 'filtered_outputs': ...}

This works, because the state is stored into the ``custom_method`` instance, and we only
change the objective. Also ELFI calls ``iterate`` internally in the ``infer`` method.

The last finishing touch to our algorithm is to convert the ``state`` dict to a more user
friendly format in the ``extract_result`` method. First we want to convert the list of
filtered arrays from the batches to a numpy array. We will then wrap the result to a
``elfi.methods.results.Sample`` object and return it instead of the ``state`` dict. Below
is the final complete implementation of our inference method class::

   import numpy as np

   from elfi.methods.inference.parameter_inference import ParameterInference
   from elfi.methods.results import Sample


   class CustomMethod(ParameterInference):
       def __init__(self, model, discrepancy_name, threshold, **kwargs):
           # Create a name list of nodes whose outputs we wish to receive
           output_names = [discrepancy_name] + model.parameter_names
           super(CustomMethod, self).__init__(model, output_names, **kwargs)

           self.threshold = threshold
           self.discrepancy_name = discrepancy_name

           # Prepare lists to push the filtered outputs into
           self.state['filtered_outputs'] = {name: [] for name in output_names}

       def set_objective(self, n_sim):
           self.objective['n_sim'] = n_sim

       def update(self, batch, batch_index):
           super(CustomMethod, self).update(batch, batch_index)

           # Make a filter mask (logical numpy array) from the distance array
           filter_mask = batch[self.discrepancy_name] <= self.threshold

           # Append the filtered parameters to their lists
           for name in self.output_names:
               values = batch[name]
               self.state['filtered_outputs'][name].append(values[filter_mask])

       def extract_result(self):
           filtered_outputs = self.state['filtered_outputs']
           outputs = {name: np.concatenate(filtered_outputs[name]) for name in self.output_names}

           return Sample(
               method_name='CustomMethod',
               outputs=outputs,
               parameter_names=self.parameter_names,
               discrepancy_name=self.discrepancy_name,
               n_sim=self.state['n_sim'],
               threshold=self.threshold
               )

Running the inference with the above implementation should now produce an user friendly
output::

   Method: CustomMethod
   Number of posterior samples: 82
   Number of simulations: 10000
   Threshold: 0.1
   Posterior means: t1: 0.687, t2: 0.152


Where to go from here
.....................

When implementing your own method it is advisable to read the documentation of the
`ParameterInference`_ class. In addition we recommend reading the ``Rejection``, ``SMC``
and/or ``BayesianOptimization`` class implementations from the source for some more
advanced techniques. These methods feature e.g. how to inject values from outside into the
ELFI model (acquisition functions in BayesianOptimization), how to modify the user
provided model to get e.g. the pdf:s of the parameters (SMC) and so forth.

Good to know
------------

ELFI guarantees that computing a batch with the same index will always produce the same
output given the same model and ``ComputationContext`` object. The ``ComputationContext``
object holds the batch size, seed for the PRNG, the pool object of precomputed batches
of nodes. If your method uses random quantities in the algorithm, please make sure
to use the seed attribute of ``ParameterInference`` so that your results will be
consistent.

If you want to provide values for outputs of certain nodes from outside the generative
model, you can return them from ``prepare_new_batch`` method. They will replace any
default value or operation in that node. This is used e.g. in ``BOLFI`` where values from
the acquisition function replace values coming from the prior in the Bayesian optimization
phase.

The `ParameterInference`_ instance has also the following helper classes:

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

.. autoclass:: elfi.methods.inference.parameter_inference.ParameterInference
   :members:
   :inherited-members: