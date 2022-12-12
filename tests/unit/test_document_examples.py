"""
This file contains tests for some of the examples used in the documentation that
are not automatically produced from the notebooks.

Note that if you change anything in this file (e.g. imports), you should change
the documentation accordingly. For this reason the imports and class definitions
are inside their respective functions.
"""


def test_implementing_new_algorithm():
    import numpy as np

    from elfi.methods.inference.parameter_inference import ParameterInference
    from elfi.methods.results import Sample

    import elfi.examples.ma2 as ma2

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
                threshold=self.threshold)

    # Below is from the part where we demonstrate iterative advancing

    # Run it
    m = ma2.get_model()
    custom_method = CustomMethod(m, 'd', threshold=.1, batch_size=1000)

    # Continue inference from the previous state (with n_sim=2000)
    custom_method.infer(n_sim=4000)

    # Or use it iteratively
    custom_method.set_objective(n_sim=6000)

    custom_method.iterate()
    assert custom_method.finished == False

    # Investigate the current state
    custom_method.extract_result()

    custom_method.iterate()
    assert custom_method.finished
    custom_method.extract_result()
