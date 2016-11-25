from elfi.graph import Graph


class InferenceTask(Graph):
    """
    A context class for the inference. There is always one implicit `InferenceTask`
    created in the background. One can have several named inference tasks. Each will
    have a unique name and they keep book of e.g. the seed and the nodes related to the
    inference.

    All `Operation` nodes will belong to exactly one `InferenceTask`.
    """

    # Index of the default named `InferenceTask` to differentiate them
    default_task_index = 0

    def __init__(self, parameters=None, name=None, seed=0, sub_stream_index=0):
        """

        Parameters
        ----------
        parameters : list
        name : hashable
        seed : uint32
        sub_stream_index : uint

        """
        super(InferenceTask, self).__init__(name)
        self._parameters = parameters or []
        self.name = None
        self._set_name(name)
        self.seed = seed
        self.sub_stream_index = sub_stream_index

    def _set_name(self, name):
        if name is None:
            name = "default{}".format(self.__class__.default_task_index or "")
            self.__class__.default_task_index += 1
        self.name = name