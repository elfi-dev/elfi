from elfi.env import client as elfi_client

import distributed.client as dc

FIRST_COMPLETED = 0
ALL_COMPLETED = 1

# Mimics https://docs.python.org/3.4/library/asyncio-task.html#asyncio.wait
def wait(collections, client=None, return_when=FIRST_COMPLETED):
    """Calculates collections on client.

    Parameters
    ----------
    collections : list of dask collections or futures
    client : None or distributed.client.Client
        if None uses distributed.client.default_client()
    return_when : int
        if == FIRST_COMPLETED returns when first task completes
        if == ALL_COMPLETED returns when all tasks completed
        Currently supports only FIRST_COMPLETED.

    Returns
    -------
    tuple : (result, index, unfinished_futures)
    """
    if return_when not in (FIRST_COMPLETED, ALL_COMPLETED):
        raise ValueError("Unknown value for 'return_when'." +
                "Expected {} or {}.".format(FIRST_COMPLETED, ALL_COMPLETED) +
                "Received {}.".format(return_when))

    if return_when == ALL_COMPLETED:
        raise NotImplementedError("Support for ALL_COMPLETED not implemented.")

    client = client or elfi_client()
    futures = client.compute(collections)
    f = dc.as_completed(futures).__next__()
    i = futures.index(f)
    del futures[i]
    res = f.result()
    return res, i, futures
