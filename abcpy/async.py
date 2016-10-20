import distributed.client as dc

FIRST_COMPLETED = 0
ALL_COMPLETED = 1

# Mimics https://docs.python.org/3.4/library/asyncio-task.html#asyncio.wait
def wait(collections, client=None, return_when=FIRST_COMPLETED):
    """
    Currently supports only FIRST_COMPLETED
    Returns
    -------
    (tuple) (result, index, unfinished_futures)
    """

    client = client or dc.default_client()
    futures = client.compute(collections)
    f = dc.as_completed(futures).__next__()
    i = futures.index(f)
    del futures[i]
    res = f.result()
    del f
    return res, i, futures