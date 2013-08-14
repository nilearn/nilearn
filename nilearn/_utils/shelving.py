from joblib import MemorizedResult


def unshelve(data):
    if isinstance(data, MemorizedResult):
        data = data.get()
    return data
