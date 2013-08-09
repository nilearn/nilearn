from joblib import MemorizedResult


def unshelve(data):
    if isinstance(MemorizedResult):
        data = data.get()
    return data
