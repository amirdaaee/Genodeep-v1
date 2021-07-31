import numpy as np


def drop_vi(model, data, n=1000):
    """

    Parameters
    ----------
    model: keras.model
    data:list(np.ndarray)
    n: int
        number of sampling

    Returns
    -------
    list(list(np.ndarray))
    """
    new_data = []
    l = data[0].shape[0]
    for d in data:
        new_data.append(np.repeat(d, n, axis=0))

    output = model.predict(new_data)
    if output.__class__ is not list:
        output = [output]
    result = []
    for o in output:
        shape = (l, o.shape[1])
        result.append([np.zeros(shape), np.zeros(shape), np.zeros(shape)])
        for c, i in enumerate(np.arange(0, n * l, n)):
            p_hat = o[i:i + n, :]
            result[-1][0][c, :] = np.mean(p_hat, axis=0)
            result[-1][1][c, :] = np.mean(p_hat * (1 - p_hat), axis=0)
            result[-1][2][c, :] = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    return result
