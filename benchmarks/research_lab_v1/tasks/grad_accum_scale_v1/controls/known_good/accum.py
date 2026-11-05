"""Average microbatch gradients of mean(0.5 * squared error).

``accumulated_gradient(batches, params)`` returns one gradient vector.
Each batch is a dict with ``x`` rows and ``y`` targets. Average the
microbatch gradients once. An empty batch list must raise ``ValueError``.
"""


def _mse_grad(features, targets, params):
    count = len(targets)
    width = len(params)
    grad = [0.0] * width
    for row, target in zip(features, targets):
        pred = 0.0
        for index, value in enumerate(row):
            pred += value * params[index]
        resid = pred - target
        for index, value in enumerate(row):
            grad[index] += value * resid
    scale = 1.0 / count
    return [value * scale for value in grad]


def accumulated_gradient(batches, params):
    if not batches:
        raise ValueError("batches must be non-empty")
    width = len(params)
    acc = [0.0] * width
    inv = 1.0 / len(batches)
    for batch in batches:
        grad = _mse_grad(batch["x"], batch["y"], params)
        for index, value in enumerate(grad):
            acc[index] += value
    acc = [value * inv for value in acc]
    return acc
