"""Affine fit by gradient descent on mean squared error.

``fit_affine(raw_x, raw_y)`` starts at a=0, b=0 and steps with
``LEARNING_RATE`` for ``STEPS`` iterations. ``gradient_step`` applies one
mean-reduced update. Reported loss is not an acceptance metric.
"""

LEARNING_RATE = 0.025
STEPS = 80


def gradient_step(raw_x, raw_y, a, b, learning_rate):
    count = len(raw_x)
    if count == 0 or count != len(raw_y):
        raise ValueError("fit series is empty or ragged")
    grad_a = 0.0
    grad_b = 0.0
    for x, y in zip(raw_x, raw_y):
        pred = a * x + b
        resid = y - pred
        grad_a += resid * x
        grad_b += resid
    grad_a /= count
    grad_b /= count
    return a - learning_rate * grad_a, b - learning_rate * grad_b


def fit_affine(raw_x, raw_y):
    a = 0.0
    b = 0.0
    history = []
    diverged = False
    count = len(raw_x)
    for _ in range(STEPS):
        loss = 0.0
        for x, y in zip(raw_x, raw_y):
            resid = (a * x + b) - y
            loss += resid * resid
        history.append(loss / count)
        a, b = gradient_step(raw_x, raw_y, a, b, LEARNING_RATE)
        if a != a or b != b or abs(a) > 1e3 or abs(b) > 1e3:
            diverged = True
            break
    final_loss = history[-1] if history else float("inf")
    return {
        "a": a,
        "b": b,
        "loss_history": history,
        "final_loss": final_loss,
        "diverged": diverged,
    }
