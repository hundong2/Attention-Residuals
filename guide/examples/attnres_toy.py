"""Attention Residuals의 depth attention을 설명하는 무의존성 toy example."""

from math import exp, sqrt


def rms_norm(vector: list[float], eps: float = 1e-8) -> list[float]:
    scale = sqrt(sum(x * x for x in vector) / len(vector) + eps)
    return [x / scale for x in vector]


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exps = [exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def attn_res(values: list[list[float]], query: list[float]):
    """이전 depth 표현들을 query-key 점수로 가중 합한다."""
    keys = [rms_norm(value) for value in values]
    logits = [sum(q * k for q, k in zip(query, key)) for key in keys]
    weights = softmax(logits)
    output = [sum(weight * value[d] for weight, value in zip(weights, values)) for d in range(len(query))]
    return output, weights


def add(*vectors: list[float]) -> list[float]:
    return [sum(items) for items in zip(*vectors)]


if __name__ == "__main__":
    layers = [[1.0, 0.0], [0.2, 1.0], [-0.4, 0.3], [0.8, -0.2]]
    standard = add(*layers)
    full, full_weights = attn_res(layers, query=[1.0, 0.0])

    blocks = [add(*layers[:2]), add(*layers[2:])]
    blocked, block_weights = attn_res(blocks, query=[1.0, 0.0])

    assert abs(sum(full_weights) - 1.0) < 1e-9
    assert abs(sum(block_weights) - 1.0) < 1e-9
    assert all(weight >= 0.0 for weight in full_weights + block_weights)

    print("standard sum:", standard)
    print("full weights/output:", full_weights, full)
    print("block weights/output:", block_weights, blocked)
    print("All invariants passed.")
