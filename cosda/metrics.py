from __future__ import annotations

from collections import Counter, defaultdict


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def micro_f1_tags(gold_tags: list[list[str]], pred_tags: list[list[str]]) -> float:
    tp = fp = fn = 0
    for gold, pred in zip(gold_tags, pred_tags):
        for g, p in zip(gold, pred):
            if g == "O" and p == "O":
                continue
            if g == p:
                tp += 1
            else:
                if p != "O":
                    fp += 1
                if g != "O":
                    fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def expected_calibration_error(y_true: list[str], proba: list[dict[str, float]], bins: int = 10) -> float:
    buckets = defaultdict(list)
    for gold, probs in zip(y_true, proba):
        if not probs:
            continue
        pred = max(probs, key=probs.get)
        conf = probs[pred]
        idx = min(bins - 1, int(conf * bins))
        buckets[idx].append((pred == gold, conf))
    total = sum(len(v) for v in buckets.values())
    if total == 0:
        return 0.0
    ece = 0.0
    for items in buckets.values():
        acc = sum(ok for ok, _ in items) / len(items)
        conf = sum(c for _, c in items) / len(items)
        ece += len(items) / total * abs(acc - conf)
    return ece
