import numpy as np
from collections import Counter

class Node:
    def __init__(
        self,
        is_leaf=False,
        value=None,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        impurity=None,
        n_samples=None,
        class_counts=None
    ):
        self.is_leaf = is_leaf
        self.value = value

        self.feature_index = feature_index
        self.threshold = threshold

        self.left = left
        self.right = right

        self.impurity = impurity
        self.n_samples = n_samples
        self.class_counts = class_counts

def gini(y):
    counter_table = Counter(y)
    n = len(y)

    sum_of_p = 0

    for count in counter_table.values():
        p = (count / n) ** 2
        sum_of_p += p

    return 1 - sum_of_p

def mse(y):
    mean = np.mean(y)
    return np.mean((y - mean) ** 2)

def split(X, y, feature_index, threshold):
    X = np.asarray(X)
    y = np.asarray(y)

    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask

    X_left = X[left_mask]
    X_right = X[right_mask]
    y_left = y[left_mask]
    y_right = y[right_mask]
    
    return X_left, X_right, y_left, y_right

def impurity_reduction(parent, left, right, criterion):
    weighted_impurity = (len(left)/len(parent)) * criterion(left) + (len(right)/len(parent)) * criterion(right)
    parent_impurity = criterion(parent)

    return parent_impurity - weighted_impurity

def get_thresholds(feature_values):
    fv = np.unique(feature_values)

    if len(fv) < 2:
        return []
    
    thresholds = []

    for i in range(1, len(fv)):
        midpoint = (fv[i-1] + fv[i]) / 2 # mean of 2 points basically
        thresholds.append(midpoint)
    return thresholds

def find_best_split(X, y, criterion):
    n_features = X.shape[1]
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature in range(n_features):

        thresholds = get_thresholds(X[:, feature])

        for threshold in thresholds:
            _, _, y_left, y_right = split(X, y, feature, threshold=threshold)

            gain = impurity_reduction(y, y_left, y_right, criterion)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    return {
        "feature_index": best_feature,
        "threshold": best_threshold,
        "gain": best_gain
    }

def majority_class(y):
    class_counter = Counter(y)
    return class_counter.most_common(1)[0][0]
def mean_result(y):
    return np.mean(y)

def build_tree(X, y, criterion, leaf_value_func, depth=0, max_depth=5, min_samples_split=20):

    classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(classes, counts))

    if max_depth is not None and depth >= max_depth:
        return Node(
            is_leaf=True,
            value=leaf_value_func(y),
            impurity=criterion(y),
            n_samples=len(y),
            class_counts=class_counts
        )
    if len(y) < min_samples_split:
        return Node(
            is_leaf=True,
            value=leaf_value_func(y),
            impurity=criterion(y),
            n_samples=len(y),
            class_counts=class_counts
        )
    if len(np.unique(y)) == 1:
        return Node(
            is_leaf=True,
            value=leaf_value_func(y),
            impurity=criterion(y),
            n_samples=len(y),
            class_counts=class_counts
        )

    best_split = find_best_split(X, y, criterion)

    if best_split["gain"] <= 0:
        return Node(
            is_leaf=True,
            value=leaf_value_func(y),
            impurity=criterion(y),
            n_samples=len(y),
            class_counts=class_counts
        )
    
    feature_index = best_split["feature_index"]
    threshold = best_split["threshold"]

    X_left, X_right, y_left, y_right = split(X, y, feature_index=feature_index, threshold=threshold)

    left_child = build_tree(
        X_left,
        y_left,
        criterion=criterion,
        leaf_value_func=leaf_value_func,
        depth=depth + 1,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )

    right_child = build_tree(
        X_right,
        y_right,
        criterion=criterion,
        leaf_value_func=leaf_value_func,
        depth=depth + 1,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )

    return Node(
        value=leaf_value_func(y),
        feature_index=feature_index,
        threshold=threshold,
        left=left_child,
        right=right_child,
        impurity=criterion(y),
        n_samples=len(y),
        class_counts=class_counts
    )

def traverse_tree(x, node):
    if node.is_leaf:
        return node.value

    if x[node.feature_index] <= node.threshold:
        return traverse_tree(x, node.left)

    return traverse_tree(x, node.right)

def _format_number(value, precision):
    """Format plot values compactly while retaining the requested precision."""
    if value is None:
        return "None"
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.{precision}f}".rstrip("0").rstrip(".")


def _tree_depth(node):
    if node is None or node.is_leaf:
        return 0
    return 1 + max(_tree_depth(node.left), _tree_depth(node.right))


def _leaf_count(node):
    if node is None or node.is_leaf:
        return 1
    return _leaf_count(node.left) + _leaf_count(node.right)


def _node_positions(node):
    """Place leaves evenly and parents midway between their children."""
    positions = {}
    next_leaf_x = 0

    def visit(current, depth):
        nonlocal next_leaf_x
        if current.is_leaf:
            x = next_leaf_x
            next_leaf_x += 1
        else:
            left_x = visit(current.left, depth + 1)
            right_x = visit(current.right, depth + 1)
            x = (left_x + right_x) / 2
        positions[id(current)] = (x, -depth)
        return x

    visit(node, 0)
    return positions


def _class_color(node, classes, filled):
    if not filled:
        return "white"

    palette = [
        "#e58139", "#39e581", "#8139e5", "#e539c0", "#39a9e5",
        "#d6e539", "#e55f39", "#39e5c5", "#395fe5", "#e53963",
    ]
    counts = np.array(
        [node.class_counts.get(label, 0) for label in classes], dtype=float
    )
    total = counts.sum()
    if total == 0:
        return "white"

    winning_index = int(np.argmax(counts))
    winning_share = counts[winning_index] / total
    neutral_share = 1 / max(len(classes), 1)
    strength = (
        (winning_share - neutral_share) / (1 - neutral_share)
        if len(classes) > 1 else 1.0
    )
    strength = float(np.clip(strength, 0.08, 1.0))

    from matplotlib.colors import to_rgb
    base = np.array(to_rgb(palette[winning_index % len(palette)]))
    return tuple(1 - strength * (1 - base))


def _node_label(
    node,
    feature_names,
    class_names,
    classes,
    precision,
    impurity_name,
    is_classifier,
):
    lines = []
    if not node.is_leaf:
        feature_name = (
            feature_names[node.feature_index]
            if node.feature_index < len(feature_names)
            else f"Feature {node.feature_index}"
        )
        lines.append(
            f"{feature_name} <= {_format_number(node.threshold, precision)}"
        )

    lines.extend([
        f"{impurity_name} = {_format_number(node.impurity, precision)}",
        f"samples = {node.n_samples}",
    ])

    if is_classifier:
        counts = [int(node.class_counts.get(label, 0)) for label in classes]
        predicted_index = int(np.argmax(counts))
        predicted_class = (
            class_names[predicted_index]
            if predicted_index < len(class_names)
            else classes[predicted_index]
        )
        lines.append(f"value = {counts}")
        lines.append(f"class = {predicted_class}")
    else:
        lines.append(f"value = {_format_number(node.value, precision)}")

    return "\n".join(lines)


def print_tree(
    node,
    feature_names=None,
    class_names=None,
    *,
    is_classifier=True,
    impurity_name=None,
    filled=True,
    rounded=False,
    precision=3,
    figsize=None,
    ax=None,
    show=True,
):
    """Visualize a fitted custom decision tree with Matplotlib.

    Parameters mirror the most useful parts of ``sklearn.tree.plot_tree``.
    The returned ``(figure, axes)`` pair can be further customized or saved.
    """
    if node is None:
        raise ValueError("The tree has not been fitted yet.")

    import matplotlib.pyplot as plt

    max_depth = _tree_depth(node)
    leaves = _leaf_count(node)
    if figsize is None:
        figsize = (max(10, leaves * 2.1), max(6, (max_depth + 1) * 2.0))

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    feature_names = list(feature_names) if feature_names is not None else []
    classes = list(node.class_counts.keys()) if is_classifier else []
    class_names = list(class_names) if class_names is not None else classes
    impurity_name = impurity_name or (
        "gini" if is_classifier else "squared_error"
    )
    positions = _node_positions(node)

    def draw(current):
        x, y = positions[id(current)]

        if not current.is_leaf:
            for child, edge_text in (
                (current.left, "True"),
                (current.right, "False"),
            ):
                child_x, child_y = positions[id(child)]
                ax.annotate(
                    "",
                    xy=(child_x, child_y + 0.13),
                    xytext=(x, y - 0.13),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#333333",
                        "lw": 1.25,
                    },
                    zorder=1,
                )
                ax.text(
                    x + (child_x - x) * 0.35,
                    y + (child_y - y) * 0.35,
                    edge_text,
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "pad": 0.5,
                    },
                    zorder=2,
                )
                draw(child)

        facecolor = (
            _class_color(current, classes, filled)
            if is_classifier else ("#d9ecff" if filled else "white")
        )
        ax.text(
            x,
            y,
            _node_label(
                current,
                feature_names,
                class_names,
                classes,
                precision,
                impurity_name,
                is_classifier,
            ),
            ha="center",
            va="center",
            fontsize=9,
            linespacing=1.15,
            bbox={
                "boxstyle": (
                    "round,pad=0.35" if rounded else "square,pad=0.35"
                ),
                "facecolor": facecolor,
                "edgecolor": "#222222",
                "linewidth": 1.1,
            },
            zorder=3,
        )

    draw(node)
    ax.set_xlim(-0.75, max(leaves - 0.25, 0.75))
    ax.set_ylim(-max_depth - 0.65, 0.65)
    ax.set_axis_off()
    figure.tight_layout()

    if show:
        plt.show()

    return figure, ax
