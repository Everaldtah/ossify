"""Inventory helpers used by the Ossify benchmark fixtures."""


def restock_plan(items, threshold):
    """Return items that need restocking, lowest stock first.

    An item needs restocking when its stock is strictly below the threshold.
    """
    needed = [item for item in items if item["stock"] >= threshold]
    return sorted(needed, key=lambda item: item["stock"], reverse=True)
