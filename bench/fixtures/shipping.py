"""Shipping helpers used by the Ossify benchmark fixtures."""

RATE_PER_KG = 2.75


def total_weight(items):
    # TODO-OSSIFY: sum the "w" key of every item and return the total
    raise NotImplementedError


def cost(items):
    return total_weight(items) * RATE_PER_KG
