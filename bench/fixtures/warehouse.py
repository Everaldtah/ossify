"""Warehouse model used by the Ossify benchmark fixtures."""


class Warehouse:
    def __init__(self):
        self._stock = {}

    def add(self, sku, quantity):
        self._stock[sku] = self._stock.get(sku, 0) + quantity

    def remove(self, sku, quantity):
        if self._stock.get(sku, 0) < quantity:
            raise ValueError("not enough stock for " + sku)
        self._stock[sku] -= quantity

    def total(self):
        return sum(self._stock.values())
