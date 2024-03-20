import unittest
from math import sqrt, cos
import sys

sys.path.insert(0, '../src')

from src.classes.truss import Truss
from src.classes.node import TrussNode

node1 = TrussNode(0,0)
node2 = TrussNode(2,2)

# Create a simple truss
truss = Truss(node1, node2, 1, 1)

class TestTruss(unittest.TestCase):

    def test_truss_inclination(self):
        self.assertAlmostEqual(cos(truss.get_inclination()), 1/sqrt(2))

    def test_truss_length(self):
        self.assertEqual(truss.getlength(), sqrt(2^2+2^2))