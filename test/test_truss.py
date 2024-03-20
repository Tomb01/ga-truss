import unittest
from math import sqrt, cos
import sys

sys.path.insert(0, '../src')

from src.classes.truss import TrussNode, Truss
from src.classes.contrains import *

node1 = TrussNode(0,0)
node2 = TrussNode(2,2)

# Create a simple truss
truss = Truss(node1, node2, 1, 1)

class TestTruss(unittest.TestCase):

    def test_truss_inclination(self):
        self.assertAlmostEqual(cos(truss.get_inclination()), 1/sqrt(2))

    def test_truss_length(self):
        self.assertEqual(truss.getlength(), sqrt(2^2+2^2))

    def test_truss_constrain(self):
        from src.classes.contrains import HingeConstrain
        HingeConstrain(node1)
        self.assertEqual(node1.get_displacement(), (0,0))
        CartConstrainOrizontal(node2)
        self.assertAlmostEqual(node2.get_displacement()[1], 1/sqrt(2))
        self.assertAlmostEqual(node2.get_displacement()[0], 0)