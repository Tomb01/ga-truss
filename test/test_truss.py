import unittest
from math import sqrt, cos, pi, sin
import sys
import numpy as np

sys.path.insert(0, '../src')

from src.classes.truss import TrussNode, Truss
from src.classes.contrains import *
from src.classes.structure import Structure

class TestTruss(unittest.TestCase):

    def test_truss_inclination(self):
        node1 = TrussNode(0.42264973,0)
        node2 = TrussNode(1,1)
        HingeConstrain(node1)
        # Create a simple truss
        truss = Truss(node1, node2, 1, 1)
        self.assertAlmostEqual(cos(truss.get_inclination()), 0.5)
        self.assertAlmostEqual(truss.get_length(), 1.154700538)
        