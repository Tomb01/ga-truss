import cairo
from src.classes.structure import *
import math

class Drawing(cairo.Context):
        
    def draw_structure(self, structure: Structure):
        self.set_line_width(0.01)
        m = cairo.Matrix(yy=-1, y0=200)
        self.transform(m)
        self.scale(100,100)
        self.save()
        
        for node in structure._nodes:
            self.draw_node(node)
        
        for truss in structure._trusses:
            start, end = truss.get_nodes()
            self.move_to(start.x, start.y)
            self.line_to(end.x, end.y)
            self.stroke()         
        
        self.save()
        
        
    def draw_node(self, node: TrussNode):
        x,y = node.get_coordinate()
        self.arc(x, y, 0.05, 0, 2*math.pi)
        self.fill()