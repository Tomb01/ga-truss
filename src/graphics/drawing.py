import cairo
from src.classes.structure import *

class Drawing(cairo.Context):
        
    def draw_structure(self, structure: Structure):
        self.set_line_width(0.01)
        m = cairo.Matrix(yy=-1, y0=200)
        self.transform(m)
        self.scale(100,100)
        self.save()
        
        for truss in structure._trusses:
            start, end = truss.get_nodes()
            self.move_to(start.x, start.y)
            self.line_to(end.x, end.y)
            self.stroke()         
        
        self.save()