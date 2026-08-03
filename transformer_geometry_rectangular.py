# ============================================
# 2‑COLUMN TRANSFORMER – FULL PARAMETRIC MODEL
# Netgen / NGSolve Python Script
# ============================================

from netgen.occ import *
from ngsolve import *
from netgen.gui import *
import numpy as np
import ezdxf
from config import *

def write_dxf(shape2d, filename, points_per_edge=64):
    """Approximate every edge of a planar (constant-Z) OCC shape as a
    DXF LWPOLYLINE by sampling it along its curve parameter -- this
    works uniformly for straight and curved (fillet/circular) edges
    without needing to identify each edge's underlying curve type."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    for edge in shape2d.edges:
        t0, t1 = edge.parameter_interval
        pts = [edge.Value(t0 + (t1 - t0) * i / (points_per_edge - 1)) for i in range(points_per_edge)]
        msp.add_lwpolyline([(p.x, p.y) for p in pts])
    doc.saveas(filename)
    print(f"DXF created: {filename}")

# ============================================
# 1) INPUT PARAMETERS
# ============================================
def BoxCentered(center, width, depth, height):
    cx, cy, cz = center
    box = Box(
        Pnt(cx - width/2, cy, cz - depth/2),
        Pnt(cx + width/2,cy+height, cz + depth/2)
    )
    return box

# ============================================
# 2) DERIVED CALCULATIONS
# ============================================
hight1 = 2*gap_winding_yoke + (primary_turns_per_layer)*(p_cond_width+ 2*p_cond_insul_thickness)
hight2 = 2*gap_winding_yoke + (secondary_turns_per_layer)*(s_cond_width+ 2*s_cond_insul_thickness)
hight = max(hight1,hight2)


startingpoint = (0,0,0)
columnStart = startingpoint + (0,col_width,0)
bobinag_center = columnStart + (0, gap_winding_yoke, -col_width/2)
p_insulators = []
s_insulators = []
insulators = []
p_layer_insulators= []
s_layer_insulators = []
primary_rings = []
actual_Layer =[]
inner_radius = radius
p_counter = 1
for i in range(0,primary_n_layers):
    actual_Layer = []
    actual_ins_layer = []
    # layer insulator------------------------------------------------------------
    
    if i != 0 and p_layer_insulation != 0 :
        center = Pnt(col_width/2,col_width+0.01+gap_isulation_yoke,-col_depth/2 )
        inner_width = col_width  + 2*mandrel_thickness + 2*i * (p_cond_thickness + p_layer_insulation + 2*p_cond_insul_thickness) -2*p_layer_insulation
        outer_width = inner_width + 2*p_layer_insulation - 2*p_cond_insul_thickness
        inner_depth = col_depth + 2*mandrel_thickness + 2*i * (p_cond_thickness + p_layer_insulation + 2*p_cond_insul_thickness)-2*p_layer_insulation
        outer_depth = inner_depth + 2*p_layer_insulation - 2*p_cond_insul_thickness

        inner_box = BoxCentered(center,inner_width,inner_depth,hight-0.01-2*gap_winding_yoke +2*(gap_winding_yoke-gap_isulation_yoke))
        outer_box = BoxCentered(center,outer_width,outer_depth,hight-0.01-2*gap_winding_yoke +2*(gap_winding_yoke-gap_isulation_yoke))

        layer_ins = outer_box - inner_box

        y_edges = [e for e in layer_ins.edges
           if abs(e.start.x - e.end.x) < 1e-9
           and abs(e.start.z - e.end.z) < 1e-9]
        
        f_r_router = max(col_width, col_depth) + 2*p_cond_thickness + 2*mandrel_thickness

        inner_y_edges = [e for e in inner_box.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]
        outer_y_edges = [e for e in outer_box.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]
        
        if radius != 0 :
            layer_ins = layer_ins.MakeFillet(inner_y_edges, inner_radius + (i)*p_cond_thickness + (i-1)*p_layer_insulation)
            layer_ins = layer_ins.MakeFillet(outer_y_edges, inner_radius + (i)*p_cond_thickness + i*p_layer_insulation)
        
        
        layer_ins.name= "p_layer insulator"
        p_layer_insulators.append(layer_ins)
    # layer insulator------------------------------------------------------------------------------
    
    for j in range(0,primary_turns_per_layer):
        center = (col_width/2,col_width + gap_winding_yoke + j*(p_cond_width+ 2*p_cond_insul_thickness) +((hight2 - hight1)/2 if hight1 <= hight2 else 0),-col_depth/2 )
        inner_width = col_width  + 2*mandrel_thickness + 2*i * (p_cond_thickness + p_layer_insulation + 2*p_cond_insul_thickness)
        outer_width = inner_width + 2*p_cond_thickness
        inner_depth = col_depth + 2*mandrel_thickness + 2*i * (p_cond_thickness + p_layer_insulation + 2*p_cond_insul_thickness)
        outer_depth = inner_depth + 2*p_cond_thickness

        if p_conduc_circular and p_layer_insulation == 0  :
            if i % 2 == 1 and i > 0: 
                center = Pnt(col_width/2,col_width + gap_winding_yoke + j*(p_cond_width+ 2*p_cond_insul_thickness)+p_cond_width/2 + p_cond_insul_thickness + ((hight2 - hight1)/2 if hight1 <= hight2 else 0),-col_depth/2 )
        

        inner_box = BoxCentered(center,inner_width,inner_depth,p_cond_width)
        outer_box = BoxCentered(center,outer_width,outer_depth,p_cond_width)
        

        ring = outer_box - inner_box
        
        y_edges = [e for e in ring.edges
           if abs(e.start.x - e.end.x) < 1e-9
           and abs(e.start.z - e.end.z) < 1e-9]
        

        inner_y_edges = [e for e in inner_box.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]
        outer_y_edges = [e for e in outer_box.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]

        if radius != 0 :
            ring = ring.MakeFillet(inner_y_edges, inner_radius + i*p_layer_insulation +i*p_cond_thickness)
            ring = ring.MakeFillet(outer_y_edges, inner_radius + i*p_layer_insulation + (i+1)*p_cond_thickness)

        if p_conduc_circular :
            non_y_edges = [e for e in ring.edges if e not in y_edges]
            ring = ring.MakeFillet(non_y_edges, p_cond_thickness/2.01)
        
        ring.mat("copper")   # assign material name
        ring.name = f"ringp{p_counter}"
        primary_rings.append(ring)
        actual_Layer.append(ring)
        p_counter += 1

        ins_center = (col_width/2,col_width + gap_winding_yoke-p_cond_insul_thickness + j*(p_cond_width+ 2*p_cond_insul_thickness) +((hight2 - hight1)/2 if hight1 <= hight2 else 0),-col_depth/2 )
        ins_inner_width = col_width  + 2*mandrel_thickness + 2*i * (p_cond_thickness + p_layer_insulation + 2*p_cond_insul_thickness) - 2*p_cond_insul_thickness
        ins_outer_width = ins_inner_width + 2*p_cond_thickness + 4*p_cond_insul_thickness
        ins_inner_depth = col_depth + 2*mandrel_thickness + 2*i * (p_cond_thickness + p_layer_insulation + 2*p_cond_insul_thickness) - 2*p_cond_insul_thickness
        ins_outer_depth = ins_inner_depth + 2*p_cond_thickness + 4*p_cond_insul_thickness
        
        if p_conduc_circular and p_layer_insulation == 0 and i % 2 == 1 and i > 0:
                    ins_center = Pnt(col_width/2,col_width + gap_winding_yoke-p_cond_insul_thickness + j*(p_cond_width+ 2*p_cond_insul_thickness)+p_cond_width/2 + p_cond_insul_thickness + ((hight2 - hight1)/2 if hight1 <= hight2 else 0),-col_depth/2 )
        
        ins_inner_box = BoxCentered(ins_center,ins_inner_width,ins_inner_depth, (p_cond_width+ 2*p_cond_insul_thickness))
        ins_outer_box = BoxCentered(ins_center,ins_outer_width,ins_outer_depth, (p_cond_width+ 2*p_cond_insul_thickness))

        ins = ins_outer_box - ins_inner_box

        y_edges = [e for e in ins.edges
            if abs(e.start.x - e.end.x) < 1e-9
            and abs(e.start.z - e.end.z) < 1e-9]
        
        
        inner_y_edges = [e for e in ins_inner_box.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]
        outer_y_edges = [e for e in ins_outer_box.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]

        if radius != 0 :
            ins = ins.MakeFillet(inner_y_edges, inner_radius + i*p_layer_insulation+i*p_cond_thickness)
            ins = ins.MakeFillet(outer_y_edges, inner_radius + i*p_layer_insulation + (i+1)*p_cond_thickness)
        

        if p_conduc_circular :
            non_y_edges = [e for e in ins.edges if e not in y_edges]
            ins = ins.MakeFillet(non_y_edges, p_cond_thickness/2.01 + p_cond_insul_thickness)
        
        actual_ins_layer.append(ins)
    ins = sum(actual_ins_layer) - sum(actual_Layer)
    ins.name = f'pinsulator_{i}' 
    p_insulators.append(ins)


hight1 = 2*gap_winding_yoke + (j+1)*(p_cond_width+ 2*p_cond_insul_thickness)
max_r_outerb = outer_width/2 + p_cond_insul_thickness
max_outer_widthb = outer_width + 4*p_cond_insul_thickness
max_outer_depthb = outer_depth + 4*p_cond_insul_thickness
inner_radius = inner_radius + (i)*(p_layer_insulation + p_cond_thickness + p_cond_insul_thickness) + p_cond_thickness + p_cond_insul_thickness
print(max_r_outerb)

# gap insulator-------------------------------------------------------------------
if prim_secondary_gap > 0 :
    center = Pnt(col_width/2,col_width+0.01+gap_isulation_yoke,-col_depth/2 )
    inner_width = ins_outer_width  + 2*p_cond_insul_thickness
    outer_width = inner_width + 2*prim_secondary_gap - 2*s_cond_insul_thickness 
    inner_depth = ins_outer_depth  + 2*p_cond_insul_thickness
    outer_depth = inner_depth + 2*prim_secondary_gap - 2*s_cond_insul_thickness 



    inner_box = BoxCentered(center,inner_width,inner_depth,hight-0.01-2*gap_winding_yoke +2*(gap_winding_yoke-gap_isulation_yoke))
    outer_box = BoxCentered(center,outer_width,outer_depth,hight-0.01-2*gap_winding_yoke +2*(gap_winding_yoke-gap_isulation_yoke))

    primary_secondary_insulation = outer_box - inner_box
    primary_secondary_insulation.name= "primary_secondary_insulation"



    y_edges = [e for e in primary_secondary_insulation.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]


    inner_y_edges = [e for e in inner_box.edges
    if abs(e.start.x - e.end.x) < 1e-9
    and abs(e.start.z - e.end.z) < 1e-9]
    outer_y_edges = [e for e in outer_box.edges
    if abs(e.start.x - e.end.x) < 1e-9
    and abs(e.start.z - e.end.z) < 1e-9]

    if radius != 0 : 
        primary_secondary_insulation = primary_secondary_insulation.MakeFillet(inner_y_edges, inner_radius + 2*s_cond_insul_thickness )
        primary_secondary_insulation = primary_secondary_insulation.MakeFillet(outer_y_edges, inner_radius + prim_secondary_gap)
        inner_radius = inner_radius + prim_secondary_gap
else : 
    primary_secondary_insulation = Compound([])
# gap insulator-----------------------------------------------------------------------------



i = 0
j = 0
secondary_rings  = []
s_counter = 1
for i in range(0,secondary_n_layers):
    actual_Layer = []
    actual_ins_layer = []
    # layer insulator--------------------------------------------------------------------------------
    if i != 0 and s_layer_insulation != 0:
        center = Pnt(col_width/2,col_width+0.01+gap_isulation_yoke,-col_depth/2 )
        inner_width = max_outer_widthb + 2*prim_secondary_gap + 2*i * (s_cond_thickness + s_layer_insulation + 2*s_cond_insul_thickness) -2*s_layer_insulation 
        outer_width = inner_width + 2*s_layer_insulation - 2*s_cond_insul_thickness 
        inner_depth = max_outer_depthb + 2*prim_secondary_gap + 2*i * (s_cond_thickness + s_layer_insulation + 2*s_cond_insul_thickness)- 2*s_layer_insulation 
        outer_depth = inner_depth + 2*s_layer_insulation - 2*s_cond_insul_thickness 
        
        inner_box = BoxCentered(center,inner_width,inner_depth,hight-0.01-2*gap_winding_yoke +2*(gap_winding_yoke-gap_isulation_yoke))
        outer_box = BoxCentered(center,outer_width,outer_depth,hight-0.01-2*gap_winding_yoke +2*(gap_winding_yoke-gap_isulation_yoke))

        layer_ins = outer_box - inner_box

        y_edges = [e for e in layer_ins.edges
           if abs(e.start.x - e.end.x) < 1e-9
           and abs(e.start.z - e.end.z) < 1e-9]
        

        inner_y_edges = [e for e in inner_box.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]
        outer_y_edges = [e for e in outer_box.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]

        if radius != 0 :
            layer_ins = layer_ins.MakeFillet(inner_y_edges, inner_radius + (i)*s_cond_thickness + (i-1)*s_layer_insulation)
            layer_ins = layer_ins.MakeFillet(outer_y_edges, inner_radius + (i)*s_cond_thickness + i*s_layer_insulation)
        
        
        layer_ins.name= "s_layer insulator"
        s_layer_insulators.append(layer_ins)
    # layer insulator--------------------------------------------------------------------------------

    for j in range(0,secondary_turns_per_layer):
        center = (col_width/2,col_width + gap_winding_yoke + j*(s_cond_width+ 2*s_cond_insul_thickness) + ((hight1 - hight2)/2 if hight2 <= hight1 else 0),-col_depth/2 )
        inner_width = max_outer_widthb + 2*prim_secondary_gap + 2*i * (s_cond_thickness + s_layer_insulation + 2*s_cond_insul_thickness) 
        outer_width = inner_width + 2*s_cond_thickness 
        inner_depth = max_outer_depthb + 2*prim_secondary_gap + 2*i * (s_cond_thickness + s_layer_insulation + 2*s_cond_insul_thickness) 
        outer_depth = inner_depth + 2*s_cond_thickness 


        if s_conduc_circular and s_layer_insulation == 0  :
            if i % 2 == 1 and i > 0: 
                center = Pnt(col_width/2,col_width + gap_winding_yoke + j*(s_cond_width+ 2*s_cond_insul_thickness)+s_cond_width/2 + s_cond_insul_thickness + ((hight1 - hight2)/2 if hight2 <= hight1 else 0),-col_depth/2 )


        inner_box = BoxCentered(center,inner_width,inner_depth,s_cond_width)
        outer_box = BoxCentered(center,outer_width,outer_depth,s_cond_width)
        ring = outer_box - inner_box
        
        y_edges = [e for e in ring.edges
           if abs(e.start.x - e.end.x) < 1e-9
           and abs(e.start.z - e.end.z) < 1e-9]
        

        inner_y_edges = [e for e in inner_box.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]
        outer_y_edges = [e for e in outer_box.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]

        if radius != 0 :
            ring = ring.MakeFillet(inner_y_edges, inner_radius + i*s_layer_insulation +i*s_cond_thickness)
            ring = ring.MakeFillet(outer_y_edges, inner_radius + i*s_layer_insulation + (i+1)*s_cond_thickness)

        if s_conduc_circular :
            non_y_edges = [e for e in ring.edges if e not in y_edges]
            ring = ring.MakeFillet(non_y_edges, s_cond_thickness/2.01)
        
        ring.mat("copper")   # assign material name
        ring.name = f"rings{s_counter}"
        secondary_rings.append(ring)
        actual_Layer.append(ring)
        s_counter += 1
        ins_center = (col_width/2,col_width + gap_winding_yoke-s_cond_insul_thickness + j*(s_cond_width+ 2*s_cond_insul_thickness) + ((hight1 - hight2)/2 if hight2 <= hight1 else 0),-col_depth/2 )
        ins_inner_width = max_outer_widthb + 2*prim_secondary_gap + 2*i * (s_cond_thickness + s_layer_insulation + 2*s_cond_insul_thickness) - 2*s_cond_insul_thickness
        ins_outer_width =  ins_inner_width + 2*s_cond_thickness + 4*s_cond_insul_thickness 
        ins_inner_depth = max_outer_depthb + 2*prim_secondary_gap + 2*i * (s_cond_thickness + s_layer_insulation + 2*s_cond_insul_thickness) - 2*s_cond_insul_thickness
        ins_outer_depth = ins_inner_depth + 2*s_cond_thickness + 4*s_cond_insul_thickness

        if s_conduc_circular and s_layer_insulation == 0 and i % 2 == 1 and i>0:
                    ins_center = Pnt(col_width/2,col_width + gap_winding_yoke-s_cond_insul_thickness + j*(s_cond_width+ 2*s_cond_insul_thickness)+s_cond_width/2 + s_cond_insul_thickness + ((hight1 - hight2)/2 if hight2 <= hight1 else 0),-col_depth/2 )
        

        ins_inner_box = BoxCentered(ins_center,ins_inner_width,ins_inner_depth, (s_cond_width+ 2*s_cond_insul_thickness))
        ins_outer_box = BoxCentered(ins_center,ins_outer_width,ins_outer_depth, (s_cond_width+ 2*s_cond_insul_thickness))
            
        ins = ins_outer_box - ins_inner_box

        y_edges = [e for e in ins.edges
            if abs(e.start.x - e.end.x) < 1e-9
            and abs(e.start.z - e.end.z) < 1e-9]
        
        
        
        inner_y_edges = [e for e in ins_inner_box.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]
        outer_y_edges = [e for e in ins_outer_box.edges
        if abs(e.start.x - e.end.x) < 1e-9
        and abs(e.start.z - e.end.z) < 1e-9]

        if radius != 0 :
            ins = ins.MakeFillet(inner_y_edges, inner_radius + i*s_layer_insulation+i*s_cond_thickness)
            ins = ins.MakeFillet(outer_y_edges, inner_radius + i*s_layer_insulation + (i+1)*s_cond_thickness)
        
        if s_conduc_circular :
            non_y_edges = [e for e in ins.edges if e not in y_edges]
            ins = ins.MakeFillet(non_y_edges, s_cond_thickness/2.01 +s_cond_insul_thickness)

        actual_ins_layer.append(ins)
    ins = sum(actual_ins_layer) - sum(actual_Layer)
    ins.name = f'sinsulator_{i}' 
    s_insulators.append(ins)


hight2 = 2*gap_winding_yoke + (j+1)*(s_cond_width+ 2*s_cond_insul_thickness)

hight = max(hight1,hight2)
# #---------------------------------------------------------------------------------------------------------------
max_r_outer = outer_width/2 + s_cond_insul_thickness
max_outer_width = outer_width + 2*s_cond_insul_thickness
max_outer_depth = outer_depth + 2*s_cond_insul_thickness




# for i in range(0,primary_n_layers):
#     actual_Layer = []
#     for j in range(0,primary_turns_per_layer):
#         center = (col_width/2 + max_outer_width + bobine_spacing,col_width + gap_winding_yoke + j*(p_cond_width+ 2*p_cond_insul_thickness),-col_depth/2 )
#         inner_width = col_width  + 2*mandrel_thickness + 2*i * (p_cond_thickness + p_layer_insulation + 2*p_cond_insul_thickness)
#         outer_width = inner_width + 2*p_cond_thickness
#         inner_depth = col_depth + 2*mandrel_thickness + 2*i * (p_cond_thickness + p_layer_insulation + 2*p_cond_insul_thickness)
#         outer_depth = inner_depth + 2*p_cond_thickness
#         inner_box = BoxCentered(center,inner_width,inner_depth,p_cond_width)
#         outer_box = BoxCentered(center,outer_width,outer_depth,p_cond_width)
        
#         ring = outer_box - inner_box
#         ring.mat("copper")   # assign material name
#         ring.name = f"ringp{p_counter}"
#         primary_rings.append(ring)
#         actual_Layer.append(ring)
#         p_counter += 1
#     ins_center = (col_width/2 + max_outer_width + bobine_spacing,col_width + gap_winding_yoke-p_cond_insul_thickness,-col_depth/2 )
#     ins_inner_width = col_width  + 2*mandrel_thickness + 2*i * (p_cond_thickness + p_layer_insulation + 2*p_cond_insul_thickness) - p_cond_insul_thickness
#     ins_outer_width = ins_inner_width + 2*p_cond_thickness + 2*p_cond_insul_thickness
#     ins_inner_depth = col_depth + 2*mandrel_thickness + 2*i * (p_cond_thickness + p_layer_insulation + 2*p_cond_insul_thickness) - p_cond_insul_thickness
#     ins_outer_depth = ins_inner_depth + 2*p_cond_thickness + 2*p_cond_insul_thickness
#     ins_inner_box = BoxCentered(ins_center,ins_inner_width,ins_inner_depth, (j+1)*(p_cond_width+ 2*p_cond_insul_thickness))
#     ins_outer_box = BoxCentered(ins_center,ins_outer_width,ins_outer_depth, (j+1)*(p_cond_width+ 2*p_cond_insul_thickness))
# max_r_outerb = outer_width/2
# max_outer_widthb = outer_width
# max_outer_depthb = outer_depth


# i = 0
# j = 0
# for i in range(0,secondary_n_layers):
#     actual_Layer = []
#     for j in range(0,secondary_turns_per_layer):
#         center = (col_width/2 + + max_outer_width + bobine_spacing,col_width + gap_winding_yoke + j*(s_cond_width+ 2*s_cond_insul_thickness),-col_depth/2 )
#         inner_width = max_outer_widthb + 2*prim_secondary_gap + 2*i * (s_cond_thickness + s_layer_insulation + 2*s_cond_insul_thickness)
#         outer_width = inner_width + 2*s_cond_thickness
#         inner_depth = max_outer_depthb + 2*prim_secondary_gap + 2*i * (s_cond_thickness + s_layer_insulation + 2*s_cond_insul_thickness)
#         outer_depth = inner_depth + 2*s_cond_thickness
#         inner_box = BoxCentered(center,inner_width,inner_depth,s_cond_width)
#         outer_box = BoxCentered(center,outer_width,outer_depth,s_cond_width)
#         ring = outer_box - inner_box
#         ring.mat("copper")   # assign material name
#         ring.name = f"ringp{s_counter}"
#         secondary_rings.append(ring)
#         actual_Layer.append(ring)
#         s_counter += 1
#     ins_center = (col_width/2 + + max_outer_width + bobine_spacing,col_width + gap_winding_yoke-s_cond_insul_thickness,-col_depth/2 )
#     ins_inner_width = max_outer_widthb + 2*prim_secondary_gap + 2*i * (s_cond_thickness + s_layer_insulation + 2*s_cond_insul_thickness) - s_cond_insul_thickness
#     ins_outer_width =  ins_inner_width + 2*s_cond_thickness + 2*s_cond_insul_thickness
#     ins_inner_depth = max_outer_depthb + 2*prim_secondary_gap + 2*i * (s_cond_thickness + s_layer_insulation + 2*s_cond_insul_thickness) - s_cond_insul_thickness
#     ins_outer_depth = ins_inner_depth + 2*s_cond_thickness + 2*s_cond_insul_thickness
#     ins_inner_box = BoxCentered(ins_center,ins_inner_width,ins_inner_depth, (j+1)*(s_cond_width+ 2*s_cond_insul_thickness))
#     ins_outer_box = BoxCentered(ins_center,ins_outer_width,ins_outer_depth, (j+1)*(s_cond_width+ 2*s_cond_insul_thickness))
        
#     ins = ins_outer_box - ins_inner_box 
#     ins =  ins - sum(actual_Layer)
#     ins.name ='insolator' 
#     insulators.append(ins)



# Core and entrefer-----------------------------------
core = Box(Pnt(0,0,0),Pnt(col_width + 2*max_r_outer + bobine_spacing, 2*col_width+hight ,- col_depth )) - Box(Pnt(col_width,col_width,0),Pnt(2*max_r_outer+bobine_spacing,col_width+hight,-col_depth))
core.name = 'Core'

entrefers_list = []
if entrefer != 0 :
    for etfr in range(nombre_entrefer):
        center = (col_width/2,col_width +hight/2 + (-1)**etfr*(etfr)*(hight/2/nombre_entrefer), -col_depth/2 )
        centredbox1 = BoxCentered(center,col_width,col_depth,entrefer/nombre_entrefer) 
        centredbox1.name = f'entrefer1{etfr}'
        entrefers_list.append(centredbox1)
        center = (col_width/2 + 2*max_r_outer+bobine_spacing,col_width +hight/2 + (-1)**etfr*(etfr)*(hight/2/nombre_entrefer), -col_depth/2 )
        centredbox2 = BoxCentered(center,col_width,col_depth,entrefer/nombre_entrefer)
        centredbox2.name = f'entrefer2{etfr}'
        entrefers_list.append(centredbox2)

    
    

    core = core - sum(entrefers_list)
    core = Glue(core)
    for i,solid in enumerate(core.solids):      # <-- assign to EVERY resulting solid explicitly
            solid.name = f"Core{i}"
    # core.name = 'Core'
    entrefers_list = Compound(entrefers_list)

# Core and entrefer-----------------------------------

slicer = Box(Pnt(-0.01,-0.01,-col_depth/2+1),Pnt(-4*outer_depth,2*col_width+hight,-col_depth/2-1) )
slicer2 = Box(Pnt(col_width + 2*max_r_outer + bobine_spacing + 0.01,-0.01,-col_depth/2+1),Pnt(col_width + 2*max_r_outer + bobine_spacing+4*outer_depth,2*col_width+hight,-col_depth/2-1) )


# flyback winding -------------------------------------------
# P
arrays = np.array_split(primary_rings, primary_n_layers) if primary_n_layers != 0 else []
primary_rings = []
for i in range(len(arrays)):
    if i % 2 == 1 and not p_flyback:
        arrays[i] = arrays[i][::-1]
    for ele in arrays[i] :
        primary_rings.append(ele)

index = 0
for ring in primary_rings :
     ring.name = f"ringp{index+1}"
     index+=1

# S
arrays = np.array_split(secondary_rings, secondary_n_layers) if secondary_n_layers != 0 else []
secondary_rings = []
for i in range(len(arrays)):
    if i % 2 == 1 and not s_flyback:
        arrays[i] = arrays[i][::-1]
    for ele in arrays[i] :
        secondary_rings.append(ele)

index = 0
for ring in secondary_rings :
     ring.name = f"rings{index+1}"
     index+=1
# flyback winding -------------------------------------------

geo = Compound([*secondary_rings,*primary_rings,core,*p_insulators,*s_insulators,*p_layer_insulators,*s_layer_insulators,primary_secondary_insulation,entrefers_list]) - slicer -slicer2
# geo = primary_secondary_insulation * sum(s_insulators)  
wp = WorkPlane(Axes(Pnt(max_r_outer, col_width+hight/2, -col_depth/2+2),Z, X))
cut_face = wp.RectangleC(3*col_width+hight, 3*col_width + 2*max_r_outer + 2*bobine_spacing).Face() 
geo_to_cut = Compound([*secondary_rings,*primary_rings,core])
geo2d = geo_to_cut * cut_face
geo_2d = OCCGeometry(geo2d, dim=2)
geo_2d.shape.WriteStep("2D_transformer_model.step")
write_dxf(geo2d, "2D_transformer_model.dxf")

geo = OCCGeometry(geo)
geo.shape.WriteStep("transformer_model.step")
entrefer_solids = [entrefers_list] if entrefer != 0 else []
geo = Compound([*secondary_rings,*primary_rings,core,*p_insulators,*s_insulators,*p_layer_insulators,*s_layer_insulators,primary_secondary_insulation,*entrefer_solids,entrefers_list])
geo = OCCGeometry(geo)
geo.shape.WriteStep('transformer_model_closed.step')
print("step created")
Draw(geo)
# mesh = Mesh(geo.GenerateMesh(maxh=1))
# print(mesh.ne)
# Draw(mesh)
input('click to end')



 