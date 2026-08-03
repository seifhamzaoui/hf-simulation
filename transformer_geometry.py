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
p_counter = 1
for i in range(0,primary_n_layers):
    actual_Layer = []
    actual_ins_layer = []
    # layer insulator-------------------------------------------
    
    if i != 0 and p_layer_insulation != 0:
        center = Pnt(col_width/2,col_width+0.01+gap_isulation_yoke,-col_depth/2 )
        r_inner = max(col_width,col_depth)/2 + mandrel_thickness + i * (p_cond_thickness + p_layer_insulation + 2*p_cond_insul_thickness)-p_layer_insulation 
        r_outer = r_inner + p_layer_insulation - p_cond_insul_thickness-0.01
        outer = Cylinder(center, Y, r_outer, h=hight-0.01 - 2*gap_winding_yoke +2*(gap_winding_yoke-gap_isulation_yoke))
        inner = Cylinder(center, Y, r_inner, h=hight-0.01 - 2*gap_winding_yoke +2*(gap_winding_yoke-gap_isulation_yoke))
        layer_ins = outer - inner
        layer_ins.name= "p_layer insulator"
        p_layer_insulators.append(layer_ins)
     # layer insulator-------------------------------------
    for j in range(0,primary_turns_per_layer):
        center = Pnt(col_width/2,col_width + gap_winding_yoke + j*(p_cond_width+ 2*p_cond_insul_thickness) + ((hight2 - hight1)/2 if hight1 <= hight2 else 0),-col_depth/2 )
        r_inner = max(col_width,col_depth)/2 + mandrel_thickness + i * (p_cond_thickness + p_layer_insulation + 2*p_cond_insul_thickness)
        r_outer = r_inner + p_cond_thickness

        if p_conduc_circular and p_layer_insulation == 0  :
            r_inner = r_inner #- i*3*p_cond_insul_thickness # entre-enroulement
            r_outer = r_inner + p_cond_thickness
            if i % 2 == 1 and i > 0: 
                center = Pnt(col_width/2,col_width + gap_winding_yoke + j*(p_cond_width+ 2*p_cond_insul_thickness)+p_cond_width/2 + p_cond_insul_thickness + ((hight2 - hight1)/2 if hight1 <= hight2 else 0),-col_depth/2 )
        

        outer = Cylinder(center, Y, r_outer, h=p_cond_width)
        inner = Cylinder(center, Y, r_inner, h=p_cond_width)
        ring = outer - inner

        if p_conduc_circular :
            ring = ring.MakeFillet(ring.edges, p_cond_thickness/2.01)
        

        ring.mat("copper")   # assign material name
        ring.name = f"ringp{p_counter}"
        primary_rings.append(ring)
        actual_Layer.append(ring)
        p_counter += 1

        center_ins = Pnt(col_width/2,col_width + gap_winding_yoke-p_cond_insul_thickness + j*(p_cond_width+ 2*p_cond_insul_thickness) + ((hight2 - hight1)/2 if hight1 <= hight2 else 0),-col_depth/2 )

        if p_conduc_circular and p_layer_insulation == 0 and i % 2 == 1 and i > 0:
                    center_ins = Pnt(col_width/2,col_width + gap_winding_yoke-p_cond_insul_thickness + j*(p_cond_width+ 2*p_cond_insul_thickness)+p_cond_width/2 + p_cond_insul_thickness + ((hight2 - hight1)/2 if hight1 <= hight2 else 0),-col_depth/2 )
        
        outer = Cylinder(center_ins, Y, r_inner + p_cond_thickness + p_cond_insul_thickness , h=(p_cond_width+ 2*p_cond_insul_thickness))
        inner = Cylinder(center_ins, Y, r_inner - p_cond_insul_thickness , h=(p_cond_width+ 2*p_cond_insul_thickness))
        ins = outer - inner
        if p_conduc_circular :
            ins = ins.MakeFillet(ins.edges, p_cond_thickness/2.01 + p_cond_insul_thickness)
        actual_ins_layer.append(ins)
        
    ins = sum(actual_ins_layer) - sum(actual_Layer)
    ins.name = f'pinsulator_{i}' 
    p_insulators.append(ins)

max_r_outerb = r_outer + p_cond_insul_thickness

print(max_r_outerb)


# gap insulator-------------------------------------------------------------
center = Pnt(col_width/2,col_width+0.01+gap_isulation_yoke,-col_depth/2 )
r_inner = r_outer + p_cond_insul_thickness 
r_outer = r_inner + prim_secondary_gap - s_cond_insul_thickness
outer = Cylinder(center, Y, r_outer, h=hight-0.01-2*gap_winding_yoke +2*(gap_winding_yoke-gap_isulation_yoke))
inner = Cylinder(center, Y, r_inner, h=hight-0.01-2*gap_winding_yoke +2*(gap_winding_yoke-gap_isulation_yoke))
primary_secondary_insulation = outer - inner
primary_secondary_insulation.name= "primary_secondary_insulation"
# gap insulator----------------------------------------------------------


i = 0
j = 0
secondary_rings  = []
s_counter = 1
for i in range(0,secondary_n_layers):
    actual_Layer = []
    actual_ins_layer = []
    # layer insulator-----------------------------------------
    
    if i != 0 and s_layer_insulation != 0:
        center = Pnt(col_width/2,col_width+0.01 + gap_isulation_yoke,-col_depth/2 )
        r_inner = max_r_outerb + prim_secondary_gap + i * (s_cond_thickness + s_layer_insulation + 2*s_cond_insul_thickness)-s_layer_insulation
        r_outer = r_inner + s_layer_insulation - s_cond_insul_thickness
        outer = Cylinder(center, Y, r_outer, h=hight-0.01 - 2*gap_winding_yoke +2*(gap_winding_yoke-gap_isulation_yoke))
        inner = Cylinder(center, Y, r_inner, h=hight-0.01 - 2*gap_winding_yoke +2*(gap_winding_yoke-gap_isulation_yoke))
        layer_ins = outer - inner
        layer_ins.name= "s_layer insulator"
        s_layer_insulators.append(layer_ins)
    # layer insulator---------------------------------------------
    for j in range(0,secondary_turns_per_layer):
        center = Pnt(col_width/2,col_width + gap_winding_yoke + j*(s_cond_width + 2*s_cond_insul_thickness) + ((hight1 - hight2)/2 if hight2 <= hight1 else 0),-col_depth/2 )
        r_inner = max_r_outerb + prim_secondary_gap + i * (s_cond_thickness + s_layer_insulation + 2*s_cond_insul_thickness)
        r_outer = r_inner + s_cond_thickness

        if s_conduc_circular and s_layer_insulation == 0  :
            r_inner = r_inner #- i*2*s_cond_insul_thickness # entre-enroulement
            r_outer = r_inner + s_cond_thickness
            if i % 2 == 1 and i > 0: 
                center = Pnt(col_width/2,col_width + gap_winding_yoke + j*(s_cond_width+ 2*s_cond_insul_thickness)+s_cond_width/2 + s_cond_insul_thickness + ((hight1 - hight2)/2 if hight2 <= hight1 else 0),-col_depth/2 )

        outer = Cylinder(center, Y, r_outer, h=s_cond_width)
        inner = Cylinder(center, Y, r_inner, h=s_cond_width)
        ring = outer - inner
        if s_conduc_circular :
            ring = ring.MakeFillet(ring.edges, s_cond_thickness/2.01)
        
        ring.mat("copper")   # assign material name
        ring.name = f"rings{s_counter}"
        secondary_rings.append(ring)
        actual_Layer.append(ring)
        s_counter+=1
        center_ins = Pnt(col_width/2,col_width + gap_winding_yoke-s_cond_insul_thickness + j*(s_cond_width+ 2*s_cond_insul_thickness) + ((hight1 - hight2)/2 if hight2 <= hight1 else 0),-col_depth/2 )

        if s_conduc_circular and s_layer_insulation == 0 and i % 2 == 1 and i>0:
                    center_ins = Pnt(col_width/2,col_width + gap_winding_yoke-s_cond_insul_thickness + j*(s_cond_width+ 2*s_cond_insul_thickness)+s_cond_width/2 + s_cond_insul_thickness + ((hight1 - hight2)/2 if hight2 <= hight1 else 0),-col_depth/2 )
        
        inner = Cylinder(center_ins, Y, r_inner - s_cond_insul_thickness , h=(s_cond_width+ 2*s_cond_insul_thickness))
        outer = Cylinder(center_ins, Y, r_inner + s_cond_thickness + s_cond_insul_thickness , h=(s_cond_width+ 2*s_cond_insul_thickness))
        ins = outer - inner
        if s_conduc_circular :
            ins = ins.MakeFillet(ins.edges, s_cond_thickness/2.01 + s_cond_insul_thickness)
        actual_ins_layer.append(ins)
        # ins =  ins - sum(actual_Layer)
    ins = sum(actual_ins_layer) - sum(actual_Layer)
    ins.name =f'sinsulator_{i}'
    s_insulators.append(ins)


#---------------------------------------------------------------------------------------------------------------
max_r_outer = r_outer+s_cond_insul_thickness

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


# geo = [*secondary_rings,*primary_rings,core,*insulators,*p_layer_insulators,*s_layer_insulators]

slicer2 = Box(Pnt(col_width + 2*max_r_outer + bobine_spacing + 0.01,-0.01,-col_depth/2+1),Pnt(col_width + 2*max_r_outer + bobine_spacing+4*r_outer,2*col_width+hight,-col_depth/2-1) )
slicer = Box(Pnt(-0.01,-0.01,-col_depth/2+1),Pnt(-2*r_outer,2*col_width+hight,-col_depth/2-1)  )


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


geo = Compound([*secondary_rings,*primary_rings,core,*p_insulators,*s_insulators,*p_layer_insulators,*s_layer_insulators,primary_secondary_insulation,entrefers_list]) - slicer - slicer2

wp = WorkPlane(Axes(Pnt(max_r_outer, col_width+hight/2, -col_depth/2+2),Z, X))
cut_face = wp.RectangleC(3*col_width+hight, 3*col_width + 2*max_r_outer + 2*bobine_spacing).Face() 
geo_to_cut = Compound([*secondary_rings,*primary_rings,core])
geo2d = geo_to_cut * cut_face
geo_2d = OCCGeometry(geo2d, dim=2)
geo_2d.shape.WriteStep("2D_transformer_model.step")
write_dxf(geo2d, "2D_transformer_model.dxf")
# geo = Compound([*primary_rings])
print(len(primary_rings[0].faces))

# geo = sum(secondary_rings) * sum(insulators) 
geo = OCCGeometry(geo)
geo.shape.WriteStep("transformer_model.step")
geo = Compound([*secondary_rings,*primary_rings,core,*p_insulators,*s_insulators,*p_layer_insulators,*s_layer_insulators,primary_secondary_insulation,entrefers_list])
geo = OCCGeometry(geo)
geo.shape.WriteStep('transformer_model_closed.step')
print('step created')

Draw(geo,)

# mesh = Mesh(geo.GenerateMesh(maxh=1))
# print(mesh.ne)
# Draw(mesh)
input('click to end')


