from netgen.occ import *
from ngsolve import *
from netgen.gui import *

# -----------------------------
# Parameters
# -----------------------------
r_outer = 1.0     # outer radius
r_inner = 0.7     # inner radius
height  = 0.2     # ring thickness
gap     = 0.05    # gap between rings
n_rings = 5       # number of rings

# -----------------------------
# Create rings
# -----------------------------
rings = []

for i in range(n_rings):
    
    z = i * (height + gap)

    outer = Cylinder(Pnt(0,0,z), Z, r_outer, h=height)
    inner = Cylinder(Pnt(0,0,z), Z, r_inner, h=height)

    ring = outer - inner
    ring.mat(f"ring{i}")   # assign material name
    rings.append(ring)

# -----------------------------
# Combine all rings
# -----------------------------


# -----------------------------
# Print materials
# -----------------------------


for i in range(0,len(rings)-1) :
    print(i)
    if i>0:
        rings[i-1].faces.name = None
    rings[i].faces.name = "voltage"
    for j in range(i+1,len(rings)):
        if j>i+1 :
            rings[j-1].faces.name = None
        rings[j].faces.name = "ground"
        outerBox = Box(Pnt(-2,-2,-2), Pnt(2,2,2))
        air = outerBox - rings[i] - rings[j]
        air.mat(name="air")
        geo = OCCGeometry(Compound([air,rings[i], rings[j]]))
        mesh = Mesh(geo.GenerateMesh(maxh=0.1))
        fes = H1(mesh, order=2, dirichlet="voltage|ground")
        u = fes.TrialFunction()
        v = fes.TestFunction()
        epsilon = 8.854e-12
          # permittivity
        a = BilinearForm(fes)
        a += epsilon * grad(u) * grad(v) * dx
        a.Assemble()
        f = LinearForm(fes)
        f.Assemble()
        gfu = GridFunction(fes)
        # Apply voltage values on boundaries
        gfu.Set(0, definedon=mesh.Boundaries("ground"))   # first ring → 0 V
        gfu.Set(10, definedon=mesh.Boundaries("voltage")) # second ring → 10 V
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
        scene = Draw(gfu, mesh, "Potential")
        scene.SetMaterial("air", transparent=True)
        #E = -grad(gfu)
        #Draw(E, mesh, "ElectricField")
        print(mesh.GetBoundaries())
        input("next")

print("Materials in mesh:")
print(mesh.GetMaterials())
print(mesh.GetBoundaries())