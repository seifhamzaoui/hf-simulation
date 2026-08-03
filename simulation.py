from ansys.aedt.core import Q3d
from config import *
import fnmatch
import numpy as np
import pandas as pd
import re
import pickle
from scipy.io import savemat
from traitement import extract_matrix_dict



# Opens (or creates) a new AEDT session and Q3D design
q3d = Q3d(
    project="my_project",
    design="my_design",
    non_graphical=False,    # True = run headless
    new_desktop=False        # True = launch a new AEDT instance
)
q3d.close_desktop()
# Opens (or creates) a new AEDT session and Q3D design
q3d = Q3d(
    project="my_project",
    design="my_design",
    non_graphical=False,    # True = run headless
    new_desktop=False        # True = launch a new AEDT instance
)
 

q3d.modeler.import_3d_cad(
    input_file=r"transformer_model.step",
    healing=False,              # attempt geometry healing on import
    refresh_all_ids=True,       # refresh object IDs after import (slow on big files)
    import_materials=False,     # pull material names from the CAD file if present
    create_lightweight_part=False,
    group_by_assembly=False,    # group imported bodies by their CAD assembly structure
    create_group=True,          # wrap imported bodies in an AEDT group
    merge_planar_faces=True
)

def assign_material_by_prefix(prefix, material,app=q3d):
    
    matches = [n for n in app.modeler.object_names if n.startswith(prefix)]

    for name in matches:
        obj = app.modeler.get_object_from_name(name)
        obj.material_name = material
    return matches

def delete_net_by_name(app, net_name):
    for b in app.boundaries:
        if b.name == net_name:
            b.delete()
            print(f"Deleted net: {net_name}")
            return True
    print(f"Net '{net_name}' not found")
    return False


# primary --------------------------------------------------------------------------------

assign_material_by_prefix('ringp', MATERIALS["ringp"]["aedt"])
if not p_conduc_circular : 
    min_area = (p_cond_width * p_cond_thickness) * 0.9
    max_area = (p_cond_width * p_cond_thickness) * 1.1
else : 
    min_area = (p_cond_thickness/2)**2 * 3.14 * 0.9
    max_area = (p_cond_thickness/2)**2 * 3.14 * 1.1

primary_rings = [name for name in q3d.modeler.object_names if name.startswith("ringp")]
primary_rings_obejcts = [q3d.modeler.get_object_from_name(n) for n in primary_rings]

for ring in primary_rings_obejcts:

    q3d.assign_net(
        assignment=ring.name,
        net_name=ring.name,
        net_type="Signal"
    )

    faces = ring.faces
    selected_faces = [f for f in ring.faces if min_area <= f.area <= max_area]
    print(len(selected_faces), "faces found in range")

    q3d.source(
        assignment=selected_faces[0].id,
        name=f"{ring.name}_src",
        net_name=ring.name,
        terminal_type="voltage"
    )
    q3d.sink(
        assignment=selected_faces[1].id,
        name=f"{ring.name}_snk",
        net_name=ring.name,
        terminal_type="voltage" 
    )





# secondary ----------------------------------------------------------------------------

assign_material_by_prefix('rings', MATERIALS["rings"]["aedt"])

if not p_conduc_circular : 
    min_area = (s_cond_width * s_cond_thickness) * 0.9
    max_area = (s_cond_width * s_cond_thickness) * 1.1
else : 
    min_area = (s_cond_thickness/2)**2 * 3.14 * 0.9
    max_area = (s_cond_thickness/2)**2 * 3.14 * 1.1



secondary_rings = [name for name in q3d.modeler.object_names if name.startswith("rings")]
secondary_rings_obejcts = [q3d.modeler.get_object_from_name(n) for n in secondary_rings]

for ring in secondary_rings_obejcts:
    q3d.assign_net(
        assignment=ring.name,
        net_name=ring.name,
        net_type="Signal"
    )

    faces = ring.faces
    selected_faces = [f for f in ring.faces if min_area <= f.area <= max_area]
    print(len(selected_faces), "faces found in range")

    q3d.source(
        assignment=selected_faces[0].id,
        name=f"{ring.name}_src",
        net_name=ring.name,
        terminal_type="voltage"
    )
    q3d.sink(
        assignment=selected_faces[1].id,
        name=f"{ring.name}_snk",
        net_name=ring.name,
        terminal_type="voltage" 
    )


assign_material_by_prefix('pinsulator', MATERIALS["pinsulator"]["aedt"])
assign_material_by_prefix('sinsulator', MATERIALS["sinsulator"]["aedt"])
assign_material_by_prefix('p_layer insulator', MATERIALS["p_layer insulator"]["aedt"])
assign_material_by_prefix('s_layer insulator', MATERIALS["s_layer insulator"]["aedt"])
assign_material_by_prefix('primary_secondary_insulation', MATERIALS["primary_secondary_insulation"]["aedt"])
assign_material_by_prefix('Core', MATERIALS["ringp"]["aedt"])  # solved as a plain copper conductor net for this stage

q3d.assign_net(
        assignment='Core',
        net_name='Core',
        net_type="Signal"
    )




# Capacitance simulation-----------------------------------------------------------------------------------------------------

setup = q3d.create_setup(name="CapSetup")

# if "DC" in setup.props:
#     del setup.props["DC"]
if "AC" in setup.props:
    del setup.props["AC"]

setup.props["SaveFields"] = False
setup.props["Cap"]["MaxPass"] = 10
setup.props["Cap"]["MinPass"] = 1
setup.props["Cap"]["MinConvPass"] = 1
setup.props["Cap"]["PerError"] = 1     # % convergence
setup.props["AdaptiveFreq"] = f"{min_freq_kHz}kHz"


setup.props["SaveFields"] = False
setup.props["DC"]["MaxPass"] = 10
setup.props["DC"]["MinPass"] = 1
setup.props["DC"]["MinConvPass"] = 1
setup.props["DC"]["PerError"] = 1     # % convergence
setup.props["AdaptiveFreq"] = f"{min_freq_kHz}kHz"

if sweep_active :

    sweep = setup.create_frequency_sweep(
        unit="kHz",
        start_frequency=min_freq_kHz,
        stop_frequency=max_freq_kHz,
        num_of_freq_points=number_of_point,
        name="CapSweep",
        sweep_type="Interpolating"
    )
setup.update()


input('validate the silumation')
q3d.analyze_setup("CapSetup")

freq_dict, net_names, freqs_hz = extract_matrix_dict(
    q3d, "C",
    setup_sweep_name=("CapSweep" if sweep_active else "CapSetup : LastAdaptive" ),
    quantities_category="C Matrix",
    matrix_context=None,    
    apply_reduction=True,
    target_freq_unit="kHz",
)

mat_safe_dict = {f"f_{k}": v for k, v in freq_dict.items()}
mat_safe_dict["frequencies_hz"] = freqs_hz   # ground-truth numeric array, always trust this over the labels

savemat("./traited Values/cap_data.mat", mat_safe_dict)
print("Saved cap_data.mat")
print(mat_safe_dict)

freq_dict, net_names, freqs_hz = extract_matrix_dict(
    q3d, "DCR",
    setup_sweep_name=("CapSweep" if sweep_active else "CapSetup : LastAdaptive" ),
    quantities_category="DCR Matrix",
    matrix_context=None,    
    apply_reduction=False,
    target_freq_unit="kHz",
)

mat_safe_dict = {f"f_{k}": v for k, v in freq_dict.items()}
mat_safe_dict["frequencies_hz"] = freqs_hz   # ground-truth numeric array, always trust this over the labels

savemat("./traited Values/DCR.mat", mat_safe_dict)
print("Saved cap_data.mat")
print(mat_safe_dict)


# q3d.export_equivalent_circuit(
#     output_file="circuit.cir"
# )
# End  Capacitance simulation-----------------------------------------------------------------------------------------------------



input('validate the second setup')

# Inductance simulation-----------------------------------------------------------------------------------------------------
assign_material_by_prefix('Core', MATERIALS["Core"]["aedt"])
delete_net_by_name(q3d, "Core")


setup = q3d.create_setup(name="InducSetup")

if "DC" in setup.props:
    del setup.props["DC"]
if "Cap" in setup.props:
    del setup.props["Cap"]

setup.props["SaveFields"] = False
setup.props["AC"]["MaxPass"] = 10
setup.props["AC"]["MinPass"] = 1
setup.props["AC"]["MinConvPass"] = 1
setup.props["AC"]["PerError"] = 1     # % convergence
setup.props["AdaptiveFreq"] = f"{min_freq_kHz}kHz"

if sweep_active :
    sweep = setup.create_frequency_sweep(
        unit="kHz",
        start_frequency=min_freq_kHz,
        stop_frequency=max_freq_kHz,
        num_of_freq_points=number_of_point,
        name="Induc_Sweep",
        sweep_type="Interpolating"
    )

setup.update()
input('validate the second simulation')

q3d.analyze_setup("InducSetup")

freq_dict, net_names, freqs_hz = extract_matrix_dict(q3d, "ACL",('Induc_Sweep' if sweep_active else 'InducSetup : LastAdaptive'),
                                                     "ACL Matrix",matrix_context=None,apply_reduction=False,target_freq_unit="kHz",)

mat_safe_dict = {f"f_{k}": v for k, v in freq_dict.items()}
mat_safe_dict["frequencies_hz"] = freqs_hz   # ground-truth numeric array, always trust this over the labels
mat_safe_dict = {f"f_{k}": v for k, v in freq_dict.items()}
savemat("./traited Values/induc.mat", mat_safe_dict)
print(mat_safe_dict)


freq_dict, net_names, freqs_hz =  extract_matrix_dict(q3d,'ACR',('Induc_Sweep' if sweep_active else 'InducSetup : LastAdaptive'),
                                                      'ACR Matrix', matrix_context=None, apply_reduction=False,target_freq_unit="kHz",)
mat_safe_dict = {f"f_{k}": v for k, v in freq_dict.items()}
mat_safe_dict["frequencies_hz"] = freqs_hz   # ground-truth numeric array, always trust this over the labels
savemat("./traited Values/Resis.mat", mat_safe_dict)
print(mat_safe_dict)








# 8. EXPORT AND PARSE THE Inductance MATRIX

# matrix_file = "./non traited Values/ind_matrix.csv"

# q3d.export_matrix_data(
#    file_name= matrix_file,
#     problem_type="AC RL",
#     setup="InducSetup",
#     matrix_type="Maxwell"
# )


# q3d.export_equivalent_circuit(
#     output_file="circuit.cir"
# )

# solution = q3d.post.get_solution_data(
#     expressions=expressions,
#     report_category="C"
# )

# ============================================================
# 9. RELEASE DESKTOP
# ============================================================
# q3d.release_desktop()


