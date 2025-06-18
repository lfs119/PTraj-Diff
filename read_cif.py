import os

import numpy as np
from diff import residue_constants as rc
from Bio.PDB import MMCIFParser

def get_residue_info(structure):
    residue_count = 0
    residue_names = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # 排除非标准残基（如水分子）
                    residue_count += 1
                    residue_names.append(residue.get_resname())  # 获取残基名称

    return residue_count, residue_names

def structure_to_atom14(structure, length = 197):
    residue_count = len(list(structure.get_residues()))
    arr = np.zeros((1, residue_count, 14, 3), dtype=np.float32)
    for i, residue in enumerate(structure.get_residues()):
        resi_name = residue.get_resname()
        for atom in residue:
            at_name = atom.get_name()
            if at_name not in rc.restype_name_to_atom14_names[resi_name]:
                    #  print(resi_name, at_name, 'not found'); 
                     continue
            j = rc.restype_name_to_atom14_names[resi_name].index(at_name)
            coords = atom.get_coord()
            arr[:, i, j] = coords

    # for i, resi in enumerate(traj.top.residues):
    #     for at in resi.atoms:
    #         if at.name not in rc.restype_name_to_atom14_names[resi.name]:
    #             print(resi.name, at.name, 'not found'); continue
    #         j = rc.restype_name_to_atom14_names[resi.name].index(at.name)
    #         arr[:,i,j] = traj.xyz[:,at.index] * 10.0
    return arr[:, :length, :]

def find_cif_files_and_parent_dirs(base_path):
    cif_files_info = []

    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".cif"):  #  .cif 
                full_path = os.path.join(root, file)  
                parent_dir = os.path.basename(os.path.dirname(full_path))  # 获取父目录名
                cif_files_info.append({"file_path": full_path, "parent_dir": parent_dir})

    return cif_files_info

# parser = MMCIFParser()

# base_path_cif = '/home/xukai/af_output/ppi1'
# cif_files = find_cif_files_and_parent_dirs(base_path_cif)
# for item in cif_files:
#     print(item['file_path'], item['parent_dir'])
# # 读取CIF文件
#     structure = parser.get_structure('example', item['file_path'])
#     # print(type(structure))
#     arr = structure_to_atom14(structure)
#     print(arr.shape)

# for model in structure:
#     for chain in model:
#         for residue in chain:
#             for atom in residue:
#                 print(atom)