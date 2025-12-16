#!/usr/bin/python
# -*- coding:utf-8 -*-

from .pdb_utils import AgAbComplex, Protein, VOCAB
from Bio import PDB
import os, json, pickle
import numpy as np

def _generate_chain_data(residues, start):
    backbone_atoms = VOCAB.backbone_atoms
    # Coords, Sequence, residue positions, mask for loss calculation (exclude missing coordinates)
    X, S, res_pos, xloss_mask = [], [], [], []
    # global node
    # coordinates will be set to the center of the chain
    X.append([[0, 0, 0] for _ in range(VOCAB.MAX_ATOM_NUMBER)])  
    S.append(VOCAB.symbol_to_idx(start))
    res_pos.append(0)
    xloss_mask.append([0 for _ in range(VOCAB.MAX_ATOM_NUMBER)])
    # other nodes
    for residue in residues:
        residue_xloss_mask = [0 for _ in range(VOCAB.MAX_ATOM_NUMBER)]
        bb_atom_coord = residue.get_backbone_coord_map()
        sc_atom_coord = residue.get_sidechain_coord_map()
        if 'CA' not in bb_atom_coord:
            for atom in bb_atom_coord:
                ca_x = bb_atom_coord[atom]
                print_log(f'no ca, use {atom}', level='DEBUG')
                break
        else:
            ca_x = bb_atom_coord['CA']
        x = [ca_x for _ in range(VOCAB.MAX_ATOM_NUMBER)]
        
        i = 0
        for atom in backbone_atoms:
            if atom in bb_atom_coord:
                x[i] = bb_atom_coord[atom]
                residue_xloss_mask[i] = 1
            i += 1
        for atom in residue.sidechain:
            if atom in sc_atom_coord:
                x[i] = sc_atom_coord[atom]
                residue_xloss_mask[i] = 1
            i += 1

        X.append(x)
        S.append(VOCAB.symbol_to_idx(residue.get_symbol()))
        res_pos.append(residue.get_id()[0])
        xloss_mask.append(residue_xloss_mask)
    X = np.array(X)
    center = np.mean(X[1:].reshape(-1, 3), axis=0)
    X[0] = center  # set center
    if start == VOCAB.BOA:  # epitope does not have position encoding
        res_pos = [0 for _ in res_pos]
    data = {'X': X, 'S': S, 'residue_pos': res_pos, 'xloss_mask': xloss_mask}
    return data


def get_epi_surf(cplx, msms_file = '/home/wenda_wang/workSpace/GMEAN/msms/') : 
    name = cplx.get_id()[:4]
    
    peptides = {}
    for chain in cplx.antigen.get_chain_names() : 
        peptides[chain] = cplx.antigen.get_chain(chain)
    protein = Protein(cplx.get_id(), peptides)
    structure = protein.to_bio()
    
    residue_info = []
    for epi in cplx.epitope : 
        c = epi[1]
        num = epi[0].get_id()[0]
        residue_info.append((c, num))
    residue_info = sorted(residue_info, key=lambda x: x[1])
    
    res_list = extract_residues(structure, residue_info)
    save_residues_to_pdb(res_list, msms_file + name + '_msms.pdb')
    
    target_pdb = msms_file + name + '_msms.pdb'
    cmd = 'pdb_to_xyzr ' + target_pdb + ' > ' + msms_file + name + '.xyzr;'
    cmd += 'msms -probe_radius 1.5 -if ' + msms_file + name + '.xyzr -af ' + msms_file + name + ' -of ' + msms_file + name
    
    try : 
        os.system(cmd)
        
        verts = []
        with open(msms_file + name + '.vert', 'r') as file:
            for i, line in enumerate(file):
                if i >= 3:
                    parts = line.split()
                    if len(parts) >= 3:
                        vert = [parts[0], parts[1], parts[2]]
                        verts.append(vert)
        to_rmove = [msms_file + name + '.xyzr', msms_file + name + '.face', msms_file + name + '.area']
        for f in to_rmove : 
            os.remove(f)
        
        return np.array(verts, dtype=np.float32)
    except Exception as e:
        print(f"Error occurred: {e}")
        os.remove(msms_file + name + '.xyzr')
        return np.array([])

def extract_residues(structure, residue_info):
    """
    Extract residues with specified serial numbers from the PDB structure.
    Parameters:
        structure: The input PDB structure object.
        residue_info: A two-dimensional list of residue information. Each element is [chain, residue_number].
    Return:
    A list of residues. Each element is of the class Bio.PDB.Residue.Residue.
    """
    residues = []
    for model in structure:
        for chain_id, residue_number in residue_info:
            chain = model[chain_id]
            for residue in chain.get_residues():
                if residue.get_id()[1] == residue_number:
                    residues.append(residue)
                    break  # 只添加第一个匹配的残基
    return residues

def save_residues_to_pdb(residues, output_pdb_file):
    """
    Save the residue list as a PDB file.
    Parameters:
        residues: A list of residues. Each element is of the class Bio.PDB.Residue.Residue.
        output_pdb_file: The path of the output PDB file.
    Return:
    There is no return value. The residues are directly saved as a PDB file.
    """
    # 创建一个空的PDB结构
    structure = PDB.Structure.Structure('structure')

    # 创建一个空的模型
    model = PDB.Model.Model(0)
    structure.add(model)

    # 创建一个空的链
    chain = PDB.Chain.Chain('1')
    model.add(chain)

    # 将残基添加到链中
    for i,  residue in enumerate(residues, start=1) :
        new_residue = PDB.Residue.Residue((' ', i, ' '), residue.get_resname(), residue.get_segid())
        for atom in residue:
            new_residue.add(atom.copy())
        chain.add(new_residue)
        

    # 创建PDBIO对象
    io = PDB.PDBIO()
    io.set_structure(structure)

    # 保存为PDB文件
    io.save(output_pdb_file)


def _cal_verts(verts , epitope_data : dict) : 
    X = epitope_data['X']
    num_epitope = X[1:, :].shape[0]

    shortest_list = []
    if verts.size == 0 : return {key: np.array([]) for key in range(X[1:, :].shape[0])}
    else : 
        for v in range(verts.shape[0]) : 
            vert_vec = verts[v]
            shortest_res = 0
            shortest_dist = np.inf
            for resc in range(1, X.shape[0]) : 
                res_mat = X[resc]
                dist = np.linalg.norm(res_mat - vert_vec, axis=1)
                if np.min(dist) < shortest_dist : 
                    shortest_dist = np.min(dist)
                    shortest_res = resc
            shortest_list.append(shortest_res)
        shortest_dist = np.array(shortest_list, dtype=np.int8)
        verts_info = np.column_stack((verts, shortest_dist))
        vert_by_res = [verts_info[np.where(verts_info[:, 3].astype(int) == i)][:, :3] for i in range(1, num_epitope + 1)]
        res_verts_dict = {}
        for i, vert_mat in enumerate(vert_by_res, start=1) : 
            res_verts_dict[i] = np.array(vert_mat)
        
        return res_verts_dict 

def pad_sample(epi_surface_verts : dict, length=50) : 
    fixed_verts = []
    for _, item in epi_surface_verts.items():
        n = item.shape[0]
        if n == 0 : fixed_verts.append(np.zeros((length, 3)))
        elif n == length : fixed_verts.append(item)
        elif n > length:
            sampled_array = np.random.choice(n, length, replace=False)
            fixed_verts.append(item[sampled_array])
        else : 
            padding = np.zeros((length - n, 3))
            fixed_verts.append(np.vstack((item, padding)))
            
    return np.array(fixed_verts)


def generate_surf_pkl(in_f, ou_f, length) :
    with open(in_f, 'r') as fin:
        lines = fin.read().strip().split('\n')
    test_surf = {}
    for line in lines : 
        item = json.loads(line)
        cplx = AgAbComplex.from_pdb(item['pdb_data_path'], item['heavy_chain'], item['light_chain'], item['antigen_chains'])
        pdb_name = cplx.pdb_id[:4]
        ag_residues = []
        for residue, chain, i in cplx.get_epitope():
            ag_residues.append(residue)
        ag_data = _generate_chain_data(ag_residues, VOCAB.BOA)
        epi_verts = get_epi_surf(cplx)
        epi_surface_verts = _cal_verts(epi_verts, ag_data)
        verts = pad_sample(epi_surface_verts, length)
        test_surf[pdb_name] = epi_surface_verts

        with open(ou_f, 'wb') as f : 
            pickle.dump(test_surf, f)