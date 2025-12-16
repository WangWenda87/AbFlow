from models.isMEAN.isMEAN_model import isMEANModel
from models.isMEAN.isMEANOpt_model import isMEANOptModel
from data.pdb_utils import VOCAB, AgAbComplex, Protein
from utils.nn_utils import _knn_edges
from data.dataset import E2EDataset, _generate_chain_data
import json, torch, sys, pickle
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from Bio import PDB
from Bio.PDB.Structure import Structure as BStructure
from Bio.PDB.Model import Model as BModel
from Bio.PDB import PDBParser, PDBIO

from models.modules.am_egnn import AM_E_GCL, MS_E_GCL
from models.modules.am_enc import AMEncoder

args = {
    'embed_dim' : 64,
    'hidden_size' : 128,
    'k_neighbors' : 9,
    'n_layers' : 3,
    'bind_dist_cutoff' : 6.6, 
    "batch_size": 16, 
    "num_workers" : 4
}

## test model:
# test_set = E2EDataset('all_data/RAbD/test.json', pep_file='all_data/RAbD/test.pkl', surf_file='all_data/RAbD/test_surf.pkl', cdr='H3')
# train_set = E2EDataset('all_data/RAbD/bak_train.json', pep_file='all_data/RAbD/train.pkl', surf_file='all_data/RAbD/train_surf.pkl', cdr='H3')
# valid_set = E2EDataset('all_data/SKEMPI/bak_valid.json', pep_file='all_data/SKEMPI/valid.pkl', surf_file='all_data/SKEMPI/valid_surf.pkl', cdr='H3')
# test_loader = DataLoader(test_set, batch_size=args['batch_size'], num_workers=args['num_workers'], collate_fn=test_set.collate_fn)
# train_loader = DataLoader(train_set, batch_size=args['batch_size'], num_workers=args['num_workers'], collate_fn=train_set.collate_fn)
# valid_loader = DataLoader(valid_set, batch_size=args['batch_size'], num_workers=args['num_workers'], collate_fn=valid_set.collate_fn)


# sing = next(iter(train_loader))
# with open('sing.pkl', 'wb') as f:
#     pickle.dump(sing, f)
with open('sing.pkl', 'rb') as f:
    sing = pickle.load(f)
X, S, cmask, smask, residue_pos, paratope_mask, template, xloss_mask, X_pep, S_pep, surface, lengths = sing['X'], sing['S'], sing['cmask'], sing['smask'], sing['residue_pos'], sing['paratope_mask'], sing['template'], sing['xloss_mask'], sing['X_pep'], sing['S_pep'], sing['surface'], sing['lengths']
model = isMEANModel(args['embed_dim'], args['hidden_size'], VOCAB.MAX_ATOM_NUMBER, VOCAB.get_num_amino_acid_type(), num_verts=50, mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6.6,
                    n_layers=3, iter_round=3, dropout=0.1, pep_seq=True, pep_struct=True, struct_only=False, backbone_only=False, fix_channel_weights=False, pred_edge_dist=True, keep_memory=True, cdr_type='H3', paratope='H3', relative_position=False)

# loss, (snll, aar), (struct_loss, *struct_loss_details), (dock_loss, interface_loss, ed_loss, r_ed_losses), (pdev_loss, prmsd_loss) = model(X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, xloss_mask)
gen_X, gen_S, metric = model.sample(X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, n_steps=20)
# model = isMEANOptModel(args['embed_dim'], args['hidden_size'], VOCAB.MAX_ATOM_NUMBER,
#                    VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
#                    k_neighbors=9, bind_dist_cutoff=6.6,
#                    n_layers=3, struct_only=False,
#                    fix_atom_weights=False, cdr_type='H3')

## extract *_surf.pkl:
# from data.surface import get_epi_surf, _cal_verts
# def generate_surf_pkl(in_f, ou_f) :
#     with open(in_f, 'r') as fin:
#         lines = fin.read().strip().split('\n')
#     test_surf = {}
#     for line in lines : 
#         item = json.loads(line)
#         cplx = AgAbComplex.from_pdb(item['pdb_data_path'], item['heavy_chain'], item['light_chain'], item['antigen_chains'])
#         pdb_name = cplx.pdb_id[:4]
#         ag_residues = []
#         for residue, chain, i in cplx.get_epitope():
#             ag_residues.append(residue)
#         ag_data = _generate_chain_data(ag_residues, VOCAB.BOA)
#         epi_verts = get_epi_surf(cplx)
#         epi_surface_verts = _cal_verts(epi_verts, ag_data)
#         # verts = pad_sample(epi_surface_verts)
#         test_surf[pdb_name] = epi_surface_verts

#         with open(ou_f, 'wb') as f : 
#             pickle.dump(test_surf, f)

# import argparse
# def parse():
#     parser = argparse.ArgumentParser(description='Process surface data')
#     parser.add_argument('--in_f', type=str, required=True, help='input json file')
#     parser.add_argument('--ou_f', type=str, required=True, help='output pkl file')
#     return parser.parse_args()
    
# if __name__ == '__main__' :
#     args = parse()
#     generate_surf_pkl(args.in_f, args.ou_f)

# test dockq:
# from evaluation.bak_dockq import dockq as dq_old
# from evaluation.dockq import dockq as dq_new
# import csv

# def stat_dq(file : str, output_file : str) : 
#     with open(file, 'rb') as fin:
#         lines = fin.read().strip().split('\n')

#     items = [json.loads(line) for line in lines]
#     keys = ['mod_pdb', 'ref_pdb', 'H', 'L', 'A', 'cdr_type']

#     with open('dq_diff.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['name', 'score_old', 'score_new'])
#         for item in items : 
#             inputs = [item[key] for key in keys]
#             mod_pdb, ref_pdb, H, L, A, cdr_type = inputs
#             mod_cplx = AgAbComplex.from_pdb(mod_pdb, H, L, A, skip_epitope_cal=True)
#             ref_cplx = AgAbComplex.from_pdb(ref_pdb, H, L, A, skip_epitope_cal=False)
#             name = ref_cplx.get_id()[:4]
#             score_old = dq_old(mod_cplx, ref_cplx, cdrh3_only=True)
#             # score_new = dq_new(mod_cplx, ref_cplx, cdrh3_only=True)
#             writer.writerow([name, score_old])

