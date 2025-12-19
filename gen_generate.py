#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm
from copy import deepcopy
import torch
from torch.utils.data import DataLoader

from data.dataset import E2EDataset
from data.pdb_utils import VOCAB, Residue, Peptide, Protein, AgAbComplex
from utils.logger import print_log
from utils.random_seed import setup_seed


def to_cplx(ori_cplx, ab_x, ab_s) -> AgAbComplex:

    ori_cplx_copy = deepcopy(ori_cplx)
    heavy_chain, light_chain = [], []
    chain = None
    for residue, residue_x in zip(ab_s, ab_x):
        residue = VOCAB.idx_to_symbol(residue)
        if residue == VOCAB.BOA:
            continue
        elif residue == VOCAB.BOH:
            chain = heavy_chain
            continue
        elif residue == VOCAB.BOL:
            chain = light_chain
            continue
        if chain is None:  # still in antigen region
            continue
        coord, atoms = {}, VOCAB.backbone_atoms + VOCAB.get_sidechain_info(residue)

        for atom, x in zip(atoms, residue_x):
            coord[atom] = x
        chain.append(Residue(
            residue, coord, _id=(len(chain), ' ')
        ))
    heavy_chain = Peptide(ori_cplx_copy.heavy_chain, heavy_chain)
    light_chain = Peptide(ori_cplx_copy.light_chain, light_chain)
    for res, ori_res in zip(heavy_chain, ori_cplx_copy.get_heavy_chain()):
        res.id = ori_res.id
    for res, ori_res in zip(light_chain, ori_cplx_copy.get_light_chain()):
        res.id = ori_res.id

    peptides = {
        ori_cplx_copy.heavy_chain: heavy_chain,
        ori_cplx_copy.light_chain: light_chain
    }
    antibody = Protein(ori_cplx_copy.pdb_id, peptides)
    cplx = AgAbComplex(
        ori_cplx_copy.antigen, antibody, ori_cplx_copy.heavy_chain,
        ori_cplx_copy.light_chain, skip_epitope_cal=True,
        skip_validity_check=True
    )
    cplx.cdr_pos = ori_cplx_copy.cdr_pos
    return cplx


def generate(args):

    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # model_type
    print_log(f'Model type: {type(model)}')

    # cdr type
    cdr_type = model.cdr_type
    print_log(f'CDR type: {cdr_type}')
    print_log(f'Paratope definition: {model.paratope}')

    # load test set
    test_set = E2EDataset(args.test_set, pep_file=args.pep_file, surf_file=args.surf_file, cdr=cdr_type)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             collate_fn=E2EDataset.collate_fn)


    # create save dir
    if args.save_dir is None:
        save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results'
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    idx = 0
    summary_items = []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            # generate
            del batch['xloss_mask']
            
            # Use model.sample_many to generate multiple samples
            list_X, list_S, list_pmets = model.sample_many(**batch, n_samples=args.n_samples)

            # Prepare batch_id for each protein in the batch
            if 'bid' in batch:
                batch_id = batch['bid']
            else:
                lengths = batch['lengths']
                batch_id = torch.zeros_like(batch['S'])  # [N]
                batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
                batch_id.cumsum_(dim=0)  # [N], item idx in the batch

            # Process each protein in the batch
            for sample_idx in range(args.n_samples):
                for i, bid in enumerate(batch_id):
                    ori_cplx = test_set.data[idx]
                    x = list_X[sample_idx][i]  # sample_idx选择第sample_idx个样本
                    s = list_S[sample_idx][i]
                    
                    # Generate complex for this sample
                    cplx = to_cplx(ori_cplx, x, s)
                    pdb_id = cplx.get_id().split('(')[0]

                    # Save mod_pdb for this sample
                    mod_pdb = os.path.join(save_dir, f'{pdb_id}_sample{sample_idx}.pdb')
                    cplx.to_pdb(mod_pdb)

                    # Save reference pdb (original) only once
                    ref_pdb = os.path.join(save_dir, f'{pdb_id}_original.pdb')
                    if sample_idx == 0:
                        ori_cplx.to_pdb(ref_pdb)

                    # Record the summary for this sample
                    summary_items.append({
                        'mod_pdb': mod_pdb,
                        'ref_pdb': ref_pdb,
                        'H': cplx.heavy_chain,
                        'L': cplx.light_chain,
                        'A': cplx.antigen.get_chain_names(),
                        'cdr_type': cdr_type,
                        'pdb': pdb_id,
                        'sample_idx': sample_idx,
                        'pmetric': None  # Assuming you don't have pmetrics here, otherwise modify accordingly
                    })
                    
                    del cplx  # Free memory
                idx += 1

    # write down the summary
    summary_file = os.path.join(save_dir, 'summary.json')
    with open(summary_file, 'w') as fout:
        fout.writelines(list(map(lambda item: json.dumps(item) + '\n', summary_items)))
    print_log(f'Summary of generated complexes written to {summary_file}')


def parse():
    parser = argparse.ArgumentParser(description='Generate antibodies given epitopes')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test_set', type=str, required=True, help='Path to test set')
    parser.add_argument('--pep_file', type=str, default='all_data/RAbD/test.pkl', help='pep file')
    parser.add_argument('--surf_file', type=str, default='all_data/RAbD/test_surf.pkl', help='pep file')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated antibodies')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')
    parser.add_argument('--n_samples', type=int, default=30, help='Number of samples to generate per protein')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of flow steps')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(2023)
    generate(parse())