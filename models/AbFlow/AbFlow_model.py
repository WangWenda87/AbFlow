#!/usr/bin/python
# -*- coding:utf-8 -*-
import math, time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from data.pdb_utils import VOCAB
from utils.nn_utils import SeparatedAminoAcidFeature, ProteinFeature
from utils.nn_utils import GMEdgeConstructor, SeperatedCoordNormalizer
from utils.nn_utils import _knn_edges
from evaluation.rmsd import kabsch_torch

from ..modules.am_enc import AMEncoder
from ..modules.am_egnn import AMEGNN


class AbFlowModel(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, num_classes, num_verts, 
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6,
                 n_layers=3, iter_round=3, dropout=0.1, 
                 pep_seq=True, pep_struct=True, struct_only=False,
                 backbone_only=False, fix_channel_weights=False, pred_edge_dist=True,
                 keep_memory=True, cdr_type='H3', paratope='H3', relative_position=False,
                 sigma_min=0.01, flow_weight=1.0, sequence_flow_weight=1.0,
                 time_embed_dim=32) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.num_classes = num_classes
        self.bind_dist_cutoff = bind_dist_cutoff
        self.k_neighbors = k_neighbors
        self.round = iter_round
        
        self.pep_seq = pep_seq
        self.pep_struct = pep_struct
        self.struct_only = struct_only

        # options
        self.backbone_only = backbone_only
        self.fix_channel_weights = fix_channel_weights
        self.pred_edge_dist = pred_edge_dist
        self.keep_memory = keep_memory
        if self.backbone_only:
            n_channel = 4
        self.cdr_type = cdr_type
        self.paratope = paratope
        self.sigma_min = sigma_min
        self.flow_weight = flow_weight
        self.sequence_flow_weight = sequence_flow_weight

        if time_embed_dim % 2 != 0:
            raise ValueError('time_embed_dim must be even')
        time_frequencies = 2.0 ** torch.arange(time_embed_dim // 2, dtype=torch.float)
        self.register_buffer('time_frequencies', time_frequencies)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, embed_size),
            nn.SiLU(),
            nn.Linear(embed_size, embed_size)
        )

        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            relative_position=relative_position,
            edge_constructor=GMEdgeConstructor,
            fix_atom_weights=fix_channel_weights,
            backbone_only=backbone_only
        )
        self.protein_feature = ProteinFeature(backbone_only=backbone_only)
        if keep_memory:
            self.memory_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, embed_size)
            )
        if self.pred_edge_dist:  # use predicted dist for KNN-graph at the interface
            if self.keep_memory:  # this ffn acts on the memory
                self.edge_H_ffn = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
            self.edge_dist_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(2 * hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
            # this GNN encodes the initial hidden states for initial edge distance prediction
            self.init_gnn = AMEGNN(
                embed_size, hidden_size, hidden_size, n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=0, n_layers=n_layers, residual=True,
                dropout=dropout, dense=False)
        if not struct_only:
            self.ffn_residue = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.num_classes)
            )
            # The continuous sequence state contains the 20 amino-acid
            # categories and the special states up to and including [MASK].
            self.sequence_state_dim = self.mask_id + 1
            self.ffn_sequence_velocity = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.sequence_state_dim)
            )
        else:
            self.prmsd_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        self.gnn = AMEncoder(
            embed_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=0, num_verts=num_verts, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False)
        
        self.normalizer = SeperatedCoordNormalizer()

        # training related cache
        self.batch_constants = {}

        # self.timing_stats = {
        #     'surface_processing': 0.0,
        #     'sme_encoding': 0.0,
        #     'count': 0
        # }


    def init_mask(self, X, S, cmask, smask, template):
        if not self.struct_only:
            S[smask] = self.mask_id
        X[cmask] = template
        return X, S
    
    def replace_pep(self, X, S, paratope_mask, X_pep, S_pep):
        if X_pep.abs().sum() == 0:
            return X, S
        pep_seq = getattr(self, 'pep_seq', True)
        pep_struct = getattr(self, 'pep_struct', True)

        if self.pep_seq:
            S[paratope_mask] = S_pep
        if self.pep_struct:
            X[paratope_mask] = X_pep
        return X, S
    
    def align_epi_ab(self, local_inter_edges, local_is_ab) : 
        aligned = torch.zeros_like(local_inter_edges)
        try : 
            for i in range(local_inter_edges.shape[1]) : 
                if local_is_ab[local_inter_edges[0][i]] == False and local_is_ab[local_inter_edges[1][i]] == True : 
                    aligned[:, i] = local_inter_edges[:, i]
                elif local_is_ab[local_inter_edges[0][i]] == True and local_is_ab[local_inter_edges[1][i]] == False : 
                    aligned[0, i] = local_inter_edges[1][i]
                    aligned[1, i] = local_inter_edges[0][i]
        except Exception as e : 
            print(e)
        
        epi_index = torch.nonzero(~local_is_ab).squeeze()
        return aligned, epi_index
    
    def optimal_alignment(self, X0, target_X):
        """
        计算X0到target_X的最优旋转和排序
        Args:
            X0: [N, n_channel, 3] 初始构象
            target_X: [N, n_channel, 3] 目标构象
        Returns:
            R: [3, 3] 最优旋转矩阵
            perm: [N] 最优排序
            X0_aligned: [N, n_channel, 3] 经过旋转和排序后的X0
        """
        from scipy.optimize import linear_sum_assignment
        # Align the backbone atoms first.  The subsequent assignment is at
        # residue level so atom identities within a residue are preserved.
        X0_flat = X0[:, :4].reshape(-1, 3)
        target_X_flat = target_X[:, :4].reshape(-1, 3)
        _, R, t = kabsch_torch(X0_flat, target_X_flat)
        X0_rotated = torch.matmul(X0, R.T) + t
        
        # Match residues using their C-alpha positions.  source_for_target[j]
        # gives the source residue assigned to target residue j.
        cost_matrix = torch.cdist(X0_rotated[:, 1], target_X[:, 1])
        source, target = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        source_for_target = torch.empty(X0.shape[0], dtype=torch.long, device=X0.device)
        source_for_target[torch.as_tensor(target, device=X0.device)] = torch.as_tensor(
            source, device=X0.device)
        X0_aligned = X0_rotated[source_for_target]
        
        return R, source_for_target, X0_aligned

    @torch.no_grad()
    def align_interface_batch(self, X0, S0, target_X, interface_batch_id):
        """Apply OT alignment independently to every complex in a batch."""
        aligned_X = torch.empty_like(X0)
        aligned_S = torch.empty_like(S0)
        batch_size = int(interface_batch_id.max().item()) + 1
        for batch_idx in range(batch_size):
            mask = interface_batch_id == batch_idx
            _, source_for_target, batch_X = self.optimal_alignment(
                X0[mask], target_X[mask])
            aligned_X[mask] = batch_X
            aligned_S[mask] = S0[mask][source_for_target]
        return aligned_X, aligned_S

    def time_embedding(self, flow_t, batch_id, dtype):
        """Return a Fourier time embedding for every residue node."""
        if not torch.is_tensor(flow_t):
            flow_t = torch.tensor(flow_t, device=batch_id.device, dtype=dtype)
        flow_t = flow_t.to(device=batch_id.device, dtype=dtype)
        if flow_t.ndim == 0 or flow_t.numel() == 1:
            node_t = flow_t.reshape(1).expand(batch_id.shape[0])
        elif flow_t.numel() == int(batch_id.max().item()) + 1:
            node_t = flow_t.reshape(-1)[batch_id]
        elif flow_t.numel() == batch_id.shape[0]:
            node_t = flow_t.reshape(-1)
        else:
            raise ValueError('flow_t must be scalar, per-complex, or per-node')

        angles = 2 * math.pi * node_t.unsqueeze(-1) * self.time_frequencies.to(dtype)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.time_mlp(embedding)
        

    def message_passing(self, X, S, residue_pos, interface_X, surf, paratope_mask, batch_id, t, memory_H=None, smooth_prob=None, smooth_mask=None):
        # embeddings, hidden state, (internal edges, external edges), (A : c * d, w : c * 1)
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(X, S, batch_id, self.k_neighbors, residue_pos, smooth_prob=smooth_prob, smooth_mask=smooth_mask)
        H_0 = H_0 + self.time_embedding(t, batch_id, H_0.dtype)

        if not self.keep_memory:
            memory_H = None

        if memory_H is not None:
            H_0 = H_0 + self.memory_ffn(memory_H)

        if self.pred_edge_dist:
            if memory_H is not None:
                edge_H = self.edge_H_ffn(memory_H)
            else:
                # replace the MLP with gnn for initial edge distance prediction
                edge_H, dumb_X = self.init_gnn(H_0, X, ctx_edges,
                                       channel_attr=atom_embeddings,
                                       channel_weights=atom_weights)
                X = X + dumb_X * 0  # to cheat the autograd check

        # update coordination of the global node
        X = self.aa_feature.update_global_coordinates(X, S)

        # prepare local complex
        local_mask = self.batch_constants['local_mask']
        local_is_ab = self.batch_constants['local_is_ab']
        local_batch_id = self.batch_constants['local_batch_id']
        local_X = X[local_mask].clone()
        # prepare local complex edges
        local_ctx_edges = self.batch_constants['local_ctx_edges']  # [2, Ec]
        local_inter_edges = self.batch_constants['local_inter_edges']  # [2, Ei]
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        offsets, max_n, gni2lni = self.batch_constants['local_edge_infos']
        # for context edges, use edges in the native paratope
        local_ctx_edges = _knn_edges(
            local_X, atom_pos, local_ctx_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni))
        # for interative edges, use edges derived from the predicted distance
        local_X[local_is_ab] = interface_X
        if self.pred_edge_dist:
            local_H = edge_H[local_mask]
            src_H, dst_H = local_H[local_inter_edges[0]], local_H[local_inter_edges[1]]
            p_edge_dist = self.edge_dist_ffn(torch.cat([src_H, dst_H], dim=-1)) +\
                          self.edge_dist_ffn(torch.cat([dst_H, src_H], dim=-1))  # perm-invariant
            p_edge_dist = p_edge_dist.squeeze()
        else:
            p_edge_dist = None
        local_inter_edges = _knn_edges(
            local_X, atom_pos, local_inter_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni), given_dist=p_edge_dist)
        local_edges = torch.cat([local_ctx_edges, local_inter_edges], dim=1)
        
        #prepare surface
        # surf_start = time.time()
        aligned_local_inter_edges, epi_index = self.align_epi_ab(local_inter_edges, local_is_ab)
        # self.timing_stats['surface_processing'] += time.time() - surf_start

        # message passing
        # sme_start = time.time()
        H, pred_X, pred_local_X = self.gnn(H_0, X, ctx_edges,
                                           local_mask, local_X, surf, local_edges,
                                           paratope_mask, local_is_ab,
                                           aligned_local_inter_edges, epi_index,
                                           channel_attr=atom_embeddings,
                                           channel_weights=atom_weights)
        # self.timing_stats['sme_encoding'] += time.time() - sme_start

        interface_X = pred_local_X[local_is_ab]
        pred_logits = None if self.struct_only else self.ffn_residue(H)
        pred_sequence_velocity = None if self.struct_only else self.ffn_sequence_velocity(H)

        return pred_logits, pred_sequence_velocity, pred_X, interface_X, H, p_edge_dist  # [N, num_classes], [N, sequence_state_dim], [N, n_channel, 3], [Ncdr, n_channel, 3], [N, hidden_size]
    
    @torch.no_grad()
    def init_interface(self, X, S, paratope_mask, batch_id, init_noise=None):
        ag_centers = X[S == self.aa_feature.boa_idx][:, 0]  # [bs, 3]
        init_local_X = torch.zeros_like(X[paratope_mask])
        init_local_X = init_local_X + ag_centers[batch_id[paratope_mask]].unsqueeze(1)
        noise = torch.randn_like(init_local_X) if init_noise is None else init_noise
        ca_noise = noise[:, 1]
        noise = noise / 10  + ca_noise.unsqueeze(1) # scale other atoms
        noise[:, 1] = ca_noise
        init_local_X = init_local_X + noise

        init_local_S = torch.randint(0, self.num_classes, 
                                   (paratope_mask.sum(),), 
                                   device=X.device,
                                   dtype=torch.long)
        return init_local_X, init_local_S

    @torch.no_grad()
    def _prepare_batch_constants(self, S, paratope_mask, lengths):
        # generate batch id
        batch_id = torch.zeros_like(S)  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        self.batch_constants['batch_id'] = batch_id
        self.batch_constants['batch_size'] = torch.max(batch_id) + 1

        segment_ids = self.aa_feature._construct_segment_ids(S)
        self.batch_constants['segment_ids'] = segment_ids

        # interface relatd
        is_ag = segment_ids == self.aa_feature.ag_seg_id
        not_ag_global = S != self.aa_feature.boa_idx
        local_mask = torch.logical_or(
            paratope_mask, torch.logical_and(is_ag, not_ag_global)
        )
        local_segment_ids = segment_ids[local_mask]
        local_is_ab = local_segment_ids != self.aa_feature.ag_seg_id
        local_batch_id = batch_id[local_mask]
        self.batch_constants['is_ag'] = is_ag
        self.batch_constants['local_mask'] = local_mask
        self.batch_constants['local_is_ab'] = local_is_ab
        self.batch_constants['local_batch_id'] = local_batch_id
        self.batch_constants['local_segment_ids'] = local_segment_ids
        # interface local edges
        (row, col), (offsets, max_n, gni2lni) = self.aa_feature.edge_constructor.get_batch_edges(local_batch_id)
        row_segment_ids, col_segment_ids = local_segment_ids[row], local_segment_ids[col]
        is_ctx = row_segment_ids == col_segment_ids
        is_inter = torch.logical_not(is_ctx)

        self.batch_constants['local_ctx_edges'] = torch.stack([row[is_ctx], col[is_ctx]])  # [2, Ec]
        self.batch_constants['local_inter_edges'] = torch.stack([row[is_inter], col[is_inter]])  # [2, Ei]
        self.batch_constants['local_edge_infos'] = (offsets, max_n, gni2lni)

        interface_batch_id = batch_id[paratope_mask]
        self.batch_constants['interface_batch_id'] = interface_batch_id
    
    def _clean_batch_constants(self):
        self.batch_constants = {}

    @torch.no_grad()
    def _get_inter_edge_dist(self, X, S):
        local_mask = self.batch_constants['local_mask']
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        src_dst = self.batch_constants['local_inter_edges'].T
        dist = X[local_mask][src_dst]  # [Ef, 2, n_channel, 3]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
        dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
        pos_pad = atom_pos[src_dst] == self.aa_feature.atom_pos_pad_idx # [Ef, 2, n_channel]
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
        dist = dist + pos_pad * 1e10  # [Ef, n_channel, n_channel]
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
        return dist
        is_binding = dist <= self.bind_dist_cutoff
        return is_binding
    
    def _forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                 surface, residue_pos, template, lengths, init_noise=None,
                 flow_X=None, flow_S_probs=None, flow_t=None):
        
        batch_id = self.batch_constants['batch_id']
        interface_batch_id = self.batch_constants['interface_batch_id']

        # The full-antibody tensors encode the conditional context.  The
        # local flow state is supplied independently through flow_X and
        # flow_S_probs so it cannot be overwritten by context initialization.
        X, S = X.clone(), S.clone()

        # mask sequence and initialize coordinates with template
        X, S = self.init_mask(X, S, cmask, smask, template)
        
        # replace paratope with peptide
        X, S = self.replace_pep(X, S, paratope_mask, X_pep, S_pep)

        # normalize
        X = self.normalizer.centering(X, S, batch_id, self.aa_feature)
        X = self.normalizer.normalize(X)
        surface = self.normalizer.normalize(surface)

        # update center
        X = self.aa_feature.update_global_coordinates(X, S)

        if flow_X is None:
            interface_X, _ = self.init_interface(
                X, S, paratope_mask, batch_id, init_noise)
        else:
            # flow_X is expressed in the original Cartesian frame.  The local
            # shadow interface uses the antigen-centred normalized frame.
            interface_X = flow_X - self.normalizer.ag_centers[
                interface_batch_id].unsqueeze(1)
            interface_X = self.normalizer.normalize(interface_X)

        if flow_t is None:
            flow_t = torch.zeros(
                int(self.batch_constants['batch_size'].item()),
                device=X.device, dtype=X.dtype)

        # sequence and structure loss
        r_pred_S_logits, r_pred_S_velocity = [], []
        pred_S_dist = flow_S_probs
        smooth_mask = paratope_mask if flow_S_probs is not None else smask
        r_interface_X = [interface_X.clone()]  # init
        r_edge_dist = []
        memory_H = None
        # message passing
        for round_idx in range(self.round):
            pred_S_logits, pred_S_velocity, pred_X, interface_X, H, edge_dist = self.message_passing(
                X, S, residue_pos, interface_X, surface, paratope_mask,
                batch_id, flow_t, memory_H, pred_S_dist, smooth_mask)
            memory_H = H
            r_interface_X.append(interface_X.clone())
            r_pred_S_logits.append((pred_S_logits, smask))
            r_pred_S_velocity.append(pred_S_velocity)
            r_edge_dist.append(edge_dist)
            # 1. update X
            X = X.clone()
            X[cmask] = pred_X[cmask]
            X = self.aa_feature.update_global_coordinates(X, S)

            if not self.struct_only:
                # 2. update S
                S = S.clone()
                if round_idx == self.round - 1:
                    S[smask] = torch.argmax(pred_S_logits[smask], dim=-1)
                else:
                    pred_S_dist = torch.softmax(pred_S_logits[smask], dim=-1)
                    smooth_mask = smask

        if self.struct_only:
            # predicted rmsd
            prmsd = self.prmsd_ffn(H[cmask]).squeeze()  # [N_ab]
        else:
            prmsd = None

        # uncentering and unnormalize
        pred_X = self.normalizer.unnormalize(pred_X)
        pred_X = self.normalizer.uncentering(pred_X, batch_id)
        for i, interface_X in enumerate(r_interface_X):
            interface_X = self.normalizer.unnormalize(interface_X)
            interface_X = self.normalizer.uncentering(interface_X, interface_batch_id, _type=4)
            r_interface_X[i] = interface_X
        self.normalizer.clear_cache()


        return (H, S, r_pred_S_logits, r_pred_S_velocity, pred_X,
                r_interface_X, r_edge_dist, prmsd)

    def forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, xloss_mask, context_ratio=0):
        '''
        :param X: [N, n_channel, 3], Cartesian coordinates
        :param context_ratio: float, rate of context provided in masked sequence, should be [0, 1) and anneal to 0 in training
        '''
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
            xloss_mask = xloss_mask[:, :4]
        # clone ground truth coordinates, sequence
        true_X, true_S = X.clone(), S.clone()

        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)
        batch_id = self.batch_constants['batch_id']
        interface_batch_id = self.batch_constants['interface_batch_id']
        batch_size = int(self.batch_constants['batch_size'].item())

        # provide some ground truth for annealing sequence training
        if context_ratio > 0:
            not_ctx_mask = torch.rand_like(smask, dtype=torch.float) >= context_ratio
            smask = torch.logical_and(smask, not_ctx_mask)
        
        gt_interface_X, gt_interface_S = true_X[paratope_mask], true_S[paratope_mask]
        interface_X, interface_S = self.init_interface(X, S, paratope_mask, batch_id)
        interface_X_aligned, interface_S_aligned = self.align_interface_batch(
            interface_X, interface_S, gt_interface_X, interface_batch_id)

        # Sample one flow time per complex and construct the OT conditional
        # path from Equation (5) of the original method.
        flow_t = torch.rand(batch_size, device=X.device, dtype=X.dtype)
        interface_t = flow_t[interface_batch_id].view(-1, 1, 1)
        sigma_t = 1.0 - (1.0 - self.sigma_min) * interface_t
        Xt = sigma_t * interface_X_aligned + interface_t * gt_interface_X

        if self.struct_only:
            St = None
            target_sequence_velocity = None
        else:
            q0 = F.one_hot(
                torch.full_like(interface_S_aligned, self.mask_id),
                num_classes=self.sequence_state_dim).to(X.dtype)
            q1 = F.one_hot(
                gt_interface_S, num_classes=self.sequence_state_dim).to(X.dtype)
            sequence_t = flow_t[interface_batch_id].unsqueeze(-1)
            St = (1.0 - sequence_t) * q0 + sequence_t * q1
            target_sequence_velocity = q1 - q0

        # get results
        (H, pred_S, r_pred_S_logits, r_pred_S_velocity, pred_X,
         r_interface_X, r_edge_dist, prmsd) = self._forward(
            X, S, cmask, smask, paratope_mask, X_pep, S_pep,
            surface, residue_pos, template, lengths,
            flow_X=Xt, flow_S_probs=St, flow_t=flow_t)

        # sequence negtive log likelihood
        snll, total = 0, 0
        if not self.struct_only:
            for logits, mask in r_pred_S_logits:
                snll = snll + F.cross_entropy(logits[mask], true_S[mask], reduction='sum')
                total = total + mask.sum()
            snll = snll / total.clamp_min(1)

        # structure loss
        struct_loss, struct_loss_details, bb_rmsd, ops = self.protein_feature.structure_loss(pred_X, true_X, true_S, cmask, batch_id, xloss_mask, self.aa_feature)

        # docking loss
        
        # 1. interface loss (shadow paratope)
        interface_atom_pos = self.aa_feature._construct_atom_pos(true_S[paratope_mask])
        interface_atom_mask = interface_atom_pos != self.aa_feature.atom_pos_pad_idx
        predicted_velocity = r_interface_X[-1] - r_interface_X[0]
        predicted_endpoint = ((1.0 - self.sigma_min) * r_interface_X[0]
                              + sigma_t * predicted_velocity)
        interface_loss = F.smooth_l1_loss(
            predicted_endpoint[interface_atom_mask],
            gt_interface_X[interface_atom_mask])

        # Conditional flow-matching loss.  The network displacement is the
        # predicted instantaneous velocity at the sampled intermediate state.
        target_velocity = (gt_interface_X
                           - (1.0 - self.sigma_min) * interface_X_aligned)
        coordinate_flow_loss = F.smooth_l1_loss(
            predicted_velocity[interface_atom_mask],
            target_velocity[interface_atom_mask])
        if self.struct_only:
            sequence_flow_loss = coordinate_flow_loss.new_zeros(())
        else:
            predicted_sequence_velocity = r_pred_S_velocity[-1][paratope_mask]
            sequence_flow_loss = F.smooth_l1_loss(
                predicted_sequence_velocity, target_sequence_velocity)
        flow_loss = (self.flow_weight * coordinate_flow_loss
                     + self.sequence_flow_weight * sequence_flow_loss)


        # 2. edge dist loss
        if self.pred_edge_dist:
            gt_edge_dist = self._get_inter_edge_dist(self.normalizer.normalize(true_X), true_S)
            ed_loss, r_ed_losses = 0, []
            for edge_dist in r_edge_dist:
                r_ed_loss = F.smooth_l1_loss(edge_dist, gt_edge_dist)
                ed_loss = ed_loss + r_ed_loss
                r_ed_losses.append(r_ed_loss)
        else:
            r_ed_losses = [0 for _ in range(self.round)]
            ed_loss = 0
        dock_loss = interface_loss + ed_loss

        if self.struct_only:
            # predicted rmsd
            prmsd_loss = F.smooth_l1_loss(prmsd, bb_rmsd)
            pdev_loss = prmsd_loss
        else:
            pdev_loss, prmsd_loss = None, None

        # comprehensive loss
        loss = (snll + struct_loss + dock_loss + flow_loss
                + (0 if pdev_loss is None else pdev_loss))

        self._clean_batch_constants()

        # AAR
        with torch.no_grad():
            aa_hit = pred_S[smask] == true_S[smask]
            aar = aa_hit.long().sum() / max(aa_hit.shape[0], 1)

        return (loss, (snll, aar), (struct_loss, *struct_loss_details),
                (dock_loss, interface_loss, ed_loss, r_ed_losses),
                (pdev_loss, prmsd_loss),
                (flow_loss, coordinate_flow_loss, sequence_flow_loss))

    def sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, n_steps=10, init_noise=None, return_hidden=False):
        
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
        
        # self.timing_stats = {
        #     'surface_processing': 0.0,
        #     'sme_encoding': 0.0,
        #     'count': 0
        # }
        
        gen_X, gen_S = X.clone(), S.clone()
        
        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size = int(self.batch_constants['batch_size'].item())
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]
        interface_cmask = paratope_mask[cmask]

        if n_steps <= 0:
            raise ValueError('n_steps must be positive')

        interface_X, _ = self.init_interface(
            X, S, paratope_mask, batch_id, init_noise)
        dt = 1.0 / n_steps
        Xt = interface_X.clone()
        if self.struct_only:
            sequence_state = None
        else:
            sequence_state = F.one_hot(
                torch.full((Xt.shape[0],), self.mask_id,
                           device=X.device, dtype=torch.long),
                num_classes=self.sequence_state_dim).to(X.dtype)
        
        for i in range(n_steps):
            flow_t = torch.tensor(i * dt, device=X.device, dtype=X.dtype)

            (H, pred_S, r_pred_S_logits, r_pred_S_velocity, pred_X,
             r_interface_X, r_edge_dist, prmsd) = self._forward(
                X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                surface, residue_pos, template, lengths,
                flow_X=Xt, flow_S_probs=sequence_state, flow_t=flow_t)

            # Explicit Euler integration of the learned velocity field.
            coordinate_velocity = r_interface_X[-1] - r_interface_X[0]
            Xt = Xt + dt * coordinate_velocity
            if not self.struct_only:
                sequence_velocity = r_pred_S_velocity[-1][paratope_mask]
                sequence_state = sequence_state + dt * sequence_velocity
                # Numerical projection keeps the integrated state on the
                # probability simplex without discrete resampling at each step.
                sequence_state = sequence_state.clamp_min(0)
                sequence_state = sequence_state / sequence_state.sum(
                    dim=-1, keepdim=True).clamp_min(1e-8)

        # A final network evaluation propagates the transported local state to
        # the complete antibody.  It is not an additional ODE step.
        final_t = torch.tensor(1.0, device=X.device, dtype=X.dtype)
        (H, pred_S, r_pred_S_logits, r_pred_S_velocity, pred_X,
         r_interface_X, _, prmsd) = self._forward(
            X, S, cmask, smask, paratope_mask, X_pep, S_pep,
            surface, residue_pos, template, lengths,
            flow_X=Xt, flow_S_probs=sequence_state, flow_t=final_t)

        gen_X[cmask] = pred_X[cmask]
        if not self.struct_only:
            gen_S[smask] = pred_S[smask]
            amino_acid_probs = sequence_state[:, :self.num_classes]
            amino_acid_mass = amino_acid_probs.sum(dim=-1, keepdim=True)
            normalized_amino_acid_probs = amino_acid_probs / amino_acid_mass.clamp_min(1e-8)
            uniform_amino_acid_probs = torch.full_like(
                amino_acid_probs, 1.0 / self.num_classes)
            amino_acid_probs = torch.where(
                amino_acid_mass > 1e-8,
                normalized_amino_acid_probs,
                uniform_amino_acid_probs)
            sampled_interface_S = torch.multinomial(
                amino_acid_probs, num_samples=1).squeeze(-1)
            gen_S[paratope_mask] = sampled_interface_S

            S_logits = r_pred_S_logits[-1][0][smask]
            S_probs = torch.max(torch.softmax(S_logits, dim=-1), dim=-1)[0]
            metric = scatter_mean(-torch.log(S_probs.clamp_min(1e-8)), s_batch_id)
        else:
            metric = scatter_mean(
                prmsd[interface_cmask], interface_batch_id)

        # Rigidly place the generated antibody at the transported interface,
        # then retain the ODE state itself for the paratope coordinates.
        for batch_idx in range(batch_size):
            is_cur_graph = batch_id == batch_idx
            current_paratope = torch.logical_and(is_cur_graph, paratope_mask)
            generated_cdr = gen_X[current_paratope][:, :4]
            transported_cdr = Xt[interface_batch_id == batch_idx][:, :4]
            _, R, translation = kabsch_torch(
                generated_cdr.reshape(-1, 3), transported_cdr.reshape(-1, 3))
            is_cur_ab = is_cur_graph & is_ab
            gen_X[is_cur_ab] = torch.matmul(gen_X[is_cur_ab], R.T) + translation
        gen_X[paratope_mask] = Xt

        self._clean_batch_constants()

        # self.timing_stats['count'] += 1

        if return_hidden:
            return gen_X, gen_S, metric, H
        return gen_X, gen_S, metric
    
    def struct_sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, init_noise=None, return_hidden=False):
        
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
        gen_X, gen_S = X.clone(), S.clone()
        
        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size = int(self.batch_constants['batch_size'].item())
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]

        best_metric = torch.ones(batch_size, dtype=torch.float, device=X.device) * 1e10
        interface_cmask = paratope_mask[cmask]

        n_tries = 10 if self.struct_only else 1
        for i in range(n_tries):
        
            # generate
            (H, pred_S, r_pred_S_logits, _, pred_X,
             r_interface_X, _, prmsd) = self._forward(
                X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                surface, residue_pos, template, lengths, init_noise)

            # PPL or PRMSD
            if not self.struct_only:
                S_logits = r_pred_S_logits[-1][0][smask]
                S_probs = torch.max(torch.softmax(S_logits, dim=-1), dim=-1)[0]
                nlls = -torch.log(S_probs)
                metric = scatter_mean(nlls, s_batch_id)  # [batch_size]
            else:
                metric = scatter_mean(prmsd[interface_cmask], interface_batch_id)  # [batch_size]

            update = metric < best_metric
            cupdate = cmask & update[batch_id]
            supdate = smask & update[batch_id]
            # update metric history
            best_metric[update] = metric[update]

            # 1. set generated part
            gen_X[cupdate] = pred_X[cupdate]
            if not self.struct_only:
                gen_S[supdate] = pred_S[supdate]
        
            interface_X = r_interface_X[-1]
            # 2. align by cdr
            for i in range(batch_size):
                if not update[i]:
                    continue
                # 1. align CDRH3
                is_cur_graph = batch_id == i
                cdrh3_cur_graph = torch.logical_and(is_cur_graph, paratope_mask)
                ori_cdr = gen_X[cdrh3_cur_graph][:, :4]  # backbone
                pred_cdr = interface_X[interface_batch_id == i][:, :4]
                _, R, t = kabsch_torch(ori_cdr.reshape(-1, 3), pred_cdr.reshape(-1, 3))

                # 2. tranform antibody
                is_cur_ab = is_cur_graph & is_ab
                ab_X = torch.matmul(gen_X[is_cur_ab], R.T) + t
                gen_X[is_cur_ab] = ab_X

        self._clean_batch_constants()

        if return_hidden:
            return gen_X, gen_S, metric, H
        return gen_X, gen_S, metric

    def sample_many(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, n_samples=5, n_steps=20, return_hidden=False):
        """
        Generate multiple samples in a single call
        
        Args:
            X, S, cmask, smask, paratope_mask, residue_pos, template, lengths: 
                Same parameters as in sample() method
            n_samples: Number of samples to generate
            n_steps: Number of flow steps for each sample
            return_hidden: Whether to return hidden states
            
        Returns:
            list_gen_X: List of n_samples generated coordinates
            list_gen_S: List of n_samples generated sequences
            list_metrics: List of n_samples metrics
            list_H: (Optional) List of n_samples hidden states if return_hidden=True
        """
        list_gen_X = []
        list_gen_S = []
        list_metrics = []
        list_H = [] if return_hidden else None
        
        # Generate multiple samples with different random noise
        for i in range(n_samples):
            # Generate different noise for each sample
            if self.backbone_only:
                init_noise = torch.randn(paratope_mask.sum(), 4, 3, device=X.device)
            else:
                init_noise = torch.randn(paratope_mask.sum(), X.shape[1], 3, device=X.device)
                
            # Generate a sample
            if return_hidden:
                gen_X, gen_S, metric, H = self.sample(
                    X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, 
                    template, lengths, n_steps=n_steps, init_noise=init_noise, return_hidden=True
                )
                list_H.append(H)
            else:
                gen_X, gen_S, metric = self.sample(
                    X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, 
                    template, lengths, n_steps=n_steps, init_noise=init_noise
                )
            
            # Store results
            list_gen_X.append(gen_X)
            list_gen_S.append(gen_S)
            list_metrics.append(metric)
            
        if return_hidden:
            return list_gen_X, list_gen_S, list_metrics, list_H
        else:
            return list_gen_X, list_gen_S, list_metrics

isMEANModel = AbFlowModel
dyMEANModel = AbFlowModel
