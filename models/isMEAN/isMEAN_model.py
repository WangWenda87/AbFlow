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


class isMEANModel(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, num_classes, num_verts, 
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6,
                 n_layers=3, iter_round=3, dropout=0.1, 
                 pep_seq=True, pep_struct=True, struct_only=False,
                 backbone_only=False, fix_channel_weights=False, pred_edge_dist=True,
                 keep_memory=True, cdr_type='H3', paratope='H3', relative_position=False) -> None:
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

        self.timing_stats = {
            'surface_processing': 0.0,
            'sme_encoding': 0.0,
            'count': 0
        }


    def init_mask(self, X, S, cmask, smask, template):
        if not self.struct_only:
            S[smask] = self.mask_id
        X[cmask] = template
        return X, S
    
    def replace_pep(self, X, S, paratope_mask, X_pep, S_pep) :
        if self.pep_seq:
            S[paratope_mask] = S_pep
        if self.pep_struct : 
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
        # 1. 先计算最优旋转
        X0_flat = X0.reshape(-1, 3)
        target_X_flat = target_X.reshape(-1, 3)
        _, R, t = kabsch_torch(X0_flat, target_X_flat)
        X0_rotated = torch.matmul(X0, R.T) + t
        
        # 2. 计算最优排序 (使用匈牙利算法)
        cost_matrix = torch.cdist(X0_rotated.reshape(-1, 3), target_X.reshape(-1, 3))
        cost_matrix = cost_matrix.reshape(X0.shape[0], X0.shape[1], -1)  # [N, n_channel, N*n_channel]
        cost_matrix = cost_matrix.reshape(X0.shape[0]*X0.shape[1], -1)  # [N*n_channel, N*n_channel]
        
        # 使用匈牙利算法找最优匹配
        perm = linear_sum_assignment(cost_matrix.cpu().numpy())[1]
        perm = torch.from_numpy(perm).to(X0.device)
        
        # 应用旋转和排序
        X0_aligned = X0_rotated.reshape(-1, 3)[perm].reshape(X0.shape)
        
        return R, perm, X0_aligned
        

    def message_passing(self, X, S, residue_pos, interface_X, surf, paratope_mask, batch_id, t, memory_H=None, smooth_prob=None, smooth_mask=None):
        # embeddings, hidden state, (internal edges, external edges), (A : c * d, w : c * 1)
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(X, S, batch_id, self.k_neighbors, residue_pos, smooth_prob=smooth_prob, smooth_mask=smooth_mask)

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
        surf_start = time.time()
        aligned_local_inter_edges, epi_index = self.align_epi_ab(local_inter_edges, local_is_ab)
        self.timing_stats['surface_processing'] += time.time() - surf_start

        # message passing
        sme_start = time.time()
        H, pred_X, pred_local_X = self.gnn(H_0, X, ctx_edges,
                                           local_mask, local_X, surf, local_edges,
                                           paratope_mask, local_is_ab,
                                           aligned_local_inter_edges, epi_index,
                                           channel_attr=atom_embeddings,
                                           channel_weights=atom_weights)
        self.timing_stats['sme_encoding'] += time.time() - sme_start

        interface_X = pred_local_X[local_is_ab]
        pred_logits = None if self.struct_only else self.ffn_residue(H)

        return pred_logits, pred_X, interface_X, H, p_edge_dist  # [N, num_classes], [N, n_channel, 3], [Ncdr, n_channel, 3], [N, hidden_size]
    
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
    
    def _forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, init_noise=None):
        
        batch_id = self.batch_constants['batch_id']
        # print(batch_id[paratope_mask].shape, residue_pos[paratope_mask].shape)
        
        # print(X.shape, S.shape, cmask.shape, smask.shape, paratope_mask.shape, X_pep.shape, S_pep.shape, template.shape)

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

        # prepare initial interface
        interface_X, interface_S = self.init_interface(X, S, paratope_mask, batch_id, init_noise)
        # initial interface is replaced by peptide
        # interface_X = X[paratope_mask]

        # sequence and structure loss
        r_pred_S_logits, pred_S_dist, = [], None
        r_interface_X = [interface_X.clone()]  # init
        r_edge_dist = []
        memory_H = None
        # message passing
        for t in range(self.round):
            pred_S_logits, pred_X, interface_X, H, edge_dist = self.message_passing(X, S, residue_pos, interface_X, surface, paratope_mask, batch_id, t, memory_H, pred_S_dist, smask)
            memory_H = H
            r_interface_X.append(interface_X.clone())
            r_pred_S_logits.append((pred_S_logits, smask))
            r_edge_dist.append(edge_dist)
            # 1. update X
            X = X.clone()
            X[cmask] = pred_X[cmask]
            X = self.aa_feature.update_global_coordinates(X, S)

            if not self.struct_only:
                # 2. update S
                S = S.clone()
                if t == self.round - 1:
                    S[smask] = torch.argmax(pred_S_logits[smask], dim=-1)
                else:
                    pred_S_dist = torch.softmax(pred_S_logits[smask], dim=-1)

        interface_batch_id = self.batch_constants['interface_batch_id']

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


        return H, S, r_pred_S_logits, pred_X, r_interface_X,  r_edge_dist, prmsd

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

        # provide some ground truth for annealing sequence training
        if context_ratio > 0:
            not_ctx_mask = torch.rand_like(smask, dtype=torch.float) >= context_ratio
            smask = torch.logical_and(smask, not_ctx_mask)
        
        gt_interface_X, gt_interface_S = true_X[paratope_mask], true_S[paratope_mask]
        interface_X, interface_S = self.init_interface(X, S, paratope_mask, batch_id)
        R, perm, interface_X_aligned = self.optimal_alignment(interface_X, gt_interface_X)
        interface_S_aligned = interface_S[torch.unique(perm // interface_X.shape[1])]
        
        t = torch.rand(1, device=X.device)
        Xt = (1-t) * interface_X_aligned + t * gt_interface_X
        St = (1-t) * interface_S_aligned + t * gt_interface_S
        X[paratope_mask] = Xt.to(X.dtype)
        S[paratope_mask] = St.to(S.dtype)

        # get results
        H, pred_S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd = self._forward(X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths)

        # sequence negtive log likelihood
        snll, total = 0, 0
        if not self.struct_only:
            for logits, mask in r_pred_S_logits:
                snll = snll + F.cross_entropy(logits[mask], true_S[mask], reduction='sum')
                total = total + mask.sum()
            snll = snll / total

        # structure loss
        struct_loss, struct_loss_details, bb_rmsd, ops = self.protein_feature.structure_loss(pred_X, true_X, true_S, cmask, batch_id, xloss_mask, self.aa_feature)

        # docking loss
        
        # 1. interface loss (shadow paratope)
        interface_atom_pos = self.aa_feature._construct_atom_pos(true_S[paratope_mask])
        interface_atom_mask = interface_atom_pos != self.aa_feature.atom_pos_pad_idx
        interface_loss = F.smooth_l1_loss(
            r_interface_X[-1][interface_atom_mask],
            gt_interface_X[interface_atom_mask])

        # flow loss
        dX = r_interface_X[-1][interface_atom_mask] - interface_X_aligned[interface_atom_mask]
        dS = pred_S[paratope_mask]
        true_vx = gt_interface_X[interface_atom_mask] - interface_X_aligned[interface_atom_mask]
        true_vh = gt_interface_S.float() - interface_S_aligned.float()
        # flow_loss = F.smooth_l1_loss(dX, true_vx) + F.smooth_l1_loss(dS, true_vh)
        flow_loss = 0.01 * F.smooth_l1_loss(dX, true_vx)


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
        loss = snll + struct_loss + dock_loss + (0 if pdev_loss is None else pdev_loss)
        # loss = snll + struct_loss + dock_loss + flow_loss + (0 if pdev_loss is None else pdev_loss)

        self._clean_batch_constants()

        # AAR
        with torch.no_grad():
            aa_hit = pred_S[smask] == true_S[smask]
            aar = aa_hit.long().sum() / aa_hit.shape[0]

        return loss, (snll, aar), (struct_loss, *struct_loss_details), (dock_loss, interface_loss, ed_loss, r_ed_losses), (pdev_loss, prmsd_loss)

    def sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, n_steps=10, init_noise=None, return_hidden=False):
        
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
        
        self.timing_stats = {
            'surface_processing': 0.0,
            'sme_encoding': 0.0,
            'count': 0
        }
        
        gen_X, gen_S = X.clone(), S.clone()
        
        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]

        best_metric = torch.ones(batch_size, dtype=torch.float, device=X.device) * 1e10
        interface_cmask = paratope_mask[cmask]

        interface_X, interface_S = self.init_interface(X, S, paratope_mask, batch_id)
        dt = 1.0 / n_steps
        Xt = interface_X.clone()
        St = interface_S.clone()
        
        for i in range(n_steps):
            t = torch.tensor(i * dt, device=X.device)
            
            # 更新当前状态
            X_cur = X.clone()
            S_cur = S.clone()
            X_cur[paratope_mask] = Xt
            S_cur[paratope_mask] = St
            
            # 使用message passing获取速度场
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd = self._forward(
                X_cur, S_cur, cmask, smask, paratope_mask, 
                X_pep, S_pep, surface, residue_pos, template, lengths
            )

            # 计算速度场
            dX = r_interface_X[-1] - Xt
            if not self.struct_only:
                cur_logits = r_pred_S_logits[-1][0][paratope_mask]
                # 1. 数值稳定性处理
                cur_logits = cur_logits - cur_logits.max(dim=-1, keepdim=True)[0]
                # 2. 计算当前概率
                cur_probs = F.softmax(cur_logits, dim=-1)
                # 3. 计算序列的速度场
                dS = cur_probs - F.one_hot(St, num_classes=self.num_classes).float()
                
            # Euler步进
            Xt = Xt + dX * dt
            if not self.struct_only:
                next_probs = F.one_hot(St, num_classes=self.num_classes).float() + dS * dt
                # 确保数值稳定性
                next_probs = next_probs.clamp(min=1e-6)
                next_probs = next_probs / next_probs.sum(dim=-1, keepdim=True)
                # 从更新后的概率分布中采样
                St = torch.multinomial(next_probs, num_samples=1).squeeze(-1)
        
        X[paratope_mask] = Xt
        S[paratope_mask] = St
            
        n_tries = 10 if self.struct_only else 1
        for i in range(n_tries):
        
            # generate
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, _, prmsd = self._forward(X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, init_noise)

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

        self.timing_stats['count'] += 1
        
        # 打印timing信息
        print(f"\n{'='*50}")
        print(f"Timing Statistics (n_steps={n_steps}):")
        print(f"{'='*50}")
        print(f"Surface Processing: {self.timing_stats['surface_processing']:.4f}s")
        print(f"SME Encoding:       {self.timing_stats['sme_encoding']:.4f}s")
        total = self.timing_stats['surface_processing'] + self.timing_stats['sme_encoding']
        print(f"{'-'*50}")
        print(f"Total (Surf+SME):   {total:.4f}s")
        if total > 0:
            surf_pct = self.timing_stats['surface_processing'] / total * 100
            sme_pct = self.timing_stats['sme_encoding'] / total * 100
            print(f"Surface %: {surf_pct:.2f}%")
            print(f"SME %:     {sme_pct:.2f}%")
        print(f"{'='*50}\n")

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

# if __name__ == '__main__':
#     torch.random.manual_seed(0)
#     # equivariance test
#     embed_size, hidden_size = 64, 128
#     n_channel, d = 14, 3
#     n_aa_type = 20
#     scale = 10
#     dtype = torch.float
#     device = torch.device('cuda:0')
#     model = isMEANModel(embed_size, hidden_size, n_channel,
#                    VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
#                    k_neighbors=9, bind_dist_cutoff=6.6, n_layers=3)
#     model.to(device)
#     model.eval()

#     ag_len, h_len, l_len = 48, 120, 110
#     center_x = torch.randn(3, 1, n_channel, d, device=device, dtype=dtype) * scale
#     ag_X = torch.randn(ag_len, n_channel, d, device=device, dtype=dtype) * scale
#     h_X = torch.randn(h_len, n_channel, d, device=device, dtype=dtype) * scale
#     l_X = torch.randn(l_len, n_channel, d, device=device, dtype=dtype) * scale

#     X = torch.cat([center_x[0], ag_X, center_x[1], h_X, center_x[2], l_X], dim=0)
#     S = torch.cat([torch.tensor([model.aa_feature.boa_idx], device=device),
#                    torch.randint(low=0, high=20, size=(ag_len,), device=device),
#                    torch.tensor([model.aa_feature.boh_idx], device=device),
#                    torch.randint(low=0, high=20, size=(h_len,), device=device),
#                    torch.tensor([model.aa_feature.bol_idx], device=device),
#                    torch.randint(low=0, high=20, size=(l_len,), device=device)], dim=0)
#     cmask = torch.tensor([0] + [0 for _ in range(ag_len)] + [0] + [1 for _ in range(h_len)] + [0] + [1 for _ in range(l_len)], device=device).bool()
#     smask = torch.zeros_like(cmask)
#     smask[ag_len+10:ag_len+20] = 1
#     paratope_mask = smask
#     residue_pos = torch.tensor([0] + [0 for _ in range(ag_len)] + [0] + list(range(1, h_len + 1)) + [0] + list(range(1, l_len + 1)), device=device)
#     template = torch.randn_like(X[cmask]) * scale
#     lengths = torch.tensor([ag_len + h_len + l_len + 3], device=device)
#     noise = torch.randn_like(X[paratope_mask])

#     torch.random.manual_seed(1)
#     gen_X, _, _ = model.sample(X, S, cmask, smask, smask, residue_pos, template, lengths, init_noise=noise)
#     # tmpx1 = model.tmpx

#     # random rotaion matrix
#     U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
#     if torch.linalg.det(U) * torch.linalg.det(V) < 0:
#         U[:, -1] = -U[:, -1]
#     Q_ag, t_ag = U.mm(V), torch.randn(3, device=device)

#     U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
#     if torch.linalg.det(U) * torch.linalg.det(V) < 0:
#         U[:, -1] = -U[:, -1]
#     Q_ab, t_ab = U.mm(V), torch.randn(3, device=device)

#     ag_X = torch.matmul(ag_X, Q_ag) + t_ag
#     X = torch.cat([center_x[0], ag_X, center_x[1], h_X, center_x[2], l_X], dim=0)
#     template = torch.matmul(template, Q_ab) + t_ab

#     # this is f([Q1x1+t1, Q2x2+t2])
#     torch.random.manual_seed(1)
#     noise = torch.matmul(noise, Q_ag)
#     gen_op_X, _, _ = model.sample(X, S, cmask, smask, smask, residue_pos, template, lengths, init_noise=noise)
#     # tmpx2 = model.tmpx
#     # gt_tmpx = torch.matmul(tmpx1, Q_ag) + t_ag
#     # error = torch.abs(gt_tmpx[:, :4] - tmpx2[:, :4]).sum(-1).flatten().mean()
#     # # error = torch.abs(tmpx2 - tmpx1).sum(-1).flatten().mean()
#     # print(error.item())
#     # assert error < 1e-3
#     # print('independent equivariance check passed')


#     gt_op_X = torch.matmul(gen_X, Q_ag) + t_ag

#     error = torch.abs(gt_op_X[cmask][:, :4] - gen_op_X[cmask][:, :4]).sum(-1).flatten().mean()
#     print(error.item())
#     assert error < 1e-3
#     print('independent equivariance check passed')
