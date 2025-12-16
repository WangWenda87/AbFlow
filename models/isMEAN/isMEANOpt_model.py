#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_std

from data.pdb_utils import VOCAB
from utils.nn_utils import SeparatedAminoAcidFeature, ProteinFeature
from utils.nn_utils import EdgeConstructor

from ..modules.am_egnn import AMEGNN
from ..modules.am_enc import AMEncoder

'''
Masked 1D & 3D language model
Add noise to ground truth 3D coordination
Add mask to 1D sequence
'''
class isMEANOptModel(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, num_classes,
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6,
                 num_verts=50, n_layers=3, iter_round=3, dropout=0.1, struct_only=False,
                 fix_atom_weights=False, cdr_type='H3', relative_position=False) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.num_classes = num_classes
        self.bind_dist_cutoff = bind_dist_cutoff
        self.k_neighbors = k_neighbors
        self.round = iter_round
        self.cdr_type = cdr_type  # only to indicate the usage of the model

        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            relative_position=relative_position,
            edge_constructor=EdgeConstructor,
            fix_atom_weights=fix_atom_weights)
        self.protein_feature = ProteinFeature()
        
        self.memory_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, embed_size)
        )
        self.struct_only = struct_only
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
        self.gnn = AMEGNN(
            embed_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=0, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False)
        # self.gnn_s = AMEncoder(embed_size, hidden_size, hidden_size, n_channel,
        #     channel_nf=atom_embed_size, radial_nf=hidden_size,
        #     in_edge_nf=0, num_verts=num_verts, n_layers=n_layers, residual=True,
        #     dropout=dropout, dense=False)
        
        # training related cache
        self.start_seq_training = False
        self.batch_constants = {}

    def init_mask(self, X, S, cmask, smask, init_noise):
        if not self.struct_only:
            S[smask] = self.mask_id
        coords = X[cmask]
        noise = torch.randn_like(coords) if init_noise is None else init_noise
        X = X.clone()
        X[cmask] = coords + noise
        return X, S
    
    def replace_pep(self, X, S, paratope_mask, X_pep, S_pep) :
        # if S_pep :
        #     S[paratope_mask] = S_pep
        # if X_pep : 
        #     X[paratope_mask] = X_pep
        return X, S

    def message_passing(self, X, S, residue_pos, batch_id, t, memory_H=None, smooth_prob=None, smooth_mask=None):
        # embeddings
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(X, S, batch_id, self.k_neighbors, residue_pos, smooth_prob=smooth_prob, smooth_mask=smooth_mask)
        inter_edges = self._get_binding_edges(X, S, inter_edges)
        edges = torch.cat([ctx_edges, inter_edges], dim=1)

        if memory_H is not None:
            H_0 = H_0 + self.memory_ffn(memory_H)

        # update coordination of the global node
        X = self.aa_feature.update_global_coordinates(X, S)

        H, pred_X = self.gnn(H_0, X, edges,
                             channel_attr=atom_embeddings,
                             channel_weights=atom_weights)


        pred_logits = None if self.struct_only else self.ffn_residue(H)

        return pred_logits, pred_X, H # [N, num_classes], [N, n_channel, 3], [N, hidden_size]
    
    @torch.no_grad()
    def _prepare_batch_constants(self, S, lengths):
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
        self.batch_constants['is_ag'] = is_ag
    
    @torch.no_grad()
    def _get_binding_edges(self, X, S, inter_edges):
        atom_pos = self.aa_feature._construct_atom_pos(S)
        src_dst = inter_edges.T
        dist = X[src_dst]  # [Ef, 2, n_channel, 3]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
        dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
        pos_pad = atom_pos[src_dst] == self.aa_feature.atom_pos_pad_idx # [Ef, 2, n_channel]
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
        dist = dist + pos_pad * 1e10  # [Ef, n_channel, n_channel]
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
        is_binding = dist <= self.bind_dist_cutoff
        return src_dst[is_binding].T
    
    @torch.no_grad()
    def init_interface(self, X, S, paratope_mask, batch_id, init_noise=None):
        ag_centers = X[S == self.aa_feature.boa_idx][:, 0]  # [bs, 3]
        init_local_X = torch.zeros_like(X[paratope_mask])
<<<<<<< Updated upstream
=======
        
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
        # print(paratope_mask)
>>>>>>> Stashed changes
        return init_local_X, init_local_S
    
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
        from evaluation.rmsd import kabsch_torch
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

    def _clean_batch_constants(self):
        self.batch_constants = {}

    def _forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, residue_pos, init_noise=None):
        batch_id = self.batch_constants['batch_id']

        # mask sequence and add noise to ground truth coordinates
        X, S = self.init_mask(X, S, cmask, smask, init_noise)
        
        # replace pepglad
        # print(len(S[smask]), len(S_pep))
        X, S = self.replace_pep(X, S, paratope_mask, X_pep, S_pep)

        # update center
        X = self.aa_feature.update_global_coordinates(X, S)

        # sequence and structure loss
        r_pred_S_logits, pred_S_dist = [], None
        memory_H = None
        # message passing
        for t in range(self.round):
            pred_S_logits, pred_X, H = self.message_passing(X, S, residue_pos, batch_id, t, memory_H, pred_S_dist, smask)
            r_pred_S_logits.append((pred_S_logits, smask))
            memory_H = H
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

        if self.struct_only:
            # predicted rmsd
            prmsd = self.prmsd_ffn(H[cmask]).squeeze()  # [N_ab]
        else:
            prmsd = None

        return H, S, r_pred_S_logits, pred_X, prmsd

    def forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, lengths, xloss_mask, context_ratio=0, seq_alpha=1):
        '''
        :param bind_ag: [N_bind], node idx of binding residues in antigen
        :param bind_ab: [N_bind], node idx of binding residues in antibody
        :param bind_ag_X: [N_bind, 3], coordinations of the midpoint of binding pairs relative to ag
        :param bind_ab_X: [N_bind, 3], coordinations of the midpoint of binding pairs relative to ab
        :param context_ratio: float, rate of context provided in masked sequence, should be [0, 1) and anneal to 0 in training
        :param seq_alpha: float, weight of SNLL, linearly increase from 0 to 1 at warmup phase
        '''
        # clone ground truth coordinates, sequence
        true_X, true_S = X.clone(), S.clone()

        # prepare constants
        self._prepare_batch_constants(S, lengths)
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
        H, pred_S, r_pred_S_logits, pred_X, prmsd = self._forward(X, S, cmask, smask, paratope_mask, X_pep, S_pep, residue_pos)

        # sequence negtive log likelihood
        snll, total = 0, 0
        if not self.struct_only:
            for logits, mask in r_pred_S_logits:
                snll = snll + F.cross_entropy(logits[mask], true_S[mask], reduction='sum')
                total = total + mask.sum()
            snll = snll / total

        # coordination loss
        struct_loss, struct_loss_details, bb_rmsd, _ = self.protein_feature.structure_loss(pred_X, true_X, true_S, cmask, batch_id, xloss_mask, self.aa_feature)

        if self.struct_only:
            # predicted rmsd
            prmsd_loss = F.smooth_l1_loss(prmsd, bb_rmsd)
            pdev_loss = prmsd_loss# + prmsd_i_loss
        else:
            pdev_loss, prmsd_loss = None, None

        # comprehensive loss
        loss = seq_alpha * snll + struct_loss + (0 if pdev_loss is None else pdev_loss)

        self._clean_batch_constants()

        # AAR
        with torch.no_grad():
            aa_hit = pred_S[smask] == true_S[smask]
            aar = aa_hit.long().sum() / aa_hit.shape[0]

        return loss, (snll, aar), (struct_loss, *struct_loss_details), (pdev_loss, prmsd_loss)

    def sample(self, X, S, cmask, smask, paratope_mask, residue_pos, lengths, X_pep=False, S_pep=False, return_hidden=False, init_noise=None, n_steps=10):
        gen_X, gen_S = X.clone(), S.clone()
        
        # prepare constants
        self._prepare_batch_constants(S, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        s_batch_id = batch_id[smask]

        Xt, St = self.init_interface(X, S, paratope_mask, batch_id)
        dt = 1.0 / n_steps
        # Xt = interface_X.clone()
        # St = interface_S.clone()
        
        with torch.no_grad():
            for i in range(n_steps):
                t = torch.tensor(i * dt, device=X.device)
                
                # 更新当前状态
                X_cur = X.clone()
                S_cur = S.clone()
                X_cur[paratope_mask] = Xt
                S_cur[paratope_mask] = St
                
                # 使用message passing获取速度场
                H, pred_S, r_pred_S_logits, pred_X, _ = self._forward(
                    X_cur, S_cur, cmask, smask, paratope_mask, 
                    X_pep, S_pep, residue_pos, init_noise
                )

                # 计算速度场
                dX = pred_X[paratope_mask] - Xt
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

        # generate
        H, pred_S, r_pred_S_logits, pred_X, _ = self._forward(X, S, cmask, smask, paratope_mask, X_pep, S_pep, residue_pos, init_noise)

        # PPL
        if not self.struct_only:
            S_logits = r_pred_S_logits[-1][0][smask]
            S_dists = torch.softmax(S_logits, dim=-1)
            pred_S[smask] = torch.multinomial(S_dists, num_samples=1).squeeze()
            S_probs = S_dists[torch.arange(s_batch_id.shape[0], device=S_dists.device), pred_S[smask]]
            nlls = -torch.log(S_probs)
            ppl = scatter_mean(nlls, s_batch_id)  # [batch_size]
        else:
            ppl = torch.zeros(batch_size, device=pred_S.device)

        # 1. set generated part
        gen_X[cmask] = pred_X[cmask]
        if not self.struct_only:
            gen_S[smask] = pred_S[smask]
        
        self._clean_batch_constants()

        if return_hidden:
            return gen_X, gen_S, ppl, H
        return gen_X, gen_S, ppl

    def optimize_sample(self, X, S, cmask, smask, paratope_mask, residue_pos, lengths, predictor, opt_steps=10, init_noise=None, mask_only=False):
        self._prepare_batch_constants(S, lengths)
        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        opt_mask = smask if mask_only else cmask
        # noise_batch_id = batch_id[smask].unsqueeze(1).repeat(1, X.shape[1] * X.shape[2]).flatten()
        # noise_batch_id = batch_id[cmask].unsqueeze(1).repeat(1, X.shape[1] * X.shape[2]).flatten()
        noise_batch_id = batch_id[opt_mask].unsqueeze(1).repeat(1, X.shape[1] * X.shape[2]).flatten()

        final_X, final_S = X.clone(), S.clone()
        best_metric = torch.ones(batch_size, dtype=torch.float, device=X.device) * 1e10

        all_noise = torch.randn_like(X, requires_grad=False)
        # init_noise = torch.randn_like(X[smask], requires_grad=True)
        # init_noise = torch.randn_like(X[cmask], requires_grad=True)
        init_noise = torch.randn_like(X[opt_mask], requires_grad=True)
        optimizer = torch.optim.Adam([init_noise], lr=1.0)
        optimizer.zero_grad()
        
        for i in range(opt_steps):
            all_noise = all_noise.detach()
            X, S, cmask, smask, residue_pos, lengths = X.clone(), S.clone(), cmask.clone(), smask.clone(), residue_pos.clone(), lengths.clone()
            # all_noise[smask] = init_noise
            # all_noise[cmask] = init_noise
            all_noise[opt_mask] = init_noise
            gen_X, gen_S, _, H = self.sample(X, S, cmask, smask, paratope_mask, residue_pos, lengths, return_hidden=True, init_noise=all_noise[cmask])
            h = scatter_mean(H, batch_id, dim=0)
            pmetric = predictor.inference(h)

            # use KL to regularize noise
            mean = scatter_mean(init_noise.flatten(), noise_batch_id)  # [bs]
            std = scatter_std(init_noise.flatten(), noise_batch_id)
            # std, mean = torch.std_mean(init_noise.flatten())
            kl = -0.5 * (1 + 2 * torch.log(std) - std ** 2 - mean ** 2)

            (pmetric + kl).sum().backward()
            pmetric = pmetric.detach()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                update = pmetric < best_metric
                cupdate = cmask & update[batch_id]
                supdate = smask & update[batch_id]
                # update pmetric best history
                best_metric[update] = pmetric[update]

                final_X[cupdate] = gen_X[cupdate].detach()
                if not self.struct_only:
                    final_S[supdate] = gen_S[supdate].detach()
            
        return final_X, final_S, best_metric


if __name__ == '__main__':
    torch.random.manual_seed(0)
    # equivariance test
    embed_size, hidden_size = 64, 128
    n_channel, d = 14, 3
    n_aa_type = 20
    scale = 10
    dtype = torch.float
    device = torch.device('cuda:0')
    model = isMEANOptModel(embed_size, hidden_size, n_channel,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   k_neighbors=9, bind_dist_cutoff=6.6, n_layers=3)
    model.to(device)
    model.eval()

    ag_len, h_len, l_len = 48, 120, 110
    center_x = torch.randn(3, 1, n_channel, d, device=device, dtype=dtype) * scale
    ag_X = torch.randn(ag_len, n_channel, d, device=device, dtype=dtype) * scale
    h_X = torch.randn(h_len, n_channel, d, device=device, dtype=dtype) * scale
    l_X = torch.randn(l_len, n_channel, d, device=device, dtype=dtype) * scale

    X = torch.cat([center_x[0], ag_X, center_x[1], h_X, center_x[2], l_X], dim=0)
    S = torch.cat([torch.tensor([model.aa_feature.boa_idx], device=device),
                   torch.randint(low=0, high=20, size=(ag_len,), device=device),
                   torch.tensor([model.aa_feature.boh_idx], device=device),
                   torch.randint(low=0, high=20, size=(h_len,), device=device),
                   torch.tensor([model.aa_feature.bol_idx], device=device),
                   torch.randint(low=0, high=20, size=(l_len,), device=device)], dim=0)
    cmask = torch.tensor([0] + [0 for _ in range(ag_len)] + [0] + [1 for _ in range(h_len)] + [0] + [1 for _ in range(l_len)], device=device).bool()
    smask = torch.zeros_like(cmask)
    smask[ag_len+10:ag_len+20] = 1
    residue_pos = torch.tensor([0] + [0 for _ in range(ag_len)] + [0] + list(range(1, h_len + 1)) + [0] + list(range(1, l_len + 1)), device=device)
    lengths = torch.tensor([ag_len + h_len + l_len + 3], device=device)

    torch.random.manual_seed(1)
    gen_X, _, _ = model.sample(X, S, cmask, smask, residue_pos, lengths)
    tmpx1 = model.tmpx

    # random rotaion matrix
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q, t = U.mm(V), torch.randn(3, device=device)

    X = torch.matmul(X, Q) + t

    # this is f(Qx+t)
    model.op = (Q, t)
    torch.random.manual_seed(1)
    gen_op_X, _, _ = model.sample(X, S, cmask, smask, residue_pos, lengths)
    tmpx2 = model.tmpx
    gt_tmpx = torch.matmul(tmpx1, Q) + t
    error = torch.abs(gt_tmpx[:, :4] - tmpx2[:, :4]).sum(-1).flatten().mean()
    print(error.item())
    assert error < 1e-3
    print('independent equivariance check passed')


    gt_op_X = torch.matmul(gen_X, Q) + t

    error = torch.abs(gt_op_X[cmask][:, :4] - gen_op_X[cmask][:, :4]).sum(-1).flatten().mean()
    print(error.item())
    assert error < 1e-3
    print('independent equivariance check passed')


