# Note: Working with denseTNT ver Jul 2022
TRAJ_PT_MODALS = 6
TRAJ_LENGTH = 49
MAP_LENGTH = 9
MLP_RATIO = 4
TRAJ_MASK_TOKEN_SIZE = 10
MAP_MASK_TOKEN_SIZE = 12
OUTPUT_DIM = 2

import copy
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from modeling.lib import GlobalGraphRes, CrossAttention, GlobalGraph, MLP, LayerNorm, CrossAttentionRes
import utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

class NewSubGraph(nn.Module):

    def __init__(self, hidden_size, depth=None):
        super(NewSubGraph, self).__init__()
        if depth is None:
            depth = args.sub_graph_depth
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])

        self.layer_0 = MLP(hidden_size)
        self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) for _ in range(depth)])
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        self.layer_0_again = MLP(hidden_size)

    def forward(self, input_list: list):
        batch_size = len(input_list)
        device = input_list[0].device
        hidden_states, lengths = utils.merge_tensors(input_list, device)
        hidden_size = hidden_states.shape[2]
        max_vector_num = hidden_states.shape[1]

        attention_mask = torch.zeros([batch_size, max_vector_num, max_vector_num], device=device)
        hidden_states = self.layer_0(hidden_states)
        hidden_states = self.layer_0_again(hidden_states)
        for i in range(batch_size):
            assert lengths[i] > 0
            attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.layers_2[layer_index](hidden_states)

        return torch.max(hidden_states, dim=1)[0], torch.cat(utils.de_merge_tensors(hidden_states, lengths))

class Pre_Train_DecoderRes(nn.Module):
    def __init__(self, hidden_size, out_features):
        super(Pre_Train_DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features, bias = True)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class MaskedVectorNet(nn.Module):
    r"""
    Vector-Net for pre-training.
    Note: Use 'pretrain=num' in args for doing pre-training.
    'num' between 0 and 1 indicates how much data to be masked.
    The program consists of 4 parts:
        1. Masked Dataset: Masking num part of trajectories & map info.
            See self.mask for more info.
        2. Vector-net: subgraph and global graph. Exactly the same as denseTNT
        3. Decoder:  Exactly the same as denseTNT

    :param args:
    :param pretrain_scale: 'sub' for subgraph, 'all' for subgraph + globalgraph
    """

    def __init__(self, args_: utils.Args):
        super(MaskedVectorNet, self).__init__()

        # Args
        global args
        args = args_
        hidden_size = args.hidden_size
        self.hidden_size = hidden_size
        self.traj_target_size = TRAJ_LENGTH
        self.map_target_size = MAP_LENGTH
        self.traj_mask_token = nn.Parameter(torch.zeros(1, TRAJ_MASK_TOKEN_SIZE, dtype=torch.float))
        self.map_mask_token = nn.Parameter(torch.zeros(1, MAP_MASK_TOKEN_SIZE, dtype=torch.float))
        torch.nn.init.normal_(self.traj_mask_token, std=.02)
        torch.nn.init.normal_(self.map_mask_token, std=.02)
        self.traj_pos_enc = nn.Parameter(torch.zeros(TRAJ_LENGTH, hidden_size, dtype=torch.float), requires_grad=False)
        self.map_pos_enc = nn.Parameter(torch.zeros(MAP_LENGTH, hidden_size, dtype=torch.float), requires_grad=False)
        
        # Networks
        # Encoder
        self.point_level_sub_graph = NewSubGraph(hidden_size)
        self.global_graph = GlobalGraph(hidden_size)
        # Decoder
        self.map_decoder_emb = nn.Linear(hidden_size, hidden_size, bias=True)
        self.map_decoder_self_att = GlobalGraphRes(hidden_size)
        self.map_decoder_self_att_ln =LayerNorm(hidden_size)
        self.map_decoder_self_ff = nn.Sequential(
            nn.Linear(hidden_size, MLP_RATIO * hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(MLP_RATIO * hidden_size, hidden_size, bias=True)
        )
        self.map_decoder_self_ff_ln = LayerNorm(hidden_size)
        self.map_decoder_ln = LayerNorm(hidden_size)

        self.traj_decoder_emb = nn.Linear(hidden_size, hidden_size, bias=True)
        self.traj_decoder_self_att = GlobalGraphRes(hidden_size)
        self.traj_decoder_self_att_ln =LayerNorm(hidden_size)
        self.traj_decoder_self_ff = nn.Sequential(
            nn.Linear(hidden_size, MLP_RATIO * hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(MLP_RATIO * hidden_size, hidden_size, bias=True)
        )
        self.traj_decoder_self_ff_ln = LayerNorm(hidden_size)
        self.traj_decoder_ln = LayerNorm(hidden_size)

        self.pred_map = Pre_Train_DecoderRes(hidden_size, MAP_MASK_TOKEN_SIZE)
        self.pred_traj = Pre_Train_DecoderRes(hidden_size, TRAJ_PT_MODALS * TRAJ_MASK_TOKEN_SIZE)
        

        if 'enhance_global_graph' in args.other_params:
            self.global_graph = GlobalGraphRes(hidden_size)
        if 'laneGCN' in args.other_params:
            self.laneGCN_A2L = CrossAttention(hidden_size)

        self.initialize_traj_pos_weights()

    def initialize_traj_pos_weights(self):
        # Pos Enc
        hidden_size = self.hidden_size
        omega = np.arange(hidden_size // 2, dtype=float)
        omega /= (hidden_size // 2)
        # print('Omega =', omega)
        omega = 1. / (10000**omega)

        pos = np.arange(TRAJ_LENGTH)  # (M,)
        out = np.einsum('m,d->md', pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        self.traj_pos_enc.data.copy_(torch.from_numpy(emb).float())

    def initialize_map_pos_weights(self, id):
        # Pos Enc
        hidden_size = self.hidden_size
        omega = np.arange(hidden_size // 4, dtype=float)
        omega /= (hidden_size // 4)
        omega = 1. / (10000**omega)

        pos1 = np.ones(MAP_LENGTH) * id  # (M,)
        pos2 = np.arange(MAP_LENGTH)  # (M,)
        out1 = np.einsum('m,d->md', pos1, omega)
        out2 = np.einsum('m,d->md', pos2, omega)
        emb_1_sin = np.sin(out1)
        emb_1_cos = np.cos(out1)
        emb_2_sin = np.sin(out2)
        emb_2_cos = np.cos(out2)
        emb = np.concatenate([emb_1_sin, emb_1_cos, emb_2_sin, emb_2_cos], axis=1)  # (M, D)
        self.map_pos_enc.data.copy_(torch.from_numpy(emb).float())

    def mask(self, input_list, mask_ratio, map_start_polyline_idx, device):
        r"""
        Mask the input randomly with the ratio of mask_ratio
        """
        #Mask Init
        if np.random.rand() < 0.70:
            mask_option = 'map'
        else:
            mask_option = 'traj'

        if mask_option == 'map':
            # Map Masking
            length_scene = len(input_list) - 1
            len_keep = int(length_scene * (1 - mask_ratio))
            noise = torch.rand(length_scene, device=device)
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=0)
            # keep the first subset
            ids_target = ids_shuffle[len_keep:] + 1
        elif mask_option == 'traj':
            # Traj Masking  (For Only 1 Traj)
            ids_target = [0]
        else:
            assert False, 'Must mask either map or traj or mix!'

        # Mask
        target_list = []
        masked_input_list = []
        kept_input_list = []
        for id in range(len(input_list)):
            if id in ids_target:
                # Determine the length of target
                if mask_option == 'traj':
                    target_size = self.traj_target_size
                else:
                    target_size = self.map_target_size
                # Save the unmasked target
                if input_list[id].shape[0] < target_size:
                    target_resize = torch.zeros(target_size, args.hidden_size, device=device)
                    target_resize[:input_list[id].shape[0], :] = input_list[id]
                    target_list.append(copy.deepcopy(target_resize))
                elif input_list[id].shape[0] > target_size:
                    target_resize = input_list[id][:target_size, :]
                    target_list.append(copy.deepcopy(target_resize))
                else:
                    target_resize = input_list[id]
                    target_list.append(copy.deepcopy(target_resize))
                # Mask the input
                if mask_option == 'traj':
                    # Signal for masked traj node
                    traj_obj_length = input_list[id].shape[0]
                    input_list[id][1:,:TRAJ_MASK_TOKEN_SIZE] = self.traj_mask_token.expand(traj_obj_length-1,-1) 
                    input_list[id] = input_list[id] + self.traj_pos_enc
                    masked_input_list.append(input_list[id])
                elif mask_option == 'map':
                    # Signal for masked map node
                    input_list[id] = copy.deepcopy(target_resize).to(torch.float)
                    input_list[id][1:,[-18,-17,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]] = self.map_mask_token.expand(self.map_target_size-1,-1)
                    self.initialize_map_pos_weights(id)
                    input_list[id] = input_list[id] + self.map_pos_enc
                    masked_input_list.append(input_list[id])
                else:
                    assert False, 'Mask option not map nor traj!'
            else:
                kept_input_list.append(copy.deepcopy(input_list[id]))
        
        # Merge Tensor
        masked_input, _ = utils.merge_tensors(masked_input_list, device=input_list[0].device)
        return masked_input, kept_input_list, target_list, ids_target, mask_option

    def forward_encode_sub_graph(self, mapping: List[Dict], matrix: List[np.ndarray], polyline_spans: List[List[slice]],
                                 device, batch_size, mask_ratio) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Mask + Sub-graph
        """

        element_states_batch = []
        target_list_list = []
        masked_list_list = []
        ids_target_list = []
        mask_option_list = []
        kept_list_list = []
        mask_option_list = []
        for i in range(batch_size):
            # Rand Rotate
            Theta = 2 * np.pi * np.random.rand()
            input_list = []
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            for j, polyline_span in enumerate(polyline_spans[i]):
                mat = matrix[i][polyline_span].copy()
                if j < map_start_polyline_idx:
                    # Traj
                    mat[:, 0], mat[:, 1] = utils.rotate(mat[:, 0], mat[:, 1], Theta)
                    mat[:, 2], mat[:, 3] = utils.rotate(mat[:, 2], mat[:, 3], Theta)
                else:
                    # Map
                    mat[:, -3], mat[:, -4] = utils.rotate(mat[:, -3], mat[:, -4], Theta)
                    mat[:, -1], mat[:, -2] = utils.rotate(mat[:, -1], mat[:, -2], Theta)
                    mat[:, -17], mat[:, -18] = utils.rotate(mat[:, -17], mat[:, -18], Theta)
                tensor = torch.tensor(mat, device=device)
                input_list.append(copy.deepcopy(tensor))
            masked_input_list, kept_input_list, target_unmasked_list, ids_target, mask_option = \
                self.mask(input_list, mask_ratio, map_start_polyline_idx, device=device)
            kept_list_list.append(kept_input_list)
            masked_list_list.append(masked_input_list)
            target_list_list.append(target_unmasked_list)
            ids_target_list.append(ids_target)
            mask_option_list.append(mask_option)

            # sub-graph
            a, _ = self.point_level_sub_graph(kept_input_list)
            element_states_batch.append(a)

        # LaneGCN
        if 'laneGCN' in args.other_params:
            for i in range(batch_size):
                if mask_option_list[i] == 'traj':
                    lanes = element_states_batch[i][:]
                    lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0),lanes.unsqueeze(0)).squeeze(0)
                    element_states_batch[i] = lanes
                elif mask_option_list[i] == 'map':
                    agents = element_states_batch[i][:1]
                    lanes = element_states_batch[i][1:]
                    lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), torch.cat([lanes, agents[0:1]]).unsqueeze(0)).squeeze(0)
                    element_states_batch[i] = torch.cat([agents, lanes])
                else:
                    assert False, 'Mask option not map nor traj!'
                
        return element_states_batch, kept_list_list, target_list_list, masked_list_list, ids_target_list, mask_option_list

    def forward(self, mapping: List[Dict], device):
        matrix = utils.get_from_mapping(mapping, 'matrix')
        polyline_spans = utils.get_from_mapping(mapping, 'polyline_spans')
        batch_size = len(matrix)
        if args.argoverse:
            utils.batch_init(mapping)

        # Input + Sub Graph
        element_states_batch, kept_list_list, target_unmasked, target_masked, ids_target_list, mask_option_list = \
            self.forward_encode_sub_graph(mapping, matrix, polyline_spans, device, batch_size,
                                          mask_ratio=float(args.other_params['pretrain']))

        # Global Graph Input
        inputs, inputs_lengths = utils.merge_tensors(element_states_batch, device=device)
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        # Global Graph (Self-Attention)
        hidden_states = self.global_graph(inputs, attention_mask, mapping)
        

        query_lengths = []
        for tensor in target_masked:
            query_lengths.append(tensor.shape[0] * tensor.shape[1] if tensor is not None else 0)
        target_queries = torch.zeros([len(target_masked), max(query_lengths), self.hidden_size], device=device)
        for i, tensor in enumerate(target_masked):
            if tensor is not None:
                target_queries[i][:(tensor.shape[0] * tensor.shape[1])] = tensor.reshape(-1, self.hidden_size)
        

        # Decoder: Object Reconstruction
        decoder_hidden = torch.cat([target_queries, hidden_states], dim=1)
        decoder_hidden = self.map_decoder_emb(decoder_hidden)

        dec_self_mask = torch.zeros([batch_size, max_poly_num + max(query_lengths), max_poly_num + max(query_lengths)], device=device)
        for i, length in enumerate(inputs_lengths):
            dec_self_mask[i,max(query_lengths):length+max(query_lengths),:max(query_lengths):length+max(query_lengths)].fill_(1)
            dec_self_mask[i,:query_lengths[i],:query_lengths[i]].fill_(1)
        output_hidden_states = decoder_hidden + self.map_decoder_self_att(self.map_decoder_self_att_ln(decoder_hidden), dec_self_mask, mapping)
        output_hidden_states = output_hidden_states + self.map_decoder_self_ff(self.map_decoder_self_ff_ln(output_hidden_states))
        output_hidden_states = self.map_decoder_ln(output_hidden_states)
        # For Map
        target_pred_map = self.pred_map(output_hidden_states[:, :max(query_lengths), :])
        target_pred_map = target_pred_map.reshape(batch_size, -1, MAP_LENGTH, MAP_MASK_TOKEN_SIZE)
        
        decoder_hidden = torch.cat([target_queries, hidden_states], dim=1)
        decoder_hidden = self.traj_decoder_emb(decoder_hidden)

        dec_self_mask = torch.zeros([batch_size, max_poly_num + max(query_lengths), max_poly_num + max(query_lengths)], device=device)
        for i, length in enumerate(inputs_lengths):
            dec_self_mask[i,max(query_lengths):length+max(query_lengths),:max(query_lengths):length+max(query_lengths)].fill_(1)
            dec_self_mask[i,:query_lengths[i],:query_lengths[i]].fill_(1)
        output_hidden_states = decoder_hidden + self.traj_decoder_self_att(self.traj_decoder_self_att_ln(decoder_hidden), dec_self_mask, mapping)
        output_hidden_states = output_hidden_states + self.traj_decoder_self_ff(self.traj_decoder_self_ff_ln(output_hidden_states))
        output_hidden_states = self.traj_decoder_ln(output_hidden_states)
        # For Traj
        target_pred_traj = self.pred_traj(output_hidden_states[:, :TRAJ_LENGTH, :])
        target_pred_traj = target_pred_traj.reshape(batch_size, TRAJ_LENGTH, TRAJ_PT_MODALS, TRAJ_MASK_TOKEN_SIZE)
        target_pred_traj = target_pred_traj.permute(0, 2, 1, 3)
        
        # Loss
        batch_length = 0
        loss = torch.zeros(batch_size, device=device)
        count = torch.zeros(TRAJ_PT_MODALS,device=device)
        for i in range(batch_size):
            for k in range(len(ids_target_list[i])):
                if mask_option_list[i] == 'traj':
                    loss_array = torch.zeros(TRAJ_PT_MODALS,device=device)
                    for modality in range(TRAJ_PT_MODALS):
                        loss_array[modality] = F.l1_loss \
                            (target_pred_traj[i, modality, :self.traj_target_size, :].reshape(-1), \
                            target_unmasked[i][k][:, :TRAJ_MASK_TOKEN_SIZE].reshape(-1), reduction='none').mean()
                    chosen_loss, chosen_modal = torch.min(loss_array, dim = 0)
                    loss[i] = loss[i] + chosen_loss
                    count[chosen_modal] = count[chosen_modal] + 1
                    lowest_count, unused_modal = torch.min(count, dim = 0)
                    if lowest_count == 0:
                        loss[i] = loss[i] + 0.05 * loss_array[unused_modal]
                elif mask_option_list[i] == 'map':
                    loss[i] = loss[i] + F.l1_loss \
                        (target_pred_map[i, k, :self.map_target_size, :].reshape(-1), \
                         target_unmasked[i][k][:, [-18,-17,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]].reshape(-1), reduction='none').mean()
                else:
                    assert False
            batch_length = len(ids_target_list[i])
            loss[i] = loss[i] / batch_length
        
        # Plotting
        if 'pt_visualize' in args.other_params:
            plt.figure()
            for j in range(len(ids_target_list[i])):
                if mask_option_list[i] == 'traj':
                    predicted = target_pred_traj[i, 0:TRAJ_PT_MODALS, :,[2,3]].detach()
                else:
                    predicted = target_pred_map[i, j, :,[-3,-4]].detach()
                predicted = predicted.cpu().numpy()
       
                if mask_option_list[i] == 'map':
                    real_map = target_unmasked[i][j][:, [-3,-4]].detach()
                    real_map = real_map.cpu().numpy()
                    plt.plot(real_map[:, 0], real_map[:, 1],linewidth=1,color='orange')
                    plt.plot(predicted[:self.map_target_size, 0], predicted[:self.map_target_size, 1], linewidth=1, color='blue')
                elif mask_option_list[i] == 'traj':
                    real_traj = target_unmasked[i][j][:, [2, 3]].detach()
                    real_traj = real_traj.cpu().numpy()
                    for m in range(6):
                        plt.plot(predicted[m, :self.traj_target_size, 0], predicted[m, :self.traj_target_size, 1], linewidth=1.5, color='red')
                        plt.plot(predicted[m, -1, 0], predicted[m, -1, 1], markersize=8, color="red", marker="*")
                    plt.plot(real_traj[:, 0], real_traj[:, 1], linewidth=2, color='green')
                    plt.plot(real_traj[-1, 0], real_traj[-1, 1], markersize=8, color="green", marker="*")
        
            for k in range(len(kept_list_list[i])):
                if kept_list_list[i][k][0,-5]:
                    map_elem = kept_list_list[i][k][:,[-3,-4]].detach()
                    map_elem = map_elem.cpu()
                    map_elem = map_elem.numpy()
                    plt.plot(map_elem[:,0],map_elem[:,1],color='black',alpha=0.5,linewidth=0.75)
                elif kept_list_list[i][k][0,9]:
                    traj_elem = kept_list_list[i][k][:, 2:4].detach()
                    traj_elem = traj_elem.cpu()
                    traj_elem = traj_elem.numpy()
                    plt.plot(traj_elem[:, 0], traj_elem[:, 1], color='grey', linewidth=2.5)
                    plt.plot(traj_elem[-1, 0], traj_elem[-1, 1], markersize=8, color="grey", marker="*")
                else:
                    assert False, 'Element not map nor traj!'
            ax = plt.gca()
            ax.set_aspect(1)
            custom_lines = [Line2D([0], [0], color='orange', lw=4),
                            Line2D([0], [0], color='blue', lw=4),
                            Line2D([0], [0], color='green', lw=4),
                            Line2D([0], [0], color='red', lw=4)]
            ax.legend(custom_lines, ['Real Map', 'Pred Map', 'Real Traj','Pred Traj'])
            fig_path = os.path.join(args.output_dir,'fig')
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            plt.savefig(os.path.join(fig_path,'{0}-{1}.png'.format(args.cur_epoch, args.fig_name)))
            plt.close('all')
            args.fig_name += 1
        return loss.mean()
