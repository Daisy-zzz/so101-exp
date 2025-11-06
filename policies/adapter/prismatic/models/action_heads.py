"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS
import os
# from prismatic.models.moe_model import SmileMoENorm, SmileMoELinear, SmileMoEGate, get_attr
import torch.nn.functional as F
def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=ACTION_DIM,
        num_task_tokens=512,
        use_pro_version=False,
        attn_skip=1
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.model = MLPResNet(
            num_blocks=24,
            input_dim=input_dim*ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=ACTION_DIM,
            use_pro_version=use_pro_version,
            attention_interval=attn_skip  # 每 k 层启用一次 attention
        )
        
    def predict_action(
            self, 
            actions_hidden_states, 
            proprio=None, 
            proprio_projector=None,
            save_feat=False,
            phase="Inference"
            ):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]

        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=actions_hidden_states.dtype
        ).detach()  

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (batch, chunk_len, action_dim * hidden_dim)

        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # (1, seq_len, dim)
        if save_feat:
            action, feat = self.model(
                rearranged_actions_hidden_states,
                h_a=actions_hidden_states,
                p=proprio_features,
                h_t=task_hidden_states
                )
            return action, feat
        else:
            action = self.model(
                rearranged_actions_hidden_states,
                h_a=actions_hidden_states,
                p=proprio_features,
                h_t=task_hidden_states
                )

        
            return action
    

    
# class MLPResNet(nn.Module):
#     """MLP with residual connection blocks."""
#     def __init__(
#             self, 
#             num_blocks, 
#             input_dim, 
#             hidden_dim, 
#             output_dim,
#             use_pro_version=False
#             ):
        
#         super().__init__()
#         self.layer_norm1 = nn.LayerNorm(input_dim)
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.mlp_resnet_blocks = nn.ModuleList()

#         for _ in range(num_blocks):
#             if use_pro_version:
#                 self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
#             else:
#                 self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
                
#         self.layer_norm2 = nn.LayerNorm(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)


#     def forward(self, x, h_a=None, h_t=None, p= None):
 
#         # x: (batch_size, input_dim)
#         x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
#         x = self.fc1(x)  # shape: (batch_size, hidden_dim)
#         x = self.relu(x)  # shape: (batch_size, hidden_dim)
#         for i, block in enumerate(self.mlp_resnet_blocks):
#             x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
#         x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
#         x = self.fc2(x)  # shape: (batch_size, output_dim)
#         return x   
def find_action_head_path(base_dir):
    for f in os.listdir(base_dir):
        if f.startswith("action_head") and f.endswith(".pt"):
            return os.path.join(base_dir, f)
    raise FileNotFoundError(f"No action_head checkpoint found in {base_dir}")

class MLPResNet(nn.Module):
    """MLP with residual blocks, optionally interleaving attention blocks every k layers."""

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_pro_version: bool = False,
        attention_interval: int = 1  # 超参数 k: 每 k 层启用 attention
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.attention_interval = attention_interval

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.mlp_resnet_blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            # use_attention = (use_pro_version and ((i + 1) % attention_interval == 0))
            # if use_attention:
                if use_pro_version:
                    if i == num_blocks - 1:
                        self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim, last=True))
                    else:
                        self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim, last=False))
                else:
                    self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
            # else:
            #     self.mlp_resnet_blocks.append(IdentityBlock())

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        ### test MoE experts
        base_dir_1 = "/home/uqzzha39/vla-merging/zzz_data/checkpoint/VLA_Adapter/LIBERO/plus/configs+libero_spatial_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_spatial_no_noops--20251026_103027--30000_chkpt"
        base_dir_2 = "/home/uqzzha39/vla-merging/zzz_data/checkpoint/VLA_Adapter/LIBERO/plus/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_object_no_noops--20251028_131207--30000_chkpt"
        base_dir_3 = "/home/uqzzha39/vla-merging/zzz_data/checkpoint/VLA_Adapter/LIBERO/plus/configs+libero_goal_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_goal_no_noops--20251027_104908--30000_chkpt"
        base_dir_4 = "/home/uqzzha39/vla-merging/zzz_data/checkpoint/VLA_Adapter/LIBERO/plus/configs+libero_10_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_10_no_noops--20251027_212102--50000_chkpt"
        # base_dir_1 = "/home/uqzzha39/vla-merging/zzz_data/checkpoint/VLA_Adapter/LIBERO/noselfatt-sigmoid/LIBERO-Spatial-Pro--merge"
        # base_dir_2 = "/home/uqzzha39/vla-merging/zzz_data/checkpoint/VLA_Adapter/LIBERO/noselfatt-sigmoid/LIBERO-Object-Pro--merge"
        # base_dir_3 = "/home/uqzzha39/vla-merging/zzz_data/checkpoint/VLA_Adapter/LIBERO/noselfatt-sigmoid/LIBERO-Goal-Pro--merge"
        # base_dir_4 = "/home/uqzzha39/vla-merging/zzz_data/checkpoint/VLA_Adapter/LIBERO/noselfatt-sigmoid/LIBERO-Long-Pro--merge"
        # ah1_path = find_action_head_path(base_dir_1)
        # ah2_path = find_action_head_path(base_dir_2)
        # ah3_path = find_action_head_path(base_dir_3)
        # ah4_path = find_action_head_path(base_dir_4)
        # self.ah1 = torch.load(ah1_path, map_location="cpu")
        # self.ah2 = torch.load(ah2_path, map_location="cpu")
        # self.ah3 = torch.load(ah3_path, map_location="cpu")
        # self.ah4 = torch.load(ah4_path, map_location="cpu")

    def forward(self, x, h_a=None, h_t=None, p=None):
        """
        x: (batch_size, seq_len, input_dim)
        h_a: adapter tokens
        h_t: task tokens
        p: conditioning vector
        """
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for i, block in enumerate(self.mlp_resnet_blocks):
            # if i == 23:
            #     expert_sds = [self.ah1, self.ah2, self.ah3, self.ah4]
            #     gate_weights = test_gate_layer_weight(expert_sds)
            #     # gate_weights = torch.load("/home/uqzzha39/vla-merging/zzz_data/checkpoint/VLA_Adapter/LIBERO/noselfatt-sigmoid/gate_weights.pt", map_location=x.device)
            #     # print("gate_weights:", gate_weights)
            #     avg_routing_weights = gate_MoE_forward(x, h_t=h_t[:,i+1,:], h_a=h_a[:,i+1,:], p=p, gate_weights=gate_weights, num_experts=4, k=8)
            #     print("avg_routing_weights:", avg_routing_weights)
            x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
                
        x = self.layer_norm2(x)

        x = self.fc2(x)
        return x

        


def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class IdentityBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, h_a=None, h_t=None, p=None):
        return x
    



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))



    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x






# class MLPResNetBlock_Pro(nn.Module):
#     """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

#     def __init__(self, dim, last=False, num_heads=8):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads

#         self.ffn = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, dim),
#             nn.ReLU(),
#             )

#         # Q (from x only)
#         self.q_proj = nn.Linear(dim, dim)

#         # Self-Attention: K, V
#         self.k_self = nn.Linear(dim, dim)
#         self.v_self = nn.Linear(dim, dim)

#         # Adapter cross-attention: K, V
#         self.k_adapter = nn.Linear(dim, dim)
#         self.v_adapter = nn.Linear(dim, dim)

#         # Task cross-attention: K, V
#         self.k_task = nn.Linear(dim, dim)
#         self.v_task = nn.Linear(dim, dim)

#         self.o_proj = nn.Linear(dim, dim)

#         # gating
#         self.gating_factor = nn.Parameter(torch.zeros(1))

#         # RoPE
#         self.rope = RotaryPositionEmbedding(self.head_dim)

#         # ---- FiLM ----
#         # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
#         self.film_gen = nn.Sequential(
#             nn.Linear(dim, dim * 2),  # output γ and β
#             )


#     def apply_film(self, x, gamma, beta):
#         """FiLM: per-channel modulation"""
#         return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


#     def forward(self, x, h_a=None, h_t=None, p=None):
#         """
#         h_a: adapter tokens
#         h_t: task tokens
#         p:   possible conditioning vector (for FiLM)
#         """
#         g = self.gating_factor
#         ratio_g = torch.tanh(g)

#         conditions = []
#         if h_a is not None:
#             conditions.append(h_a)
#         if p is not None:
#             conditions.append(p)

#         h_adapter = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

#         # concat h_a and p
#         # h_adapter = torch.cat((h_a, p),dim=1)

#         h_task = h_t
#         B, T, C = x.shape
#         K_a = h_adapter.size(1) #if h_a is not None else 0
#         K_t = h_task.size(1) #if h_task is not None else 0

#         # Q
#         q_1 = self.q_proj(x)

#         # self tokens
#         k_tokens = self.k_self(x)
#         v_tokens = self.v_self(x)

#         # adapter tokens
#         k_adapter = self.k_adapter(h_adapter)
#         v_adapter = self.v_adapter(h_adapter)

#         # task tokens
#         k_task = self.k_task(h_task)
#         v_task = self.v_task(h_task)


#         # reshape -> multi-head
#         def reshape_heads(t, B, L):
#             return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


#         q_1 = reshape_heads(q_1, B, T)
#         k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
#         k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
#         k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

#         # RoPE
#         cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
#         q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
#         cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
#         _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)     
#         cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
#         _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

#         # attention scores
#         attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
#         attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
#         attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
#         attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
#         attn_weights = torch.softmax(attn_scores, dim=-1)

#         # combine V
#         v_list = [v_tokens,v_adapter,v_task]
#         v_combined = torch.cat(v_list, dim=2)

#         output = torch.matmul(attn_weights, v_combined)
#         output = output.transpose(1, 2).contiguous().view(B, T, C)
#         output = self.o_proj(output)

#         # # ---- FiLM ---- 
#         # gamma_beta = self.film_gen(p)  # [B, 2C]
#         # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
#         # output = self.apply_film(output, gamma, beta)

#         # residual + FFN
#         x = self.ffn(output + x)
#         return x

import math
import torch
import torch.nn as nn

class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

    def __init__(self, dim, num_heads=8, last=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.last = last
        # Self-Attention: K, V
        self.q_proj_self = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.q_proj_adapter = nn.Linear(dim, dim)
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.q_proj_task = nn.Linear(dim, dim)
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM ----
        # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # output γ and β
        )

    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)

    def forward(self, x, h_a=None, h_t=None, p=None):
        """
        h_a: adapter tokens
        h_t: task tokens
        p:   possible conditioning vector (for FiLM)
        """
        g = self.gating_factor
        # ratio_g = torch.tanh(g)
        ratio_g = torch.sigmoid(g)
        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        if len(conditions) > 0:
            h_adapter = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)
        else:
            # if no adapter/p provided, create zero-length tensor to avoid errors later
            # but we will guard usages below
            h_adapter = torch.zeros(x.size(0), 0, x.size(2), device=x.device, dtype=x.dtype)

        h_task = h_t if h_t is not None else torch.zeros(x.size(0), 0, x.size(2), device=x.device, dtype=x.dtype)

        B, T, C = x.shape
        K_a = h_adapter.size(1)
        K_t = h_task.size(1) 

        # self tokens
        # q_self = self.q_proj_self(x)
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)

        # adapter tokens
        q_adapter = self.q_proj_adapter(x)
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        q_task = self.q_proj_task(x)
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)


        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # q_self = reshape_heads(q_self, B, T)
        q_adapter = reshape_heads(q_adapter, B, T)
        q_task = reshape_heads(q_task, B, T)

        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        # q_self, k_tokens = apply_rope(q_self, k_tokens, cos_main, sin_main)
        q_adapter, _ = apply_rope(q_adapter, k_tokens, cos_main, sin_main)
        q_task, _ = apply_rope(q_task, k_tokens, cos_main, sin_main)

        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores (each is (B, heads, T, Lk))
        # attn_scores_self = torch.matmul(q_self, k_tokens.transpose(-2, -1))  # (B, heads, T, T)
        attn_scores_adapter = torch.matmul(q_adapter, k_adapter.transpose(-2, -1))  # (B, heads, T, K_a)
        attn_scores_task = torch.matmul(q_task, k_task.transpose(-2, -1)) * ratio_g  # (B, heads, T, K_t)

        # concat along key-length dimension
        # attn_scores = torch.cat([attn_scores_self, attn_scores_adapter, attn_scores_task], dim=-1) / math.sqrt(self.head_dim)
        attn_scores = torch.cat([attn_scores_adapter, attn_scores_task], dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V: concat along key-length dim
        # v_combined = torch.cat([v_tokens, v_adapter, v_task], dim=2)  # (B, heads, L_all, head_dim)
        v_combined = torch.cat([v_adapter, v_task], dim=2) 

        output = torch.matmul(attn_weights, v_combined)  # (B, heads, T, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # # ---- FiLM ----
        # gamma_beta = self.film_gen(p)  # [B, 2C]
        # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
        # output = self.apply_film(output, gamma, beta)

        # residual + FFN
        x = self.ffn(output + x)

         # save feature
        # if self.last:
        #     tid = os.getenv("TASK_ID", None)
        #     sid = os.getenv("SUIT_ID", None)
        #     save_dir = f"/home/uqzzha39/vla-merging/VLA-Adapter/outputs/rollout_features/{sid}"
        #     os.makedirs(save_dir, exist_ok=True)
        #     path = os.path.join(save_dir, f"rollout_{tid}_ffn1.pt")
        #     with open(path, "ab") as f:
        #         torch.save(x, f)
        #     x = nn.ReLU()(x)
        return x
    



# class L1RegressionMoEActionHead(nn.Module):

#     def __init__(
#         self,
#         input_dim=4096,
#         hidden_dim=4096,
#         action_dim=7,
#         num_experts=4,
#         num_task_tokens=512,
#         use_pro_version=True,
#     ):
#         super().__init__()
#         self.num_task_tokens = num_task_tokens
#         self.action_dim = action_dim
#         self.hidden_dim = hidden_dim
#         self.model = MLPMoEResNet(
#             num_blocks=22, 
#             input_dim=input_dim*ACTION_DIM, 
#             hidden_dim=hidden_dim, 
#             output_dim=action_dim,
#             num_experts=num_experts,
#             use_pro_version=use_pro_version,
#             )

#     def predict_action(
#             self, 
#             actions_hidden_states, 
#             proprio=None, 
#             proprio_projector=None,
#             phase="Inference"
#             ):
#         batch_size = actions_hidden_states.shape[0]
#         device = actions_hidden_states.device

#         proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
#         proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
#         proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

#         task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
#         actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]

#         cond_actions_hidden_states = torch.zeros(
#             (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
#             device=device, dtype=actions_hidden_states.dtype
#         ).detach()  

#         rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
#             batch_size, NUM_ACTIONS_CHUNK, -1
#         )  # (batch, chunk_len, action_dim * hidden_dim)

#         if phase == "Training":
#             batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
#             random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
#             rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # (1, seq_len, dim)

#         action = self.model(
#             rearranged_actions_hidden_states,
#             h_a=actions_hidden_states,
#             p=proprio_features,
#             h_t=task_hidden_states
#         )

#         return action
    


# class MLPMoEResNet(nn.Module):
#     final_head = [
#         "mlp_resnet_moe_blocks.23.gating_factor", # 0
#         "mlp_resnet_moe_blocks.23.ffn.0", # 1
#         "mlp_resnet_moe_blocks.23.ffn.1", # 2
#         "mlp_resnet_moe_blocks.23.q_proj_adapter", # 3
#         "mlp_resnet_moe_blocks.23.k_adapter", # 4
#         "mlp_resnet_moe_blocks.23.v_adapter", # 5
#         "mlp_resnet_moe_blocks.23.q_proj_task", # 6
#         "mlp_resnet_moe_blocks.23.k_task", # 7
#         "mlp_resnet_moe_blocks.23.v_task", # 8
#         "mlp_resnet_moe_blocks.23.o_proj", # 9

#         "mlp_resnet_moe_blocks.22.gating_factor", # 10
#         "mlp_resnet_moe_blocks.22.ffn.0", # 11
#         "mlp_resnet_moe_blocks.22.ffn.1", # 12
#         "mlp_resnet_moe_blocks.22.q_proj_adapter", # 13
#         "mlp_resnet_moe_blocks.22.k_adapter", # 14
#         "mlp_resnet_moe_blocks.22.v_adapter", # 15
#         "mlp_resnet_moe_blocks.22.q_proj_task", # 16
#         "mlp_resnet_moe_blocks.22.k_task", # 17
#         "mlp_resnet_moe_blocks.22.v_task", # 18
#         "mlp_resnet_moe_blocks.22.o_proj", # 19

#         "layer_norm2",
#         "fc2",
#     ]

#     def __init__(
#             self, 
#             num_blocks, 
#             input_dim, 
#             hidden_dim, 
#             output_dim,
#             num_experts=4,
#             use_pro_version=False
#             ):
        
#         super().__init__()
#         self.layer_norm1 = nn.LayerNorm(input_dim)
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.mlp_resnet_blocks = nn.ModuleList()

#         for _ in range(num_blocks):
#             if use_pro_version:
#                 self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
#             else:
#                 raise NotImplementedError("MoE version only supports Pro version now.")
                
#         self.layer_norm2 = SmileMoENorm(hidden_dim, num_experts)
#         self.fc2 = SmileMoELinear(hidden_dim, output_dim, num_experts)

#         self.mlp_resnet_moe_blocks = MLPResNetBlock_Pro_MoE(dim=hidden_dim, num_experts=num_experts)

#         modules = self._define_gate_modules()
#         self.gate = SmileMoEGate(modules)

#     def _define_gate_modules(self):
#         modules = []
#         for module_name in self.final_head:
#             if module_name == "fc2":
#                 continue
#             module_attrs = module_name.split(".")
#             module = get_attr(self, module_attrs) 
#             if isinstance(module, SmileMoELinear):
#                 modules.append(module)
#         return modules
    
#     def forward(self, x, h_a=None, h_t=None, p= None):
 
#         # x: (batch_size, input_dim)
#         x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
#         x = self.fc1(x)  # shape: (batch_size, hidden_dim)
#         x = self.relu(x)  # shape: (batch_size, hidden_dim)
#         for i, block in enumerate(self.mlp_resnet_blocks):
#             x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
        
#         # calculate gate
#         router_logits = self.gate(x) # (1, 8, 896)
#         routing_weights = F.softmax(router_logits, dim=1) # (1, 4)
#         # routing_weights = torch.tensor([[0.3, 0.1, 0.1, 0.5]]).to(x.device, dtype=x.dtype)  # NOTE: for testing purpose only
#         x = self.mlp_resnet_moe_blocks(x, routing_weights, h_t = h_t[:,i+2,:], h_a = h_a[:,i+2,:], p=p)
#         x = self.mlp_resnet_moe_blocks(x, routing_weights, h_t = h_t[:,i+3,:], h_a = h_a[:,i+3,:], p=p)

#         x = self.layer_norm2(x, routing_weights)  # shape: (batch_size, hidden_dim)
#         x = self.fc2(x, routing_weights)  # shape: (batch_size, output_dim)
#         return x



# class MLPResNetBlock_Pro_MoE(nn.Module):
#     """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

#     def __init__(self, dim, num_experts=4, num_heads=8):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads

#         self.ffn = nn.Sequential(
#             SmileMoENorm(dim, num_experts),
#             SmileMoELinear(dim, dim, num_experts),
#             nn.ReLU(),
#         )

#         # Self-Attention: K, V
#         self.q_proj_self = nn.Linear(dim, dim)
#         self.k_self = nn.Linear(dim, dim)
#         self.v_self = nn.Linear(dim, dim)

#         # Adapter cross-attention: K, V
#         self.q_proj_adapter = SmileMoELinear(dim, dim, num_experts)
#         self.k_adapter = SmileMoELinear(dim, dim, num_experts)
#         self.v_adapter = SmileMoELinear(dim, dim, num_experts)

#         # Task cross-attention: K, V
#         self.q_proj_task = SmileMoELinear(dim, dim, num_experts)
#         self.k_task = SmileMoELinear(dim, dim, num_experts)
#         self.v_task = SmileMoELinear(dim, dim, num_experts)

#         self.o_proj = SmileMoELinear(dim, dim, num_experts)

#         # gating
#         self.gating_factor = nn.Parameter(torch.zeros(num_experts))

#         # RoPE
#         self.rope = RotaryPositionEmbedding(self.head_dim)

#         # ---- FiLM ----
#         # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
#         self.film_gen = nn.Sequential(
#             nn.Linear(dim, dim * 2),  # output γ and β
#         )


#     def apply_film(self, x, gamma, beta):
#         """FiLM: per-channel modulation"""
#         return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


#     def forward(self, x, routing_weights, h_a=None, h_t=None, p=None):
#         """
#         h_a: adapter tokens
#         h_t: task tokens
#         p:   possible conditioning vector (for FiLM)
#         """
#         # mask = torch.tensor([0, 1, 0, 0], dtype=torch.bool)
#         mask = (routing_weights == routing_weights.max(dim=1, keepdim=True).values).float().argmax(dim=1)
#         # print(routing_weights)
#         # print(mask)

#         g = self.gating_factor[mask].unsqueeze(1) # (bs, 1)
#         # g = self.gating_factor
#         # ratio_g = torch.tanh(g)
#         ratio_g = torch.sigmoid(g)

#         conditions = []
#         if h_a is not None:
#             conditions.append(h_a)
#         if p is not None:
#             conditions.append(p)

#         if len(conditions) > 0:
#             h_adapter = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)
#         else:
#             # if no adapter/p provided, create zero-length tensor to avoid errors later
#             # but we will guard usages below
#             h_adapter = torch.zeros(x.size(0), 0, x.size(2), device=x.device, dtype=x.dtype)

#         h_task = h_t if h_t is not None else torch.zeros(x.size(0), 0, x.size(2), device=x.device, dtype=x.dtype)

#         B, T, C = x.shape
#         K_a = h_adapter.size(1)
#         K_t = h_task.size(1) 

#         # self tokens
#         # q_self = self.q_proj_self(x)
#         k_tokens = self.k_self(x)
#         v_tokens = self.v_self(x)

#         # adapter tokens
#         q_adapter = self.q_proj_adapter(x, routing_weights)
#         k_adapter = self.k_adapter(h_adapter, routing_weights)
#         v_adapter = self.v_adapter(h_adapter, routing_weights)

#         # task tokens
#         q_task = self.q_proj_task(x, routing_weights)
#         k_task = self.k_task(h_task, routing_weights)
#         v_task = self.v_task(h_task, routing_weights)


#         # reshape -> multi-head
#         def reshape_heads(t, B, L):
#             return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

#         # q_self = reshape_heads(q_self, B, T)
#         q_adapter = reshape_heads(q_adapter, B, T)
#         q_task = reshape_heads(q_task, B, T)

#         k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
#         k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
#         k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

#         # RoPE
#         cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
#         # q_self, k_tokens = apply_rope(q_self, k_tokens, cos_main, sin_main)
#         q_adapter, _ = apply_rope(q_adapter, k_tokens, cos_main, sin_main)
#         q_task, _ = apply_rope(q_task, k_tokens, cos_main, sin_main)

#         cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
#         _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
#         cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
#         _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

#         # attention scores (each is (B, heads, T, Lk))
#         # attn_scores_self = torch.matmul(q_self, k_tokens.transpose(-2, -1))  # (B, heads, T, T)
#         attn_scores_adapter = torch.matmul(q_adapter, k_adapter.transpose(-2, -1))  # (B, heads, T, K_a)
#         attn_scores_task = torch.matmul(q_task, k_task.transpose(-2, -1)) * ratio_g  # (B, heads, T, K_t) ######### important #* ratio_g #########

#         # concat along key-length dimension
#         # attn_scores = torch.cat([attn_scores_self, attn_scores_adapter, attn_scores_task], dim=-1) / math.sqrt(self.head_dim)
#         attn_scores = torch.cat([attn_scores_adapter, attn_scores_task], dim=-1) / math.sqrt(self.head_dim)
#         attn_weights = torch.softmax(attn_scores, dim=-1)

#         # combine V: concat along key-length dim
#         # v_combined = torch.cat([v_tokens, v_adapter, v_task], dim=2)  # (B, heads, L_all, head_dim)
#         v_combined = torch.cat([v_adapter, v_task], dim=2) 

#         output = torch.matmul(attn_weights, v_combined)  # (B, heads, T, head_dim)
#         output = output.transpose(1, 2).contiguous().view(B, T, C)
#         output = self.o_proj(output, routing_weights)

#         # # ---- FiLM ----
#         # gamma_beta = self.film_gen(p)  # [B, 2C]
#         # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
#         # output = self.apply_film(output, gamma, beta)

#         # residual + FFN
#         # x = self.ffn(output + x, routing_weights)
#         x = output + x
#         for block in self.ffn:
#             if isinstance(block, (SmileMoENorm, SmileMoELinear)):
#                 x = block(x, routing_weights)
#             else:
#                 x = block(x)

#         return x
    
# from typing import List, Optional, OrderedDict, Union, Dict
# def test_gate_layer_weight(expert_sds: List[Dict], k: int = 8):
#     from prismatic.models.moe_model import svd

#     gate_module_names = [
#         "module.model.mlp_resnet_blocks.23.ffn.1.weight",
#         "module.model.mlp_resnet_blocks.23.q_proj_adapter.weight",
#         "module.model.mlp_resnet_blocks.23.k_adapter.weight",
#         "module.model.mlp_resnet_blocks.23.v_adapter.weight",
#         "module.model.mlp_resnet_blocks.23.q_proj_task.weight",
#         "module.model.mlp_resnet_blocks.23.k_task.weight",
#         "module.model.mlp_resnet_blocks.23.v_task.weight",
#         "module.model.mlp_resnet_blocks.23.o_proj.weight",
#     ]

#     gate_weights = {}
#     gate_name_template = "model.gate.gate.{}.weights"
#     for idx, gate_module_name in enumerate(gate_module_names):
#         gate_name = gate_name_template.format(idx)
#         gate_weight = []
#         for expert_sd in expert_sds:
#             w = expert_sd[gate_module_name]
#             _, _, v = svd(w.to(dtype=torch.float32))
#             v = v[:, :k].to(torch.bfloat16)
#             gate_weight.append(v.T)
#         gate_weight = (
#             torch.stack(gate_weight, dim=0)
#             .reshape(len(expert_sds) * k, -1)
#             .contiguous()
#         ).cuda()
#         gate_weights[gate_name] = gate_weight
#     return gate_weights

# def gate_forward(x: torch.Tensor, gate_weight: torch.Tensor, num_experts: int, k: int):
#     batch_size = x.size(0)
#     # gate_weight: (num_experts * k, dim)
#     routing_weights = F.linear(x, gate_weight).view(
#         batch_size, -1, num_experts, k
#     ) # (bs, 8, 32) -> (bs, 8, num_experts, k)
#     routing_weights = routing_weights.norm(p=2, dim=3) # (bs, 8, 4)
#     return routing_weights

# def gate_MoE_forward(x, h_a, h_t, p, gate_weights: torch.Tensor, num_experts: int, k: int):
#     # All routing weights are input the same one (h_t)
#     routing_weights = []
#     gate_name_template = "model.gate.gate.{}.weights"
#     for i in range(len(gate_weights)):
#         gate_weight = gate_weights[gate_name_template.format(i)]
#     #     routing_weight = gate_forward(h_t, gate_weight, num_experts, k)
#     #     routing_weights.append(routing_weight) # (1, 8, 4)
#     # avg_routing_weights = torch.stack(routing_weights, dim=0).mean(dim=(0, 2))
#     # return avg_routing_weights
#         if i in [3, 6]:
#             routing_weight = gate_forward(h_a if i == 3 else h_t, gate_weight, num_experts, k)
#             routing_weights.append(routing_weight.mean(dim=1)) # (1, 8, 4)
  

#     avg_routing_weights = torch.stack(routing_weights, dim=0).mean(dim=(0))
#     return avg_routing_weights
    