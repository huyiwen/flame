import torch
from torch import nn
import torch.nn.functional as F
from functools import lru_cache


def forward_block(weight: torch.Tensor, block_indices: torch.Tensor, x: torch.Tensor, block_size: int = 16, hidden_ratio: int = 4):
    
    bsz, _ = block_indices.shape
    intermediate_size, hidden_size = weight.shape
    block_x_num = hidden_size // block_size
    block_y_num = intermediate_size // block_size

    results = torch.zeros(bsz, intermediate_size, dtype=x.dtype, device=x.device)

    for i in range(block_x_num):
        for j in range(block_x_num):
            
            # load cur_x
            cur_x = []
            good_k = []
            for k in range(bsz):
                idx = (j - i + block_x_num) % block_x_num
                if block_indices[k].item() == idx:
                    cur_x.append(x[k, j*block_size:(j+1)*block_size])
                    good_k.append(k)
            if len(cur_x) == 0:
                continue
            cur_x = torch.stack(cur_x, dim=0)

            cur_results = F.linear(cur_x, weight[hidden_ratio*i*block_size:hidden_ratio*(i+1)*block_size, j*block_size:(j+1)*block_size])
            for ki, k in enumerate(good_k):
                results[k, hidden_ratio*i*block_size:hidden_ratio*(i+1)*block_size] = cur_results[ki]

    return results


def get_updated_expert_bias(tokens_per_expert, expert_bias, expert_bias_update_rate, update_type="recentered"):
    """Update expert bias for biased expert routing. See https://arxiv.org/abs/2408.15664v1#

    Args:
        tokens_per_expert (torch.Tensor): The number of tokens assigned to each expert.
        expert_bias (torch.Tensor): The bias for each expert.
        expert_bias_udpate_rate (float): The update rate for the expert bias.
    """
    with torch.no_grad():
        # All Reduce Across TPxCPxDP group
        # torch.distributed.all_reduce(
        #     tokens_per_expert,
        #     group=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
        # )
        average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        offset = average_tokens - tokens_per_expert
        if update_type == "recentered":
            updated_expert_bias = expert_bias + torch.sign(offset - offset.mean(dim=-1)) * expert_bias_update_rate
        else:
            updated_expert_bias = expert_bias + torch.sign(offset) * expert_bias_update_rate
        return updated_expert_bias
    


class Network(nn.Module):

    def __init__(self, hidden_size=64, intermediate_size=256):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        
        self.block_size = 16
        self.block_x_num = hidden_size // self.block_size
        self.block_y_num = intermediate_size // self.block_size
        self.router = nn.Linear(hidden_size, self.block_x_num)

        self.register_buffer(
            "expert_bias", torch.zeros(self.block_x_num, dtype=torch.float32)
        )
        self.register_buffer(
            "local_tokens_per_expert", torch.zeros(self.block_x_num, dtype=torch.float32)
        )

    def _maintain_float32_expert_bias(self):
        if getattr(self, "expert_bias", None) is not None and self.expert_bias.dtype != torch.float32:
            self.expert_bias.data = self.expert_bias.data.to(torch.float32)

    def forward(self, x):
        self._maintain_float32_expert_bias()

        logits = F.linear(x, self.router.weight)
        scores = torch.sigmoid(logits) + self.expert_bias
        block_indices = scores.topk(1, dim=-1)[1]
        topk_map = torch.zeros_like(logits).int().scatter(1, block_indices, 1).bool()
        # print("logits", logits)
        # print("scores", scores)
        # print("topk_map", topk_map)
        # print("block_indices", block_indices)
        # block_mask = create_block_mask(block_indices, self.block_x_num, self.block_y_num)
        # print("block_mask", block_mask)

        hidden_states = forward_block(self.up_proj.weight, block_indices, x)

        if torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += topk_map.sum(dim=0)
        return hidden_states


if __name__ == "__main__":

    seed = 42
    torch.manual_seed(seed)

    net = Network()

    x = torch.randn(64, 64)
    y = net(x)
    # print(y)
    print(net.local_tokens_per_expert)

    x2 = x + torch.randn(64, 64)
    
    updated_expert_bias = get_updated_expert_bias(net.local_tokens_per_expert, net.expert_bias, 0.009)
    net.local_tokens_per_expert.zero_()
    net.expert_bias.copy_(updated_expert_bias)

    y2 = net(x)
    # print(y2)
    print(net.local_tokens_per_expert)
