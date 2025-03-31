import torch
from torch import nn
import torch.nn.functional as F
from functools import lru_cache


@torch.compile
def load_weight_with_block_mask(weight: torch.Tensor, block_mask: torch.Tensor, block_size: int = 16):

    block_x_num, block_y_num = block_mask.shape

    loaded_weight = torch.zeros_like(weight)

    for i in range(block_y_num):
        for j in range(block_x_num):
            if block_mask[j, i] == 1:  # 如果 block_mask[i, j] 为 1，则加载对应的块
                x_start, x_end = i * block_size, (i + 1) * block_size
                y_start, y_end = j * block_size, (j + 1) * block_size
                loaded_weight[x_start:x_end, y_start:y_end] = weight[x_start:x_end, y_start:y_end]

    print(loaded_weight, loaded_weight.shape)
    return loaded_weight


def forward_block(weight: torch.Tensor, block_mask: torch.Tensor, x: torch.Tensor, block_size: int = 16, hidden_ratio: int = 4):
    loaded_weight = load_weight_with_block_mask(weight, block_mask, block_size)
    results = F.linear(x, loaded_weight)
    return results


@lru_cache(maxsize=None)
def _create_block_mask(block_idx: int, block_x_num: int, block_y_num: int):
    block_mask = torch.zeros(block_x_num, block_y_num)
    for i in range(block_x_num):
        x = (block_idx + i) % block_x_num
        y = i * 4
        block_mask[x, y:y+4] = 1
    return block_mask


def create_block_mask(block_indices: torch.LongTensor, block_x_num: int, block_y_num: int):
    return _create_block_mask(block_indices.item(), block_x_num, block_y_num)


class Network(nn.Module):

    def __init__(self, hidden_size=32, intermediate_size=128):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        
        self.block_size = 16
        self.block_x_num = hidden_size // self.block_size
        self.block_y_num = intermediate_size // self.block_size
        router = torch.randn(hidden_size // self.block_size, hidden_size)
        self.register_buffer("router", router)

    def forward(self, x):

        block_logits, block_indices = F.linear(self.router, x).topk(1, dim=0)
        print(block_logits)
        print(block_indices)
        block_mask = create_block_mask(block_indices, self.block_x_num, self.block_y_num)
        print(block_mask)

        print(x)
        hidden_states = forward_block(self.up_proj.weight, block_mask, x)
        return hidden_states


if __name__ == "__main__":

    seed = 42
    torch.manual_seed(seed)

    net = Network()

    x = torch.randn(1, 32)
    y = net(x)

    print(y)
