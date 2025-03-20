from typing import Callable, Optional
import torch


def get_hidden_states_logger(
    layer_idx: int,
    num_hidden_layers: int = 1,
    log_interval: Optional[int] = None,
    prefix: str = "hs",
) -> Callable[[str, torch.Tensor], None]:

    import wandb
    assert 0 <= layer_idx <= num_hidden_layers + 1  # 0 for embed_tokens, num_hidden_layers + 1 for norm & lm_head
    if log_interval is None and (layer_idx != 0 and layer_idx != num_hidden_layers + 1):
        log_interval = (num_hidden_layers - 1) // 5

    # avoid zero division
    if log_interval == 0:
        log_interval = None

    @torch.no_grad()
    def logger(name: str, hidden_states: Optional[torch.Tensor]):
        if wandb.run is None or hidden_states is None:
            return
        hidden_states = hidden_states.detach()
        if log_interval is None or layer_idx % log_interval == 0:
            hs = {
                f"{prefix}_var/{layer_idx:02}_{name}": torch.var(hidden_states, dim=-1).mean().item(),
                f"{prefix}_mean/{layer_idx:02}_{name}": torch.mean(hidden_states, dim=-1).mean().item(),
                f"{prefix}_max/{layer_idx:02}_{name}": torch.max(hidden_states, dim=-1)[0].mean().item(),
                f"{prefix}_min/{layer_idx:02}_{name}": torch.min(hidden_states, dim=-1)[0].mean().item(),
            }
            wandb.log(hs, commit=False)

    return logger
