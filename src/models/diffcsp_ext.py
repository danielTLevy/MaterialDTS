"""DTS extensions for an unmodified upstream DiffCSP (jiaor17/DiffCSP@7121d15).

Derived from DiffCSP (MIT License, Copyright (c) 2023 Rui Jiao).
"""
import os
import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from mattergen.common.utils.globals import SELECTED_ATOMIC_NUMBERS

DIFFCSP_ROOT = Path(__file__).resolve().parents[2] / "remote" / "DiffCSP"


def _import_diffcsp() -> None:
    os.environ.setdefault("PROJECT_ROOT", str(DIFFCSP_ROOT))
    if str(DIFFCSP_ROOT) not in sys.path:
        sys.path.insert(0, str(DIFFCSP_ROOT))
    cwd = os.getcwd()
    import diffcsp.common.utils  # noqa: F401  (chdirs to PROJECT_ROOT on import)
    os.chdir(cwd)


_import_diffcsp()
from diffcsp.common.data_utils import lattice_params_to_matrix_torch  # noqa: E402


def load_model(model_path: Path) -> torch.nn.Module:
    with initialize_config_dir(str(model_path), version_base=None):
        cfg = compose(config_name="hparams")
        model = hydra.utils.instantiate(
            cfg.model, optim=cfg.optim, data=cfg.data, logging=cfg.logging, _recursive_=False
        )
    ckpt = next(p for p in model_path.glob("*.ckpt") if "last" in p.name)
    return model.load_from_checkpoint(
        str(ckpt), hparams_file=str(model_path / "hparams.yaml"), strict=False
    )


def _mask_disallowed_elements(logits: torch.Tensor) -> torch.Tensor:
    allowed = torch.zeros(logits.shape[1], device=logits.device)
    allowed[torch.tensor(SELECTED_ATOMIC_NUMBERS, device=logits.device) - 1] = 1.0
    return logits * allowed[None]


@torch.no_grad()
def sample_prior(model: torch.nn.Module, batch):
    l_T = torch.randn([batch.num_graphs, 3, 3]).to(model.device)
    x_T = torch.rand([batch.num_nodes, 3]).to(model.device)
    t_T = torch.randn([batch.num_nodes, batch.atom_types.shape[1]]).to(model.device)
    if model.keep_coords:
        x_T = batch.frac_coords
    if model.keep_lattice:
        l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
    batch.frac_coords = x_T % 1.0
    batch.atom_types = t_T
    batch.lattice = l_T
    return batch


@torch.no_grad()
def denoise_one_step(model: torch.nn.Module, batch, t: int, step_lr: float = 1e-5, mask_elements: bool = False):
    """One corrector + predictor step of CSPDiffusion.sample() at timestep t (T..1)."""
    assert t > 0, "denoise_one_step requires t > 0"
    batch = batch.clone()
    times = torch.full((batch.num_graphs,), t, device=model.device)
    time_emb = model.time_embedding(times)

    alphas = model.beta_scheduler.alphas[t]
    alphas_cumprod = model.beta_scheduler.alphas_cumprod[t]
    sigmas = model.beta_scheduler.sigmas[t]
    sigma_x = model.sigma_scheduler.sigmas[t]
    sigma_norm = model.sigma_scheduler.sigmas_norm[t]
    c0 = 1.0 / torch.sqrt(alphas)
    c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

    x_t = batch.frac_coords
    l_t = batch.lattice
    t_t = batch.atom_types

    rand_x = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
    step_size = step_lr * (sigma_x / model.sigma_scheduler.sigma_begin) ** 2
    std_x = torch.sqrt(2 * step_size)
    _, pred_x, _ = model.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch)
    pred_x = pred_x * torch.sqrt(sigma_norm)
    x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not model.keep_coords else x_t

    rand_l = torch.randn_like(l_t) if t > 1 else torch.zeros_like(l_t)
    rand_t = torch.randn_like(t_t) if t > 1 else torch.zeros_like(t_t)
    rand_x = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
    adjacent_sigma_x = model.sigma_scheduler.sigmas[t - 1]
    step_size = sigma_x ** 2 - adjacent_sigma_x ** 2
    std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))
    pred_l, pred_x, pred_t = model.decoder(time_emb, t_t, x_t_minus_05, l_t, batch.num_atoms, batch.batch)
    pred_x = pred_x * torch.sqrt(sigma_norm)

    if mask_elements:
        sqrt_ab = torch.sqrt(alphas_cumprod)
        sqrt_1m_ab = torch.sqrt(1.0 - alphas_cumprod)
        pred_t_0 = _mask_disallowed_elements((t_t - sqrt_1m_ab * pred_t) / sqrt_ab)
        pred_t = (t_t - sqrt_ab * pred_t_0) / sqrt_1m_ab

    x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not model.keep_coords else x_t
    l_t_minus_1 = c0 * (l_t - c1 * pred_l) + sigmas * rand_l if not model.keep_lattice else l_t
    t_t_minus_1 = c0 * (t_t - c1 * pred_t) + sigmas * rand_t

    batch._store["atom_types"] = t_t_minus_1
    batch._store["frac_coords"] = x_t_minus_1 % 1.0
    batch._store["lattice"] = l_t_minus_1
    return batch
