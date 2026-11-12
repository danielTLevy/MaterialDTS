import logging
import re
from datetime import datetime
from src.evaluation.pareto_metrics import ParetoMetrics
import numpy as np
import pandas as pd
import colorlog
import torch
import random
import hydra
import os
import json
from typing import List, Optional, Tuple


from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
from mattergen.generator import structure_from_model_output
from src.models.model_adapter import MattergenAdapter, DiffCSPAdapter
from mattergen.common.data.condition_factory import get_number_of_atoms_condition_loader
from mattergen.scripts.evaluate import main as mattergen_evaluate
from src.utils.pareto_utils import pareto_front

from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from src.core.sampling.dts_checkpoint import checkpoint_aware_training

_HYDRA_CONF_PATH = str(Path(__file__).resolve().parents[1] / "conf")

from mattergen.common.data.chemgraph import ChemGraphBatch
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import (
    load_model_diffusion,
)
from mattergen.common.data.collate import collate
from mattergen.generator import CrystalGenerator
from src.core.sampling.dts_materials_multi import MaterialDTSDiffusionSampler
from src.core.sampling.dts_materials_multi_cond import (
    MaterialDTSDiffusionSampler as MaterialDTSDiffusionSamplerCond,
)
import src.core.materials_rewards as rewards_module
import pprint
from hydra.utils import instantiate

import wandb


# Setup logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)-2s [%(asctime)s]%(reset)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG':    'cyan', 'INFO':     'green', 'WARNING':  'yellow',
        'ERROR':    'red', 'CRITICAL': 'red,bg_white',
    }
))

root_logger = logging.getLogger()
for h in root_logger.handlers[:]: root_logger.removeHandler(h)
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def x_to_struc(x: ChemGraphBatch):
    lengths, angles = lattice_matrix_to_params_torch(x.cell)
    x = x.replace(lengths=lengths, angles=angles)

    strucs = structure_from_model_output(
        x["pos"].reshape(-1, 3),
        x["atomic_numbers"].reshape(-1),
        x["lengths"].reshape(-1, 3),
        x["angles"].reshape(-1, 3),
        x["num_atoms"].reshape(-1),
    )
    return strucs


def reload_hydra():
    # The diffusion model loader may initialize Hydra internally. Since
    # this script is already running under Hydra (hydra.main), that
    # causes a "GlobalHydra is already initialized" error. Clear the
    # global Hydra instance here so the loader can initialize its own
    # Hydra context. This avoids modifying the mattergen code.
    try:
        from hydra.core.global_hydra import GlobalHydra
        # Only clear if already initialized to avoid unnecessary side-effects
        if getattr(GlobalHydra.instance(), "is_initialized", lambda: True)():
            logger.info("Clearing GlobalHydra to avoid re-initialization in loader")
            GlobalHydra.instance().clear()
    except Exception:
        # If anything goes wrong with clearing, continue and let the
        # loader attempt to initialize; we'll catch errors below.
        logger.debug("Could not clear GlobalHydra (continuing anyway)")


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for Python's random, NumPy and PyTorch (CPU and CUDA).

    Also sets cuDNN flags to favor deterministic behavior. This helps
    reproducibility across runs when `cfg.seed` is provided.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

def save_terminal_pareto_plot_and_samples(
    terminal_nodes: list,
    reward_fns: list,
    output_dir: str,
    batch_size: int,
    diffusion_sampler,
    filename: str = "terminal_pareto_scatter.png",
):
    if not terminal_nodes:
        logger.warning("No terminal nodes found; skipping terminal Pareto plot.")
        return None

    # ``final_output`` is a no-op for MatterGen (its node ``x_t`` is already a
    # ChemGraphBatch) but converts DiffCSP/dummy node states into a ChemGraphBatch,
    # which the reward functions and ``x_to_struc`` require.
    rewards_chunks = []
    for i in range(0, len(terminal_nodes), batch_size):
        batch = collate([diffusion_sampler.final_output(n.x_t) for n in terminal_nodes[i:i + batch_size]])
        reward_cols = []
        for fn in reward_fns:
            r = fn(batch)
            if isinstance(r, np.ndarray):
                r = torch.from_numpy(r)
            r = torch.as_tensor(r)
            if r.ndim == 0:
                r = r.unsqueeze(0)
            reward_cols.append(r.detach())
        rewards_chunks.append(torch.stack(reward_cols, dim=1))
    rewards = torch.cat(rewards_chunks, dim=0)
    rewards_np = rewards.detach().cpu().numpy()
    if rewards_np.ndim != 2 or rewards_np.shape[1] < 2:
        logger.warning("Terminal Pareto plot requires at least 2 objectives; skipping.")
        return None

    mask = pareto_front(rewards_np)
    if not np.any(mask):
        logger.warning("No Pareto points found among terminal nodes; skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(rewards_np[:, 0], rewards_np[:, 1], s=12, alpha=0.25, label="Terminal nodes")
    ax.scatter(rewards_np[mask, 0], rewards_np[mask, 1], s=30, color="red", label="Pareto front")
    ax.set_xlabel("Reward 1")
    ax.set_ylabel("Reward 2")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = Path(output_dir) / filename
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    pareto_nodes = [terminal_nodes[i] for i in np.where(mask)[0]]
    pareto_structures = []
    for node in pareto_nodes:
        pareto_structures.extend(x_to_struc(diffusion_sampler.final_output(node.x_t)))
    torch.save(
        {"structures": pareto_structures, "rewards": rewards[mask].detach().cpu()},
        Path(output_dir) / "terminal_pareto_samples.pt",
    )
    logger.info(f"Saved terminal Pareto scatter and samples to {out_path}")
    return out_path


def load_model_mattergen(checkpoint_dir):
    model_path = checkpoint_dir
    checkpoint_epoch = 'last'
    config_overrides = [
     "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
    ]
    strict_checkpoint_loading = True
    checkpoint_info = MatterGenCheckpointInfo(
        model_path=Path(model_path).resolve(),
        load_epoch=checkpoint_epoch,
        config_overrides=config_overrides,
        strict_checkpoint_loading=strict_checkpoint_loading,
    )
    reload_hydra()
    model = load_model_diffusion(checkpoint_info)
    logger.info(f"Loaded model from {model_path} at epoch {checkpoint_epoch}")
    return model, checkpoint_info


def setup_checkpoint_dir(cfg, output_path, wandb_run=None):
    """
    - Uses wandb run ID when wandb is active (stable across preemption/requeue to different jobs)
    - Falls back to SLURM_JOB_ID for non-wandb cluster runs
    - Falls back to model name + timestamp for local runs
    - User can override with cfg.checkpoint.dir
    """

    # Check if user specified a checkpoint directory
    if hasattr(cfg, 'checkpoint') and cfg.checkpoint.get('dir') is not None:
        checkpoint_dir = Path(cfg.checkpoint.dir)
        logger.info(f"Using user-specified checkpoint dir: {checkpoint_dir}")

    # Use wandb run ID — stable across preemption/requeue regardless of SLURM job ID.
    # Check the env var first (set by wandb sweep agent for both new and requeued runs),
    # then fall back to the live run object (for direct wandb.init() usage without a sweep).
    elif (wandb_run_id := os.getenv('WANDB_RUN_ID') or (wandb_run.id if wandb_run is not None else None)):
        checkpoint_dir = Path(cfg.checkpoint.root) / f"wandb_{wandb_run_id}"
        logger.info(f"Using wandb-run checkpoint dir: {checkpoint_dir}")

    # Fall back to SLURM job ID for non-wandb cluster runs
    elif hasattr(cfg, 'checkpoint') and cfg.checkpoint.get('use_slurm_job_id', False):
        slurm_job_id = os.getenv('SLURM_JOB_ID')
        if slurm_job_id:
            checkpoint_dir = Path(cfg.checkpoint.root) / f"job_{slurm_job_id}"
            logger.info(f"Using SLURM checkpoint dir: {checkpoint_dir}")
        else:
            model_name = cfg.model.model
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            checkpoint_dir = Path(cfg.checkpoint.root) / model_name / timestamp
            logger.info(f"SLURM_JOB_ID not found, using timestamped dir: {checkpoint_dir}")

    else:
        model_name = cfg.model.model
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_dir = Path(cfg.checkpoint.root) / model_name / timestamp
        logger.info(f"Using checkpoint dir: {checkpoint_dir}")
    
    # Create directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Symlink from Hydra run dir → checkpoint dir (and back) for easy navigation
    hydra_link = Path(output_path) / "dts_checkpoints"
    if not hydra_link.exists() and not hydra_link.is_symlink():
        os.symlink(str(checkpoint_dir.resolve()), str(hydra_link))
        logger.info(f"Symlinked {hydra_link} -> {checkpoint_dir}")

    ckpt_link = checkpoint_dir / "hydra_output"
    if not ckpt_link.exists() and not ckpt_link.is_symlink():
        os.symlink(str(Path(output_path).resolve()), str(ckpt_link))
        logger.info(f"Symlinked {ckpt_link} -> {output_path}")

    return str(checkpoint_dir.resolve())


def build_condition_dict(cfg: DictConfig) -> Optional[dict]:
    """Build the static classifier-free-guidance condition dict from the
    ``condition_on_*`` model flags. Returns ``None`` when unconditional.
    """
    condition_dict = None
    if getattr(cfg.model, 'condition_on_density', False):
        density_value = float(cfg.model.density_condition_value)
        if density_value <= 0:
            raise ValueError(
                "model.density_condition_value must be strictly positive."
            )
        print(f"Conditioning on density={density_value}")
        condition_dict = {
            'density': density_value
        }
    elif cfg.model.condition_on_dielectric_bandgap:
        print("Conditioning on dielectric_alignn=150 and bandgap_alignn=6.0")
        condition_dict = {
            'dielectric_alignn': 150.0,
            'bandgap_alignn': 6.0
        }
    elif cfg.model.condition_on_bulk_modulus:
        print("Conditioning on bulk_modulus=400")
        condition_dict = {
            'ml_bulk_modulus': 400.0
        }
    elif getattr(cfg.model, 'condition_on_bulk_density', False):
        print("Conditioning on ml_bulk_modulus=400 and density=5.0")
        condition_dict = {
            'ml_bulk_modulus': 400.0,
            'density': 5.0,
        }
    elif getattr(cfg.model, 'condition_on_space_group', False):
        sg = int(getattr(cfg.model, 'space_group_target', 225))
        print(f"Conditioning on space_group={sg}")
        condition_dict = {'space_group': sg}
    return condition_dict


def build_diffusion_sampler(cfg: DictConfig, checkpoint: Optional[str]) -> Tuple[object, Optional[dict]]:
    """Load the base diffusion model and wrap it in the appropriate adapter.

    Returns the diffusion sampler adapter and the static condition dict (the
    latter is ``None`` for non-MatterGen / unconditional runs).
    """
    condition_dict = None
    if cfg.model.model == "mattergen":
        logger.info(f"Using MatterGen model adapter")
        model, checkpoint_info = load_model_mattergen(checkpoint)
        # Eval mode is required for element_mask_func (SELECTED_ATOMIC_NUMBERS
        # allowlist) to be applied when sampling atom types; the checkpoint
        # loader returns the module in train mode. Previously only the
        # importance-sampling path called .eval() as a side effect.
        model.eval()
        model = model.to(cfg.device)

        # Create conditioning generator
        condition_dict = build_condition_dict(cfg)

        properties_to_condition_on = {}
        batch_size = cfg.inference.batch_size
        num_batches = int(getattr(cfg.model, 'num_batches', 10000))
        sampling_config_name = 'default'
        sampling_config_overrides = None
        record_trajectories = False
        _sampling_config_path = None
        conditioning = (
            getattr(cfg.model, 'condition_on_density', False)
            or cfg.model.condition_on_dielectric_bandgap
            or cfg.model.condition_on_bulk_modulus
            or getattr(cfg.model, 'condition_on_bulk_density', False)
            or getattr(cfg.model, 'condition_on_space_group', False)
        )
        # Use guidance factor for conditional generation (matches notebook)
        diffusion_guidance_factor = (
            float(getattr(cfg.model, 'guidance_factor', 2.0)) if conditioning else None
        )
        target_compositions = None
        generator = CrystalGenerator(
            checkpoint_info=checkpoint_info,
            properties_to_condition_on=properties_to_condition_on,
            batch_size=batch_size,
            num_batches=num_batches,
            sampling_config_name=sampling_config_name,
            sampling_config_path=_sampling_config_path,
            sampling_config_overrides=sampling_config_overrides,
            record_trajectories=record_trajectories,
            diffusion_guidance_factor=(
                diffusion_guidance_factor if diffusion_guidance_factor is not None else 0.0
            ),
            target_compositions_dict=target_compositions,
        )

        sampling_config = generator.load_sampling_config(
            batch_size=batch_size,
            num_batches=num_batches,
            target_compositions_dict=None,
        )
        mattergen_sampler_partial = instantiate(sampling_config.sampler_partial)
        mattergen_sampler = mattergen_sampler_partial(pl_module=model)
        diffusion_sampler = MattergenAdapter(mattergen_sampler, device=cfg.device)
        diffusion_sampler.set_condition_dict(condition_dict)


        num_atoms_loader = get_number_of_atoms_condition_loader(num_atoms_distribution='ALEX_MP_20',
                                     num_samples=cfg.inference.compute_budget,
                                     batch_size=1,
                                     shuffle=True,
                                     properties=diffusion_sampler.condition_dict)
        diffusion_sampler.set_num_atoms_loader(num_atoms_loader)
    elif cfg.model.model == "diffcsp":
        from src.models import diffcsp_ext
        checkpoint_dir = checkpoint
        reload_hydra()
        model = diffcsp_ext.load_model(Path(checkpoint_dir))
        model.eval()
        model = model.to(cfg.device)
        diffusion_sampler = DiffCSPAdapter(model, device=cfg.device, batch_size=cfg.inference.batch_size,
                        n_samples=cfg.inference.compute_budget,
                        mask_elements=cfg.model.diffcsp_mask_elements)
    return diffusion_sampler, condition_dict


def build_train_rewards(cfg: DictConfig) -> Optional[np.ndarray]:
    """Load the training-set reward baseline used by Pareto metrics.
    Returns ``None`` when no reward needs dataset info.
    """
    train_rewards = None
    # If one of the rewards in cfg.reward_fns needs dataset info, load it once here
    if cfg.rewards.use_pareto_metrics:
        data_root =  Path(__file__).resolve().parents[1] / 'remote/mattergen/datasets/mp_20/'
        train_csv = data_root / 'train.csv'
        train_dataset = pd.read_csv(train_csv)
        train_rewards = []
        if 'TargetDensity15Reward' in cfg.rewards.reward_fns:
            train_rewards.append( -np.abs(train_dataset['density'] - 15.)/20)
        if 'MaxDielectricReward' in cfg.rewards.reward_fns:
            train_rewards.append( train_dataset['dielectric_alignn']/1000.0)
        if 'MaxBandgapReward' in cfg.rewards.reward_fns:
            train_rewards.append( train_dataset['bandgap_alignn']/8.0)
        if 'MaxLogDielectricReward' in cfg.rewards.reward_fns:
            floor = rewards_module.LOG_REWARD_FLOOR
            w = float(cfg.rewards.log_dielectric_weight)
            train_rewards.append( np.log(np.clip(train_dataset['dielectric_alignn'], floor, None))*w)
        if 'MaxLogBandgapReward' in cfg.rewards.reward_fns:
            floor = rewards_module.LOG_REWARD_FLOOR
            w = float(cfg.rewards.log_bandgap_weight)
            train_rewards.append( np.log(np.clip(train_dataset['bandgap_alignn'], floor, None))*w)
        if 'BulkModulusReward' in cfg.rewards.reward_fns:
            bulk_col = (
                'ml_bulk_modulus'
                if 'ml_bulk_modulus' in train_dataset.columns
                else 'azure_bulk_modulus'
            )
            bulk_scale = float(cfg.rewards.get('bulk_modulus_scale', 400.0))
            train_rewards.append(train_dataset[bulk_col] / bulk_scale)
        if 'TargetBulkModulusReward' in cfg.rewards.reward_fns:
            bulk_col = (
                'ml_bulk_modulus'
                if 'ml_bulk_modulus' in train_dataset.columns
                else 'azure_bulk_modulus'
            )
            bulk_target = float(cfg.rewards.get('bulk_modulus_target', 350.0))
            bulk_scale = float(cfg.rewards.get('bulk_modulus_scale', 400.0))
            train_rewards.append(
                -np.abs(train_dataset[bulk_col] - bulk_target) / bulk_scale
            )
        if 'MinDensityReward' in cfg.rewards.reward_fns:
            train_rewards.append( -train_dataset['density']/20.)
        train_rewards = np.vstack(train_rewards).T
    return train_rewards


def build_reward_fns(cfg: DictConfig) -> list:
    """Instantiate the configured reward callables
    """
    reward_fns = []
    for fn_name in cfg.rewards.reward_fns:
        if fn_name == 'TargetDensity15Reward':
            reward_fns.append(rewards_module.TargetDensity15Reward(weight=1/20.).compute)
        elif fn_name == 'TargetDensity10Reward':
            reward_fns.append(rewards_module.TargetDensity10Reward(weight=1/20.).compute)
        elif fn_name == 'MaxBandgapReward':
            reward_fns.append(rewards_module.MaxBandgapReward(device=cfg.device, weight=1./8.).compute)
        elif fn_name == 'MaxDielectricReward':
            reward_fns.append(rewards_module.MaxDielectricReward(device=cfg.device, weight=1./1000.).compute)
        elif fn_name == 'MaxLogBandgapReward':
            reward_fns.append(rewards_module.MaxLogBandgapReward(
                device=cfg.device, weight=float(cfg.rewards.log_bandgap_weight)).compute)
        elif fn_name == 'MaxLogDielectricReward':
            reward_fns.append(rewards_module.MaxLogDielectricReward(
                device=cfg.device, weight=float(cfg.rewards.log_dielectric_weight)).compute)
        elif fn_name == "BulkModulusReward":
            bulk_scale = float(cfg.rewards.get('bulk_modulus_scale', 400.0))
            reward_fns.append(
                rewards_module.MaxBulkModulusReward(
                    device=cfg.device,
                    weight=1.0 / bulk_scale,
                ).compute
            )
        elif fn_name == "TargetBulkModulusReward":
            bulk_target = float(cfg.rewards.get('bulk_modulus_target', 350.0))
            bulk_scale = float(cfg.rewards.get('bulk_modulus_scale', 400.0))
            reward_fns.append(
                rewards_module.TargetBulkModulusReward(
                    device=cfg.device,
                    target_value=bulk_target,
                    weight=1.0 / bulk_scale,
                ).compute
            )
        elif fn_name == 'MinDensityReward':
            reward_fns.append(rewards_module.MinDensityReward(weight=1/20.).compute)
        elif fn_name == 'MinEnergyAboveHullReward':
            reward_fns.append(rewards_module.MinEnergyAboveHullReward(weight=1.).compute)
        elif fn_name == 'MinCorrectedEnergyAboveHullReward':
            reward_fns.append(rewards_module.MinCorrectedEnergyAboveHullReward(weight=1.).compute)
        elif fn_name == 'NullReward':
            reward_fns.append(rewards_module.NullReward().compute)
        elif fn_name == 'Fm3BarmReward':
            reward_fns.append(rewards_module.Fm3BarmReward(weight=1.0).compute)
        elif fn_name == 'Im3BarReward':
            reward_fns.append(rewards_module.Im3BarReward(weight=1.0).compute)
        elif m := re.fullmatch(r'SG(\d+)Reward', fn_name):
            sg_num = int(m.group(1))
            reward_fns.append(rewards_module.SpaceGroupReward(target_space_group=sg_num, weight=1.0).compute)
        else:
            print("NO REWARDS")
            raise ValueError(f"Unknown reward function: {fn_name}")
    print("USING REWARD FNS:")
    print(reward_fns)
    return reward_fns


def save_terminal_artifacts(dts_sampler, diffusion_sampler, output_dir: str, wandb_run) -> Tuple[list, list]:
    """Collect terminal nodes from the finished tree, persist their structures,
    values and NFEs, and log the structure count. Returns
    ``(all_terminal_nodes, all_terminal_structures)``.
    """
    all_terminal_nodes = [node for node in dts_sampler.get_all_nodes() if node.t <= 0]
    all_terminal_structures = [x_to_struc(diffusion_sampler.final_output(node.x_t)) for node in all_terminal_nodes]
    torch.save(all_terminal_structures, output_dir + '/all_terminal_structures.pt')
    all_terminal_values = [(node.value_est, node.value_vec) for node in all_terminal_nodes]
    torch.save(all_terminal_values, output_dir + '/all_terminal_values.pt')
    all_terminal_nfes = [(node.nfe_created) for node in all_terminal_nodes]
    torch.save(all_terminal_nfes, output_dir + '/all_terminal_nfes.pt')
    tree_stats = dts_sampler.get_tree_stats()
    torch.save(tree_stats, output_dir + '/tree_stats.pt')

    if wandb_run:
        wandb_run.log({"num_structures": len(all_terminal_structures)})
    return all_terminal_nodes, all_terminal_structures


def write_cifs_and_metadata(all_terminal_structures: list, all_terminal_nodes: list, output_dir: str) -> str:
    """Write every terminal structure to a CIF file plus a JSON metadata map of
    CIF filename -> node info. Returns the CIF output directory.
    """
    # Evaluate all resulting structures
    # First, convert ChemGraphBatch to cif files
    cif_output_dir = output_dir + '/all_cifs'
    os.makedirs(cif_output_dir, exist_ok=True)
    for i, structure in enumerate(all_terminal_structures):
        structure[0].to(filename=os.path.join(cif_output_dir, f'all_final_{i}.cif'))

    # Create mapping of CIF filename to node metadata (including NFE)
    cif_mapping = {}
    for i, node in enumerate(all_terminal_nodes):
        cif_mapping[f'all_final_{i}.cif'] = {
            'node_idx': i,
            'nfe_created': node.nfe_created,
            'value_est': float(node.value_est),
            'value_vec': node.value_vec.cpu().numpy().tolist() if node.value_vec is not None else None,
            'timestep': node.t,
            'visit_count': node.visit_count,
        }

    with open(os.path.join(output_dir, 'cif_metadata.json'), 'w') as f:
        json.dump(cif_mapping, f, indent=2)
    return cif_output_dir


def run_mattergen_evaluation(cfg: DictConfig, cif_output_dir: str, all_terminal_structures: list,
                             output_dir: str, wandb_run) -> None:
    """Run MatterGen's structure evaluation (relaxation + SUN metrics) on the
    generated CIFs and log the resulting metrics to wandb.
    """
    try:
        if cfg.evaluation.mattergen_evaluate:
            mattergen_evaluate(
                structures_path=cif_output_dir,
                relax=cfg.evaluation.relax_structures,
                structure_matcher=cfg.evaluation.structure_matcher,
                save_as=os.path.join(output_dir, cfg.evaluation.save_as),
                save_sun_structures=os.path.join(output_dir, 'sun_structures') if cfg.evaluation.save_sun_structures else None,
                structures_output_path=os.path.join(output_dir, 'relaxed_structures.extxyz')
            )
            # Read resulting metrics file
            metrics_json = json.load(open(os.path.join(output_dir, cfg.evaluation.save_as), 'r'))
            metrics_data = {key: value['value'] for key, value in metrics_json.items()}
            metrics_keys = metrics_data.keys()
            # Convert fractional metrics to total numbers
            num_structures = len(all_terminal_structures)
            for key in list(metrics_keys):
                if 'frac' in key:
                    total_key = key.replace('frac', 'num')
                    metrics_data[total_key] = metrics_data[key] * num_structures
            num_sun = metrics_data.get('frac_novel_unique_stable_structures', 0)* num_structures
            if wandb_run:
                wandb_run.log({'mattergen_eval/num_sun': num_sun})
                for key, value in metrics_data.items():
                    wandb_run.log({f'mattergen_eval/{key}': value})
    except Exception as e:
        logger.error(f"Error during mattergen evaluation: {e}")


@dataclass
class _ConditioningSpec:
    """Declarative description of one value-based conditioning setup.

    Attributes:
        value_cols: Names reported to the model as the (scaled) value columns.
        target_cols: Column names reported to the model as the target columns.
        source_cols: Columns to read from the training set; scaled to form values, kept raw for targets.
        scales: Per-source-column scaling applied to the value array for normalization.
        log_cols: Source columns whose value is log-transformed (clamped at
            ``LOG_REWARD_FLOOR``) *before* ``scales`` is applied, matching the
            ``MaxLog*Reward`` reward functions.
    """
    value_cols: List[str]
    target_cols: List[str]
    source_cols: List[str]
    scales: dict
    log_cols: Tuple[str, ...] = ()


def _load_conditioning_arrays(
    value_source_cols: List[str],
    target_source_cols: List[str],
    scales: dict,
    log_cols: Tuple[str, ...] = (),
) -> dict:
    """Load train.csv and build the scaled-value / raw-target Pareto-front arrays.

    Targets are the unscaled ``target_source_cols``; values are ``value_source_cols``
    after applying the (optional) log transform and then ``scales``. The Pareto front
    is computed on the scaled values, and both arrays are masked to that front.

    The value array must live in the *same space as the rewards*: it is what
    ``_condition_for_preference`` scalarizes to pick each traversal's
    conditioning target, and its per-objective max is the default Tchebycheff
    ideal point compared against node value vectors. Columns listed in
    ``log_cols`` are therefore transformed exactly like ``LogProperty`` does.

    Args:
        value_source_cols: Columns forming the value (reward-space) array.
        target_source_cols: Columns forming the raw target array.
        scales: Per-source-column multiplicative scale for the value array.
        log_cols: Subset of ``value_source_cols`` to log-transform first.

    Returns:
        dict with ``cond_front_values`` and ``cond_front_targets``.
    """
    data_root = Path(__file__).resolve().parents[1] / 'remote/mattergen/datasets/mp_20/'
    usecols = list(dict.fromkeys(value_source_cols + target_source_cols))
    train_dataset = pd.read_csv(data_root / 'train.csv', usecols=usecols)
    train_dataset = train_dataset.dropna()
    # Targets are the raw (unscaled) source columns, captured before scaling.
    cond_front_targets = train_dataset[target_source_cols].to_numpy(dtype=np.float32)
    scaled = train_dataset.copy()
    for col in log_cols:
        if col in scaled.columns:
            scaled[col] = np.log(np.clip(scaled[col], rewards_module.LOG_REWARD_FLOOR, None))
    for col, scale in scales.items():
        if col in scaled.columns:
            scaled[col] = scaled[col] * scale
    cond_front_values = scaled[value_source_cols].to_numpy(dtype=np.float32)
    train_pareto_mask = pareto_front(cond_front_values)
    return {
        "cond_front_values": cond_front_values[train_pareto_mask],
        "cond_front_targets": cond_front_targets[train_pareto_mask],
    }


def _using_log_dielectric_bandgap(cfg: DictConfig) -> bool:
    """Whether the run uses the log-valued dielectric/bandgap rewards.

    The conditioning value array, the Tchebycheff ideal point and the node value
    vectors must all live in the same space, so the value transform is driven by
    the configured reward functions rather than set independently. A config
    without a ``rewards`` section (partial configs in tests) means the raw path.

    Raises:
        ValueError: If raw and log variants of the same objective are mixed, which
            would leave the value array in a space no reward actually produces.
    """
    fns = set(OmegaConf.select(cfg, "rewards.reward_fns", default=None) or [])
    log_fns = {'MaxLogDielectricReward', 'MaxLogBandgapReward'}
    raw_fns = {'MaxDielectricReward', 'MaxBandgapReward'}
    if fns & log_fns and fns & raw_fns:
        raise ValueError(
            "Cannot mix raw and log dielectric/bandgap rewards: "
            f"{sorted(fns & (log_fns | raw_fns))}. The conditioning values and the "
            "Tchebycheff ideal point can only live in one space."
        )
    return bool(fns & log_fns)


def build_conditioning_kwargs(cfg: DictConfig) -> dict:
    """Build the kwargs used for conditioning on different properties

    Conditioning is based on the pareto front of the training dataset.
    This requires:
        - The name of the property in the training dataset (source)
        - The name of the property used when conditioning the model (target)
        - A way to convert from the property into a reward (value), e.g. by scaling.
           The values are needed to compute a pareto front.
    The targets and values of the training set pareto front are pre-loaded and
    passed in as arguments.
    """
    if not cfg.inference.use_conditional_dts:
        return {}

    # The value array must match the reward space, so the dielectric/bandgap
    # value transform follows whichever reward variant is configured.
    diel_band_log = _using_log_dielectric_bandgap(cfg)
    if diel_band_log:
        diel_band_scales = {
            "dielectric_alignn": float(cfg.rewards.log_dielectric_weight),
            "bandgap_alignn": float(cfg.rewards.log_bandgap_weight),
        }
        diel_band_log_cols = ("dielectric_alignn", "bandgap_alignn")
    else:
        diel_band_scales = {"dielectric_alignn": 1.0 / 1000.0, "bandgap_alignn": 1.0 / 8.0}
        diel_band_log_cols = ()

    specs = {
        "condition_on_dielectric_bandgap": _ConditioningSpec(
            value_cols=["dielectric_alignn", "bandgap_alignn"],
            target_cols=["dielectric_alignn", "bandgap_alignn"],
            source_cols=["dielectric_alignn", "bandgap_alignn"],
            scales=diel_band_scales,
            log_cols=diel_band_log_cols,
        ),
        "condition_on_bulk_density": _ConditioningSpec(
            value_cols=["ml_bulk_modulus", "density"],
            target_cols=["ml_bulk_modulus", "density"],
            source_cols=["azure_bulk_modulus", "density"],
            scales={"azure_bulk_modulus": 1.0 / 400.0, "density": -1.0 / 20.0},
        ),
    }
    for flag, spec in specs.items():
        if getattr(cfg.model, flag, False):
            return {
                "cond_front_value_cols": spec.value_cols,
                "cond_front_target_cols": spec.target_cols,
                **_load_conditioning_arrays(
                    value_source_cols=spec.source_cols,
                    target_source_cols=spec.source_cols,
                    scales=spec.scales,
                    log_cols=spec.log_cols,
                ),
            }
    return {}


def build_dts_sampler(
    cfg: DictConfig,
    diffusion_sampler,
    reward_fns: list,
    condition_dict: Optional[dict],
    conditioning_kwargs: dict,
    output_dir: str,
    wandb_run,
    pareto_metrics,
) -> object:
    """Construct the DTS sampler, conditional or unconditional.

    The conditional sampler's constructor is a superset of the unconditional
    one's, so both are built from a single shared ``common_kwargs`` dict; only
    the conditioning-specific extras and the sampler class differ between them.

    Args:
        cfg: Hydra config.
        diffusion_sampler: The diffusion model adapter to wrap.
        reward_fns: Reward callables for the (multi-objective) search.
        condition_dict: Property-conditioning targets (conditional path only).
        conditioning_kwargs: Conditioning arrays/columns
            (conditional path only); empty dict when unused.
        output_dir: Directory for visualizations/results.
        wandb_run: Active wandb run, or None.
        pareto_metrics: Pareto-metrics helper, or None.

    Returns:
        An initialized DTS sampler instance.
    """
    expand_timesteps_run = list(cfg.inference.expand_timesteps)
    num_inference_steps = int(cfg.model.num_inference_steps)
    eta = float(cfg.model.eta) if hasattr(cfg.model, 'eta') else 1.0
    visualization_timesteps = list(expand_timesteps_run + [0])

    common_kwargs = dict(
        diffusion_model=diffusion_sampler,
        reward_fns=reward_fns,
        expand_timesteps=expand_timesteps_run,
        num_nfes=cfg.inference.compute_budget,
        num_inference_steps=num_inference_steps,
        eta=eta,
        device=cfg.device,
        exploration_type=cfg.inference.exploration_type,
        use_exploration=cfg.inference.use_exploration,
        exploration_const=float(cfg.inference.exploration_const),
        progressive_width_const=float(cfg.inference.progressive_width_const),
        progressive_width_alpha=cfg.inference.progressive_width_alpha,
        use_progress_bar=bool(cfg.inference.use_progress_bar),
        use_rollout=True,
        batch_size=cfg.inference.batch_size,
        output_dir=output_dir,
        visualization_timesteps=visualization_timesteps,
        temperature=cfg.inference.temperature,
        wandb_run=wandb_run,
        pareto_metrics=pareto_metrics,
    )

    if cfg.inference.use_conditional_dts:
        if cfg.model.model != "mattergen":
            raise NotImplementedError("Conditional DTS only implemented for MatterGen model currently.")
        return MaterialDTSDiffusionSamplerCond(
            **common_kwargs,
            condition_dict=condition_dict,
            use_importance_sampling=cfg.inference.importance_sampling_correction,
            use_lattice_is_correction=cfg.inference.use_lattice_is_correction,
            use_is_in_selection=cfg.inference.use_is_in_selection,
            is_logeta_clip=cfg.inference.is_logeta_clip,
            is_self_normalize=cfg.inference.is_self_normalize,
            use_tcheby=cfg.inference.use_tcheby,

            use_dynamic_conditioning=cfg.inference.use_dynamic_conditioning,
            use_conditional_model=cfg.inference.use_conditional_dts,
            force_expand_to_fill_batch=bool(cfg.inference.force_expand_to_fill_batch),
            enable_diagnostics=bool(cfg.inference.enable_diagnostics),
            coeff_alpha=cfg.inference.coeff_alpha,
            **conditioning_kwargs,
        )
    return MaterialDTSDiffusionSampler(**common_kwargs)


@hydra.main(version_base=None, config_path=_HYDRA_CONF_PATH, config_name="default")
def main(cfg: DictConfig):
    # cfg is a MatConfig provided by Hydra (CLI overrides possible)
    logger.info("--- Initial Configuration ---")
    # Hydra passes an OmegaConf DictConfig (loaded from YAML).
    # Convert to a plain container for logging and saving.
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    logger.info("\n" + pprint.pformat(cfg_container))

    # Apply optional global seed from Hydra config for reproducibility
    if hasattr(cfg, "seed") and cfg.seed is not None:
        logger.info(f"Setting global random seed to cfg.seed={cfg.seed}")
        set_random_seed(cfg.seed)

    # Resolve relative paths in the config against the repository root so
    # Hydra's runtime working directory (which changes per-run) does not
    # affect resolving repo-local files like checkpoints.
    repo_root = Path(__file__).resolve().parents[1]

    checkpoint = None
    if cfg.model.model != "dummy":
        checkpoint_path = Path(cfg.model.checkpoint)
        if not checkpoint_path.is_absolute():
            resolved_checkpoint = (repo_root / checkpoint_path).resolve()
            logger.info(f"Resolving relative checkpoint path '{cfg.model.checkpoint}' -> '{resolved_checkpoint}'")
            checkpoint = str(resolved_checkpoint)
        else:
            checkpoint = str(checkpoint_path.resolve())

    # Replace manual output_dir handling with Hydra's run dir
    run_dir = HydraConfig.get().run.dir  # absolute path created by Hydra
    output_path = Path(os.path.join(os.getcwd(), run_dir))
    output_path.mkdir(parents=True, exist_ok=True)  # safe no-op if Hydra already created it
    output_dir = str(output_path)
    logger.info(f"Using Hydra run directory for outputs: {output_path}")
    with open(output_path / 'config.txt', 'w') as f:
        f.write("--- Initial Configuration ---\n")
        f.write(pprint.pformat(cfg_container) + '\n')


    # Load Model
    diffusion_sampler, condition_dict = build_diffusion_sampler(cfg, checkpoint)

    logger.info(f"Using reward functions: {cfg.rewards.reward_fns}")
    train_rewards = build_train_rewards(cfg)

    pareto_metrics = None
    if cfg.rewards.use_pareto_metrics:
        pareto_metrics = ParetoMetrics(gt_rewards=-train_rewards)

    reward_fns = build_reward_fns(cfg)

    wandb_run = None
    wandb_run_id = cfg.checkpoint.get('wandb_run_id', None) if hasattr(cfg, 'checkpoint') else None
    wandb_run = wandb.init(
        project=cfg.logging.project,
        config=cfg_container,
        dir=output_dir,
        reinit=True,
        id=wandb_run_id or None,
        resume='allow',
        mode="online" if cfg.logging.use_wandb else "disabled"
    )
    run_info = {
        "experiment": cfg.logging.get("experiment", "default"),
        "output_dir": str(output_path),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "timestamp": datetime.now().isoformat(),
        "wandb_run_id": wandb_run.id if wandb_run else None,
        "wandb_run_url": wandb_run.url if wandb_run else None,
    }
    with open(output_path / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    if wandb_run:
        wandb_run.log({'dir': str(output_path)})
    # Store the YaML config separately into the hydra dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (output_path / "hparams.yaml").write_text(yaml_conf)


    conditioning_kwargs = build_conditioning_kwargs(cfg)

    dts_sampler = build_dts_sampler(
        cfg,
        diffusion_sampler,
        reward_fns,
        condition_dict=condition_dict,
        conditioning_kwargs=conditioning_kwargs,
        output_dir=output_dir,
        wandb_run=wandb_run,
        pareto_metrics=pareto_metrics,
    )
    target_nfes = cfg.inference.compute_budget

    if cfg.checkpoint.use_checkpoint:
        checkpoint_dir = setup_checkpoint_dir(cfg, output_path, wandb_run=wandb_run)
        raw_load_dir = cfg.checkpoint.get('load_dir', None)
        load_dir = None
        if raw_load_dir:
            load_dir = str(Path(raw_load_dir).resolve())
            logger.info(f"Branch mode: loading tree from {load_dir}")

        checkpoint_aware_training(
            sampler=dts_sampler,
            checkpoint_dir=checkpoint_dir,
            target_nfes=target_nfes,
            checkpoint_interval=cfg.checkpoint.get('interval', 250000),
            max_checkpoints=cfg.checkpoint.get('max_checkpoints', 3),
            use_compression=cfg.checkpoint.get('use_compression', False),
            load_dir=load_dir,
            wandb_run=wandb_run,
        )
    else:
        dts_sampler.initialize_root_nodes()
        dts_sampler.build_search_tree(num_nfes=target_nfes) # Pass NFE budget


    all_terminal_nodes, all_terminal_structures = save_terminal_artifacts(
        dts_sampler, diffusion_sampler, output_dir, wandb_run
    )

    # Compute the terminal Pareto plot while diffusion_sampler is still alive
    # (its final_output normalizes node states to ChemGraphBatch), before the
    # memory-freeing below.
    save_terminal_pareto_plot_and_samples(
        terminal_nodes=all_terminal_nodes,
        reward_fns=reward_fns,
        output_dir=output_dir,
        batch_size=cfg.inference.batch_size,
        diffusion_sampler=diffusion_sampler,
    )

    # Free the DTS tree and diffusion model from memory before running mattergen_evaluate,
    # which loads MatterSim and would otherwise OOM on large trees.
    import gc
    del dts_sampler
    del diffusion_sampler
    gc.collect()
    torch.cuda.empty_cache()

    cif_output_dir = write_cifs_and_metadata(all_terminal_structures, all_terminal_nodes, output_dir)

    run_mattergen_evaluation(cfg, cif_output_dir, all_terminal_structures, output_dir, wandb_run)

if __name__ == '__main__':
    main()
