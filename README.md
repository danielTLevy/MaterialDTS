# MaterialDTS:  Inference Time Adaptation of Crystal Generative Models with Diffusion Tree Sampling


Inference-time Diffusion Tree Sampling (DTS) on top of pretrained diffusion models for multi-objective crystal material generation. Uses [MatterGen](remote/mattergen) or [DiffCSP](remote/DiffCSP) as the base generative model and optimizes material properties (stability, density, bandgap, dielectric constant, bulk modulus, space group, etc.) without retraining.

This repository contains the code for the paper [Diffusion Tree Sampling for Inference Time Adaptation of Crystal Generative Models](https://arxiv.org/abs/2611.17462).

## Overview

Pretrained diffusion models for crystals are good priors over plausible materials, but their samples are rarely optimal for a specific design objective, and fine-tuning a new model for every objective is costly. Diffusion Tree Sampling (DTS) instead adapts a fixed pretrained model at inference time. It builds a tree over the denoising trajectory: nodes are intermediate noisy structures at a set of branching timesteps, and edges are blocks of denoising steps. Rewards of completed structures are backed up the tree as soft value estimates, and children are selected in proportion to their exponentiated values, so the search samples from the pretrained model's distribution tilted toward high reward. Rewards do not need to be differentiable, e.g. predicted energy above the convex hull or space group.

**Key ideas:**
- Diffusion Tree Sampling over the denoising trajectory: reward-tilted sampling from a fixed pretrained model, with no fine-tuning or gradient guidance, reusing partial trajectories across samples
- Classifier-free-guidance-conditioned models (density, bulk modulus, bandgap, dielectric constant, space group) as proposals, with importance sampling correction to the unconditional model
- Multi-objective DTS: per-objective value estimates at every node, scalarized with a preference sampled per traversal, so one shared tree samples across the Pareto front; results are evaluated with Pareto metrics

## Installation

Requires Linux with an NVIDIA GPU (CUDA 11.8 wheels: torch 2.2.1+cu118, DGL/PyG cu118) and [uv](https://docs.astral.sh/uv/). Python 3.10 is installed by uv.

```bash
git clone --recurse-submodules https://github.com/danielTLevy/materialDTS.git
cd materialDTS
uv sync
```

## Quick Start

All scripts use Hydra and must be run as modules from the repo root:

```bash
# Stability search with MatterGen
uv run python -m scripts.run_mat model=mattergen rewards.reward_fns='[MinCorrectedEnergyAboveHullReward]'

# Use DiffCSP instead
uv run python -m scripts.run_mat model=diffcsp rewards.reward_fns='[MinCorrectedEnergyAboveHullReward]'

# Multi-objective: high dielectric constant + high bandgap
uv run python -m scripts.run_mat rewards.reward_fns='[MaxDielectricReward,MaxBandgapReward]' rewards.use_pareto_metrics=true

# Conditional DTS: space-group-conditioned MatterGen as the proposal
uv run python -m scripts.run_mat model=mattergen_space_group rewards.reward_fns='[Im3BarReward]' \
  inference.use_conditional_dts=true inference.importance_sampling_correction=true
```

## Repository Layout

- `scripts/run_mat.py` — entry point
- `src/core/sampling/` — DTS samplers (`dts_materials_multi.py` unconditional, `dts_materials_multi_cond.py` conditional) and tree checkpointing
- `src/core/materials_rewards.py` — reward functions
- `src/models/` — model adapters; `diffcsp_ext.py` adds single-step denoising for DiffCSP
- `src/evaluation/` — Pareto metrics
- `remote/mattergen` — MatterGen fork ([mattergen-DTS](https://github.com/danielTLevy/mattergen-DTS)) adding single-step denoising, guided/unconditional transition log-probabilities, extra conditioning properties, and evaluation outputs
- `remote/DiffCSP` — upstream [DiffCSP](https://github.com/jiaor17/DiffCSP)

## Data and Model Setup

MaterialDTS uses the data and checkpoints released with [MatterGen](https://github.com/microsoft/mattergen) and [DiffCSP](https://github.com/jiaor17/DiffCSP); none are distributed with this repository.

### MatterGen files (Git LFS)

MatterGen's reference dataset, MP-20 data and pretrained checkpoints are stored with [Git LFS](https://git-lfs.com) in `remote/mattergen` (see MatterGen's [installation notes](https://github.com/microsoft/mattergen#install-git-lfs)). Fetch only what is needed:

```bash
git lfs install
cd remote/mattergen
git lfs pull --include "data-release/alex-mp/reference_MP2020correction.gz,data-release/mp-20/mp_20.zip,checkpoints/mattergen_base/**,checkpoints/space_group/**,checkpoints/ml_bulk_modulus/**"
cd ../..
```

- `reference_MP2020correction.gz`: reference convex hull, used by the energy-above-hull rewards and SUN evaluation
- `mp_20.zip`: MP-20 training data (see below)
- `mattergen_base`, `space_group`, `ml_bulk_modulus`: checkpoints for the `mattergen`, `mattergen_space_group` and `mattergen_bulk` configs

### MP-20 with property columns

Models and metrics for bandgap, dielectric constant, density or bulk modulus need those properties as columns in MP-20's `train.csv` and `val.csv` (in `remote/mattergen/datasets/mp_20/`). MatterGen's MP-20 does not include them, so `scripts/add_property_columns.py` adds them:

| Column | Source |
|---|---|
| `bandgap_alignn` | ALIGNN `jv_optb88vdw_bandgap_alignn` prediction |
| `dielectric_alignn` | ALIGNN `jv_dfpt_piezo_max_dielectric_alignn` prediction |
| `density` | density in g/cm³, from pymatgen |
| `ml_bulk_modulus` | copy of MP-20's `azure_bulk_modulus` |

Unzip MP-20 as in MatterGen's [pre-processing instructions](https://github.com/microsoft/mattergen#pre-process-a-dataset-for-training), then add the columns to `train.csv`, `val.csv` and `test.csv` (requires a GPU):

```bash
cd remote/mattergen && unzip data-release/mp-20/mp_20.zip -d datasets && cd ../..
uv run python -m scripts.add_property_columns --csv-folder remote/mattergen/datasets/mp_20
```

### Fine-tuned MatterGen models

The `mattergen_dielectric_bandgap`, `mattergen_bulk_density` and `mattergen_density` configs use MatterGen fine-tuned on the columns above. Follow MatterGen's [fine-tuning instructions](https://github.com/microsoft/mattergen#fine-tuning-on-property-data) from `remote/mattergen`, using the MP-20 CSVs with the added columns. The new properties are already registered in the `remote/mattergen` fork, so of MatterGen's [custom-property steps](https://github.com/microsoft/mattergen#fine-tune-on-your-own-property-data) only `csv-to-dataset` is needed:

```bash
cd remote/mattergen
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache
```

Then run MatterGen's [multi-property fine-tuning](https://github.com/microsoft/mattergen#multi-property-fine-tuning) command with these properties:

| Config | Properties |
|---|---|
| `mattergen_dielectric_bandgap` | `dielectric_alignn`, `bandgap_alignn` |
| `mattergen_bulk_density` | `ml_bulk_modulus`, `density` |
| `mattergen_density` | `density` |

MatterGen's fine-tuning defaults (200 epochs, 20% conditioning dropout) are the settings used in the paper. Point the config at the fine-tuning output directory with `model.checkpoint=<output dir>`.

### DiffCSP checkpoint

Download the pretrained MP-20 generation checkpoint from DiffCSP's [pretrained checkpoints](https://drive.google.com/drive/folders/11WOc9lTZN4hkIY7SKLCIrbsTMGy9TsoW?usp=sharing) (linked in DiffCSP's [README](https://github.com/jiaor17/DiffCSP#training)) and place its `hparams.yaml` and `last.ckpt` in `remote/DiffCSP/checkpoints/mp_gen/`, or pass `model.checkpoint=<dir>`.

### Troubleshooting

`uv sync` downloads the DGL wheel from `data.dgl.ai`, which sometimes fails; rerun `uv sync` if it does.

ALIGNN (used by the bandgap, dielectric and bulk modulus rewards) downloads its pretrained models on first use. If that download fails, download the model zip manually into the ALIGNN install folder, `.venv/lib/python3.10/site-packages/alignn/`.

## Available Reward Functions

| Config key | Description |
|---|---|
| `MinCorrectedEnergyAboveHullReward` | Minimize the predicted energy above the hull (MatterSim energy with MP2020 corrections) |
| `TargetDensity15Reward` | Target density of 15 g/cm³ |
| `TargetDensity10Reward` | Target density of 10 g/cm³ |
| `MinDensityReward` | Minimize density |
| `MaxBandgapReward` | Maximize bandgap (ALIGNN prediction) |
| `MaxDielectricReward` | Maximize dielectric constant (ALIGNN prediction) |
| `MaxLogBandgapReward` | Maximize log bandgap (ALIGNN prediction; weight `rewards.log_bandgap_weight`) |
| `MaxLogDielectricReward` | Maximize log dielectric constant (ALIGNN prediction; weight `rewards.log_dielectric_weight`) |
| `BulkModulusReward` | Maximize bulk modulus (ML prediction) |
| `TargetBulkModulusReward` | Target bulk modulus `rewards.bulk_modulus_target` (ALIGNN prediction) |
| `NullReward` | No reward (best-of-N baseline) |
| `SG<N>Reward` | Target space group number N (e.g. `SG225Reward`) |
| `Im3BarReward` | Target space group 204 (Im-3) |
| `Fm3BarmReward` | Target space group 225 (Fm-3m) |

Pass multiple rewards as a list for multi-objective search. `rewards.use_pareto_metrics=true` additionally computes Pareto metrics against the training set.

## Configuration

Configs live in `conf/` and compose via Hydra (`conf/hydra.yaml`). Override any key on the command line.

**Model configs** (`conf/model/`):

| Config | Checkpoint | Conditioned on |
|---|---|---|
| `mattergen` | `mattergen_base` | unconditional |
| `mattergen_bulk` | `ml_bulk_modulus` | `ml_bulk_modulus` |
| `mattergen_bulk_density` | `bulk_modulus_density` | `ml_bulk_modulus`, `density` |
| `mattergen_density` | `density_finetune` | `density` (`model.density_condition_value`) |
| `mattergen_dielectric_bandgap` | `dielectric_bandgap_alignn_finetune` | `dielectric_alignn`, `bandgap_alignn` |
| `mattergen_space_group` | `space_group` | `space_group` (`model.space_group_target`) |
| `diffcsp` | DiffCSP `mp_gen` | unconditional |

**Search settings** (`conf/inference/default.yaml`):

| Key | Meaning |
|---|---|
| `inference.compute_budget` | Total number of function evaluations (NFEs) |
| `inference.batch_size` | Rollouts run in parallel per denoising call |
| `inference.temperature` | Temperature of the soft value backup and selection (inverse of λ) |
| `inference.progressive_width_alpha`, `inference.progressive_width_const` | Progressive widening: a node may have at most `C · N^α` children |
| `inference.expand_timesteps` | Denoising timesteps at which the tree may branch |
| `inference.use_conditional_dts` | Use a conditional (classifier-free guidance) model as the proposal |
| `inference.importance_sampling_correction` | Correct conditional proposals back to the unconditional model |
| `inference.is_self_normalize` | Self-normalize the importance weights in the value backup |
| `inference.use_lattice_is_correction` | Include lattice transitions in the importance weights |
| `inference.use_tcheby` | Tchebycheff instead of linear scalarization of the objectives |
| `inference.use_dynamic_conditioning` | Condition each traversal on a target chosen from the training-set Pareto front for its sampled preference |
| `inference.coeff_alpha` | Dirichlet concentration of the sampled objective preferences |

## Evaluation

At the end of a run, generated structures are evaluated with MatterGen's evaluation (`evaluation.mattergen_evaluate`): stability, uniqueness, and novelty are written to `metrics_unrelaxed.json`, and stable, unique, novel (SUN) structures to `sun_structures/` in the run's output directory.

## Citation

```bibtex
@misc{levy2026diffusiontreesamplinginference,
      title={Diffusion Tree Sampling for Inference Time Adaptation of Crystal Generative Models}, 
      author={Daniel T. Levy and Vineet Jain and Tara Akhound-Sadegh and Sékou-Oumar Kaba and Siamak Ravanbakhsh},
      year={2026},
      eprint={2611.17462},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2611.17462}, 
}
```

## License

MIT; see [LICENSE](LICENSE).

## Acknowledgements
- [Diffusion Tree Sampling](https://github.com/vineetjain96/Diffusion-Tree-Sampling) (Jain et al., 2025) — the method this work builds on
- [MatterGen](https://github.com/microsoft/mattergen) (Zeni et al., 2025) and [DiffCSP](https://github.com/jiaor17/DiffCSP) (Jiao et al., 2023) — base generative models
- [MatterSim](https://github.com/microsoft/mattersim) — energy-above-hull predictions
- [ALIGNN](https://github.com/usnistgov/alignn) — bandgap, dielectric constant, and bulk modulus predictions
