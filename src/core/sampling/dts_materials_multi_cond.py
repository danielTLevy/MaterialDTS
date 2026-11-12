import torch
import numpy as np
import math
import tqdm
import random
import time
import sys
import warnings
import logging
from collections import defaultdict, Counter
from typing import Callable, List, Optional, Union
import torch_geometric.data as pyg_data

logging.getLogger("torch_tensorrt.dynamo.conversion._TRTInterpreter").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Both operands of the binary elementwise op .* are constant")

from mattergen.common.data.chemgraph import ChemGraphBatch
from mattergen.diffusion.sampling.pc_sampler import _sample_prior
from mattergen.common.data.condition_factory import get_number_of_atoms_condition_loader


class DTSNode:
    """
    Each node represents a latent state x_t at diffusion timestep t.
    Children are possible next states x_{t-1}.
    """

    def __init__(
        self,
        x_t,
        t,
        parent=None,
        timestep_value=None,
        *,
        edge_logp_guided: Optional[Union[float, torch.Tensor]] = None,
        edge_logp_uncond: Optional[Union[float, torch.Tensor]] = None,
        log_weight: Optional[Union[float, torch.Tensor]] = None,
        cond_key: Optional[tuple] = None,
        nfe_created: int = 0,
        preference: Optional[Union[list, np.ndarray, torch.Tensor]] = None,
    ):
        """
        Args:
          x_t (torch.Tensor): The noisy latent at diffusion timestep t.
          t (int): The current diffusion timestep index.
          parent (DTSNode or None): The parent node. None if root.
          timestep_value (float or None): The actual scheduler timestep value if known.
          nfe_created (int): NFE count at the time this node was created.
          preference (list | np.ndarray | torch.Tensor | None): Scalarization
            preference w [K] of the traversal that generated this node.
        """
        self.x_t = x_t          # ChemGraphBatch
        self.t = t               # This is the timestep index
        self.timestep_value = timestep_value  # This is the actual scheduler timestep value
        self.parent = parent
        self.children = []      # list of child nodes
        self.visit_count = 1    # N(x_t) in the algorithm
        self.value_est = 0.0    # Optional scalar estimate for visualization only
        self.value_vec = None   # torch.Tensor[K], log-domain per-objective values
        self.edge_logp_guided = (
            float(edge_logp_guided)
            if edge_logp_guided is not None and not isinstance(edge_logp_guided, torch.Tensor)
            else (edge_logp_guided.detach().item() if isinstance(edge_logp_guided, torch.Tensor) else None)
        )
        self.edge_logp_uncond = (
            float(edge_logp_uncond)
            if edge_logp_uncond is not None and not isinstance(edge_logp_uncond, torch.Tensor)
            else (edge_logp_uncond.detach().item() if isinstance(edge_logp_uncond, torch.Tensor) else None)
        )
        self.log_weight = (
            float(log_weight)
            if log_weight is not None and not isinstance(log_weight, torch.Tensor)
            else (log_weight.detach().item() if isinstance(log_weight, torch.Tensor) else 0.0)
        )
        self.cond_key: Optional[tuple] = cond_key
        self.nfe_created = nfe_created  # NFE count when this node was created
        # Preference stored as a plain list of floats so checkpoint pickles
        # stay device-free and JSON-friendly.
        if preference is None:
            self.preference: Optional[list[float]] = None
        elif isinstance(preference, torch.Tensor):
            self.preference = [float(v) for v in preference.detach().cpu().reshape(-1)]
        else:
            self.preference = [float(v) for v in np.asarray(preference).reshape(-1)]

        
    def update_value_vec(self, value_vec: torch.Tensor):
        # Expect 1D tensor [K]
        if isinstance(value_vec, np.ndarray):
            value_vec = torch.from_numpy(value_vec)
        if not isinstance(value_vec, torch.Tensor):
            value_vec = torch.tensor(value_vec)
        self.value_vec = value_vec.detach().clone().to(torch.float32)
        # Provide a lightweight scalar for tooling/visualization
        with torch.no_grad():
            try:
                self.value_est = float(self.value_vec.sum().item())
            except Exception:
                pass


class MaterialMetaRootNode(DTSNode):
    """
    A special meta-root node that doesn't correspond to any actual diffusion timestep.
    Its children are created by sampling from random noise.
    """
    def __init__(self, sampler, device):
        """
        Args:
            sampler: The DTSDiffusionSampler instance to use for sampling children
        """
        # Use a timestep index that's one more than the maximum diffusion timestep
        virtual_timestep = sampler.model.N
        # Create a dummy tensor with the right shape based on args
        # Use sampler.dims for shape
        dummy_x = None

        # Initialize with the virtual timestep and dummy tensor
        super().__init__(dummy_x, virtual_timestep, parent=None, timestep_value=None)

class MaterialDTSDiffusionSampler:
    """
    DTS-style sampler for pixel-space diffusion models with a black-box reward.
    """

    def __init__(self,
                 diffusion_model,
                 reward_fns: List[Callable],
                 expand_timesteps: List[int], 
                 num_nfes: int = 1000,
                 num_inference_steps: int = 1000,
                 eta: float = 1.0,
                 device: str = 'cuda',
                 exploration_const: float = 1.0,
                 progressive_width_const: float = 1.0,
                 progressive_width_alpha: float = 0.5,
                 initial_exploration_prob: float = 0.1,
                 final_exploration_prob: float = 0.05,
                 use_progress_bar: bool = True,
                 use_rollout: bool = True,
                 maximize: bool = False,
                 exploration_type: str = 'epsilon',
                 use_exploration: bool = True,
                 batch_size: int = 32,
                 output_dir: str = None,
                 visualization_timesteps: List[int] = None,
                 # Multi-objective options
                 reward_names: Optional[List[str]] = None,
                 coeff_mode: str = 'dirichlet',           # 'dirichlet' | 'fixed' | 'grid'
                 coeff_alpha: float = 1.0,                # Dirichlet concentration
                 coeff_fixed: Optional[List[float]] = None,
                 coeff_grid: Optional[List[List[float]]] = None,
                 coeff_resample: str = 'per_traversal',   # 'per_traversal' | 'per_node'
                 temperature: float = 1.0,                 # temperature for Boltzmann selection
                 reward_norm: str = 'none',              # 'zscore' | 'none'
                 reward_norm_eps: float = 1e-6,
                 # Behavior toggles
                 use_tcheby: bool = False,
                 use_importance_sampling: bool = False,
                 use_lattice_is_correction: bool = False,
                 use_is_in_selection: bool = True,
                 is_logeta_clip: Optional[float] = None,
                 is_self_normalize: bool = False,
                 use_dynamic_conditioning: bool = False,
                 use_conditional_model: bool = False,
                 # Conditioning / dataset stats
                 condition_dict: Optional[dict] = None,
                 cond_front_value_cols: Optional[List[str]] = None,
                 cond_front_target_cols: Optional[List[str]] = None,
                 cond_front_values: Optional[np.ndarray] = None,
                 cond_front_targets: Optional[np.ndarray] = None,
                 tcheby_ideal_values: Optional[np.ndarray] = None,
                 # Materials specific
                 wandb_run=None,
                 pareto_metrics=None,
                 force_expand_to_fill_batch: bool = True,
                 enable_diagnostics: bool = True):
        """
        Args:
          reward_fns: List of functions that take materials (ChemGraphBatch) and returns scalar rewards.
          expand_timesteps: Timesteps at which we do DTS expansion.
          num_nfes: Total DTS NFE.
          exploration_const: UCB exploration constant (lambda).
          progressive_width_const: C in progressive widening.
          progressive_width_alpha: Alpha in progressive widening.
          initial_exploration_prob: Initial epsilon for epsilon-greedy.
          final_exploration_prob: Final epsilon for epsilon-greedy.
          use_progress_bar: Whether to use tqdm progress bars.
          use_rollout: Whether to use full rollout or estimate value.
          exploration_type: 'epsilon' or 'ucb'.
          batch_size: Number of nodes to process in parallel per iteration.
          output_dir: Directory to save visualizations and results.
          visualization_timesteps: Timesteps to include in visualization.
          is_logeta_clip: If not None, clamp each per-edge importance-sampling
            log-weight contribution ``edge_logp_uncond - edge_logp_guided`` to
            ``[-is_logeta_clip, is_logeta_clip]`` before accumulating it along
            a unary chain in ``_branch_endpoints`` (truncated importance
            sampling). ``None`` leaves log-weights unclipped.
          use_is_in_selection: When ``use_importance_sampling`` is True, whether
            the branch-segment log-weight is also added to child-selection
            logits in ``_compute_selection_logits``. Selection is a behavior
            policy that only decides where NFE budget goes; the tilted-target
            correctness argument for importance sampling lives entirely in the
            value backup (``_backup``), which is unaffected by this flag.
            When False, ``_compute_selection_logits`` still scores branch
            endpoints (via ``_branch_endpoints``) rather than direct children,
            but omits the ``+ segment_log_weights`` term, i.e.
            ``logits = scores / temperature`` exactly. Defaults to True.
          use_lattice_is_correction: Whether the MatterGen paired transition
            includes lattice predictor/corrector likelihoods. Defaults to
            False.
          is_self_normalize: When ``use_importance_sampling`` is True, whether
            the IS backup in ``_backup`` normalizes by the self-normalized
            log-weight denominator ``logsumexp(log_w)`` instead of
            ``log(num_samples)``. Defaults to False.
        """
        self.model = diffusion_model
        self.device = device
        self.reward_fns = reward_fns
        self.expand_timesteps = sorted(expand_timesteps, reverse=True)
        self.num_nfes = num_nfes
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.c_ucb = exploration_const
        self.use_exploration = use_exploration
        self.progressive_width_const = progressive_width_const
        self.progressive_width_alpha = progressive_width_alpha
        self.initial_exploration_prob = initial_exploration_prob
        self.final_exploration_prob = final_exploration_prob
        self.current_exploration_prob = initial_exploration_prob
        self.use_progress_bar = use_progress_bar
        self.use_rollout = use_rollout
        self.maximize = maximize
        self.exploration_type = exploration_type
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.visualization_timesteps = visualization_timesteps
        self.wandb_run = wandb_run
        self.pareto_metrics = pareto_metrics
        self.use_tcheby = use_tcheby
        self.use_importance_sampling = use_importance_sampling
        self.use_lattice_is_correction = use_lattice_is_correction
        self.use_is_in_selection = use_is_in_selection
        self.is_logeta_clip = is_logeta_clip
        self.is_self_normalize = is_self_normalize
        self.use_dynamic_conditioning = use_dynamic_conditioning
        self.use_conditional_model = use_conditional_model
        self.condition_dict = condition_dict
        self.cond_front_value_cols = cond_front_value_cols
        self.cond_front_target_cols = cond_front_target_cols
        self.cond_front_values = cond_front_values
        self.cond_front_targets = cond_front_targets
        if tcheby_ideal_values is not None:
            self.tcheby_ideal_values = np.asarray(tcheby_ideal_values, dtype=np.float32)
        elif cond_front_values is not None:
            self.tcheby_ideal_values = np.max(cond_front_values, axis=0)
        else:
            self.tcheby_ideal_values = None
        
        # Load pipeline components
        self.setup_diffusion()

        self.nfes = 0
        self.meta_root = None

        # Multi-objective state/config
        self.reward_names = reward_names
        self.num_objectives = len(reward_names) if reward_names is not None else None
        self.coeff_mode = coeff_mode
        self.coeff_alpha = coeff_alpha
        self.coeff_fixed = torch.tensor(coeff_fixed, dtype=torch.float32) if coeff_fixed is not None else None
        self.coeff_grid = (
            [torch.tensor(w, dtype=torch.float32) for w in coeff_grid]
            if coeff_grid is not None else None
        )
        self.coeff_resample = coeff_resample
        self.temperature = temperature
        self.reward_norm = reward_norm
        self.reward_norm_eps = reward_norm_eps
        self._root_weights = None
        self._rollout_call_id = 0

        # Batch-fill behaviour and diagnostics.
        self.force_expand_to_fill_batch = force_expand_to_fill_batch
        self.enable_diagnostics = enable_diagnostics
        self._reset_batch_diagnostics()

    def _reset_batch_diagnostics(self) -> None:
        """Zero the diagnostic counters accumulated over one rollout batch.

        One rollout batch is one call to :meth:`_simulate_down_full_batched`,
        which is also the cadence at which the existing ``dts/terminal_*``
        metrics are logged. Aggregates cover the whole batch so nothing is
        emitted per ``p_sample`` or per selection step.
        """
        self._diag_psample_units = 0        # sum of per-p_sample batch sizes
        self._diag_psample_count = 0        # number of p_sample calls
        self._diag_psample_max = 0          # largest per-p_sample batch size
        self._diag_selection_passes = 0     # selection traversals in this batch
        self._diag_expansions = 0           # branching expansions created
        self._diag_forced_expansions = 0    # expansions that exceeded the PW cap
        self._diag_expansion_depth_sum = 0  # sum of branch depths of expansions

    def _record_expansion(self, node: DTSNode, is_meta: bool, forced: bool) -> None:
        """Accumulate one branching expansion into the batch diagnostics.

        The branch depth is the number of expansion levels between the
        meta-root and ``node`` inclusive: 0 for the meta-root, and
        ``1 + (#expand_timesteps greater than node.t)`` for a normal expansion
        node. Because every root-to-node path descends the ``expand_timesteps``
        in fixed order, this is a well-defined function of ``node.t``.

        Args:
            node: The parent node being expanded.
            is_meta: Whether ``node`` is the meta-root.
            forced: Whether this expansion exceeded the progressive-widening cap.
        """
        if is_meta:
            depth = 0
        else:
            depth = 1 + sum(1 for ts in self.expand_timesteps if ts > node.t)
        self._diag_expansions += 1
        self._diag_expansion_depth_sum += depth
        if forced:
            self._diag_forced_expansions += 1

    def _append_expansion_unit(
        self,
        node: DTSNode,
        path_idx: int,
        is_meta: bool,
        forced: bool,
        path_weights: dict[int, torch.Tensor],
        units_by_t: dict[int, list[dict]],
        initial_timestep_idx: int,
    ) -> None:
        """Append a rollout unit that branches one new child from ``node``.

        The unit is filled by the segmented rollout in
        :meth:`_simulate_down_full_batched`, which attaches the resulting child
        to ``node`` regardless of the progressive-widening cap. This is the
        single code path used both for an in-cap/forced expansion reached during
        selection and for the batch-fill rescue of a path that bottomed out on a
        cached terminal.

        Args:
            node: Parent node to branch from (meta-root or an expand-timestep
                node, both of which retain ``x_t``).
            path_idx: Index of the simulation path this unit belongs to.
            is_meta: Whether ``node`` is the meta-root.
            forced: Whether this expansion exceeded the progressive-widening cap.
            path_weights: Per-path scalarization weights cache.
            units_by_t: Bucket of pending rollout units keyed by timestep.
            initial_timestep_idx: Countdown index used for meta expansions.
        """
        if self.enable_diagnostics:
            self._record_expansion(node, is_meta, forced)
        weights_for_path = self._get_selection_weights_for_path(path_idx, node, path_weights)
        cond_for_node, cond_key = self._condition_for_preference(weights_for_path)
        if is_meta:
            # Meta expansion starts at countdown t=N; x_T is sampled later in one batch.
            units_by_t[initial_timestep_idx].append({
                'path_idx': path_idx,
                'parent': node,
                'x_t': None,
                't': initial_timestep_idx,
                'meta': True,
                'cond_dict': cond_for_node,
                'cond_key': cond_key,
                'weights': weights_for_path,
            })
        else:
            x_t = node.x_t
            if self.use_dynamic_conditioning or self.condition_dict:
                cond_dict = cond_for_node if cond_for_node is not None else self.condition_dict
                cond_key = cond_key if cond_for_node is not None else self._make_cond_key(self.condition_dict)
                if cond_dict:
                    x_t = self._apply_conditioning(x_t, cond_dict)
            units_by_t[node.t].append({
                'path_idx': path_idx,
                'parent': node,
                'x_t': x_t,
                't': node.t,
                'meta': False,
                'cond_dict': cond_for_node,
                'cond_key': cond_key,
                'weights': weights_for_path,
            })

    def _log_batch_diagnostics(self) -> None:
        """Log the diagnostics aggregated over the rollout batch just completed.

        Called alongside the existing ``dts/terminal_*`` logging, i.e. once per
        rollout batch when terminal nodes are created, so all DTS metrics share
        one cadence. Resets the counters afterward.
        """
        if self.wandb_run is not None:
            psample_denom = max(1, self._diag_psample_count)
            expansions_denom = max(1, self._diag_expansions)
            mean_effective_batch = self._diag_psample_units / psample_denom
            self.wandb_run.log({
                "dts/batch_effective_size_mean": float(mean_effective_batch),
                "dts/batch_effective_size_max": int(self._diag_psample_max),
                "dts/batch_fill_fraction": float(mean_effective_batch / max(1, self.batch_size)),
                "dts/selection_passes_per_batch": int(self._diag_selection_passes),
                "dts/pw_forced_expansion_fraction": float(
                    self._diag_forced_expansions / expansions_denom),
                "dts/mean_expansion_depth": float(
                    self._diag_expansion_depth_sum / expansions_denom),
            }, step=int(self.nfes))
        self._reset_batch_diagnostics()

    def setup_diffusion(self):
        """
        Set up diffusion variables.
        """
        self.num_timesteps = self.model.N
        self.timesteps = torch.arange(self.num_timesteps, device=self.device).long()
        self.num_inference_steps = self.num_timesteps

    def _initialize_meta_root(self):
        """
        Initialize a single meta-root node as the starting point for the search.
        This meta-root is at a virtual timestep beyond the normal diffusion process.
        """
        self.meta_root = MaterialMetaRootNode(self, device=self.device)
        return self.meta_root
    
    def _is_meta_root(self, node):
        return node.t >= len(self.timesteps)

    def _make_cond_key(self, properties: Optional[dict]) -> Optional[tuple]:
        if not properties:
            return None
        return tuple(sorted((str(k), float(v)) for k, v in properties.items()))

    def _apply_conditioning(self, batch: ChemGraphBatch, cond_dict: Optional[dict]) -> ChemGraphBatch:
        if not cond_dict:
            return batch
        batch_size = batch.get_batch_size()
        updates = {}
        for k, v in cond_dict.items():
            if isinstance(v, torch.Tensor):
                val = v.to(device=self.device, dtype=torch.float32)
            else:
                val = torch.tensor(v, device=self.device, dtype=torch.float32)
            if val.ndim == 0:
                val = val.repeat(batch_size)
            updates[k] = val
        return batch.replace(**updates)

    def _compute_condition_targets(self, weights: Union[torch.Tensor, np.ndarray, list]) -> Optional[dict]:
        if self.cond_front_values is None or self.cond_front_targets is None:
            return None
        if self.cond_front_target_cols is None:
            return None
        values = self.cond_front_values
        targets = self.cond_front_targets
        if isinstance(weights, torch.Tensor):
            w = weights.detach().cpu().numpy().astype(np.float32)
        else:
            w = np.asarray(weights, dtype=np.float32)
        w = w.reshape(-1)
        if values.shape[1] != len(w):
            raise ValueError(
                f"Weights length must match number of objectives ({len(w)} vs {values.shape[1]})."
            )
        s = float(w.sum())
        if s <= 0:
            w = np.full_like(w, 1.0 / len(w))
        else:
            w = w / s

        if self.use_tcheby:
            ideal = self.tcheby_ideal_values
            if ideal is None:
                ideal = np.max(values, axis=0)
            diff = ideal[None, :] - values
            weighted = diff * w[None, :]
            scores = -np.max(weighted, axis=1)
        else:
            scores = values @ w

        best_score = float(np.max(scores))
        best_mask = np.isclose(scores, best_score)
        if not np.any(best_mask):
            best_mask = scores == best_score
        best_targets = targets[best_mask]
        if best_targets.ndim == 1:
            best_targets = best_targets.reshape(1, -1)
        avg_targets = np.mean(best_targets, axis=0)
        return {
            self.cond_front_target_cols[i]: float(avg_targets[i])
            for i in range(len(self.cond_front_target_cols))
        }

    def _condition_for_preference(self, weights: Optional[torch.Tensor] = None) -> tuple[Optional[dict], Optional[tuple]]:
        if not self.use_dynamic_conditioning:
            return None, None
        if weights is None:
            K = len(self.cond_front_value_cols) if self.cond_front_value_cols is not None else self.num_objectives
            if K is None or K <= 0:
                return None, None
            weights = self._maybe_sample_weights(K)
        cond_dict = self._compute_condition_targets(weights)
        cond_key = self._make_cond_key(cond_dict)
        return cond_dict, cond_key

    def sample_noise(self, batch_size=1, properties: Optional[dict] = None):
        if not self.use_conditional_model and not self.use_dynamic_conditioning:
            return self.model.sample_prior(batch_size)
        props = self.condition_dict if properties is None else properties
        num_atoms_loader = get_number_of_atoms_condition_loader(
            num_atoms_distribution='ALEX_MP_20',
            num_samples=batch_size,
            batch_size=batch_size,
            shuffle=True,
            properties=props,
        )
        conditioning_data, mask = next(iter(num_atoms_loader))
        return _sample_prior(self.model._multi_corruption, conditioning_data, mask).to(self.device)

    def _timestep_index_from_any(self, t: Union[int, torch.Tensor, list]) -> int:
        """Return one shared DTS timestep index for a homogeneous batch."""
        if isinstance(t, (int, np.integer)):
            timestep_idx = int(t)
        elif isinstance(t, torch.Tensor):
            flat_t = t.detach().flatten()
            if flat_t.numel() == 0:
                raise ValueError("Timestep tensor must not be empty.")
            if not torch.all(flat_t == flat_t[0]):
                raise ValueError(
                    "Batched MatterGen denoising requires a single shared timestep."
                )
            timestep_idx = int(flat_t[0].item())
        elif isinstance(t, list):
            if not t:
                raise ValueError("Timestep list must not be empty.")
            timestep_idx = int(t[0])
            if any(int(value) != timestep_idx for value in t[1:]):
                raise ValueError(
                    "Batched MatterGen denoising requires a single shared timestep."
                )
        else:
            raise TypeError(f"Unsupported timestep type: {type(t)}")

        if not 1 <= timestep_idx <= self.num_timesteps:
            raise IndexError(
                f"DTS timestep index {timestep_idx} is outside "
                f"[1, {self.num_timesteps}]."
            )
        return timestep_idx

    def p_sample(self, x_t, t, predict_x0=False, eta=None, print_batch=False):
        """Perform one MatterGen denoising step at an DTS timestep index.

        The tree uses a countdown from ``N`` to ``1``.  It maps exactly to
        MatterGen predictor/corrector indices ``0`` through ``N - 1``.  Tree
        index one is the final step.  With importance correction, the paired
        transition helper returns a common deterministic unconditional
        predictor mean at that step, so the final density ratio is well-defined.

        Args:
            x_t (ChemGraphBatch): Current noisy data graph.
            t (Union[int, torch.Tensor, list]): DTS timestep index.
            predict_x0 (bool): If True, use ddim_denoise to predict x0.

        Returns:
            A pair of the next state and optional per-sample log probabilities.
        """
        batch_size = (
            int(x_t.get_batch_size())
            if hasattr(x_t, "get_batch_size")
            else len(x_t)
        )
        timestep_idx = self._timestep_index_from_any(t)
        if print_batch:
            print("BATCH SIZE: ", str(batch_size), ' TIMESTEP: ', str(timestep_idx))
        if eta is None:
            eta = self.eta

        self.nfes += batch_size

        # Call method from model
        with torch.no_grad():
            if predict_x0:
                raise NotImplementedError("predict_x0 not implemented for mattergen model")
            timestep_i = self.num_timesteps - timestep_idx

            # MatterGen's reference sampler executes all indices 0..N-1 and
            # returns the mean (not a final random draw) at index N-1.
            if not self.use_conditional_model:
                if not hasattr(self.model, "_denoise_one_step"):
                    return self.model.p_sample(
                        x_t, timestep_idx, predict_x0, eta, print_batch
                    ), None
                sample_batch, mean_batch, _ = self.model._denoise_one_step(
                    x_t,
                    mask={},
                    timestep_i=timestep_i,
                    record=False,
                )
                pred = sample_batch if timestep_idx > 1 else mean_batch
                return pred, None

            step_info = None
            if self.use_importance_sampling:
                if not hasattr(self.model, "_denoise_one_step_with_logp_pair"):
                    raise RuntimeError(
                        "Importance-corrected conditional MatterGen sampling "
                        "requires _denoise_one_step_with_logp_pair."
                    )
                paired_kwargs = {
                    "mask": {},
                    "timestep_i": timestep_i,
                    "record": False,
                }
                if getattr(self, "use_lattice_is_correction", False):
                    paired_kwargs["include_lattice_logp"] = True
                sample_batch, mean_batch, _, step_info = (
                    self.model._denoise_one_step_with_logp_pair(
                        x_t,
                        **paired_kwargs,
                    )
                )
                pred = sample_batch if timestep_idx > 1 else mean_batch
            else:
                sample_batch, mean_batch, _ = self.model._denoise_one_step(
                    x_t, mask={}, timestep_i=timestep_i, record=False
                )
                pred = sample_batch if timestep_idx > 1 else mean_batch
            return pred, step_info

    def _validated_step_log_probs(
        self,
        step_info: object,
        expected_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and return per-sample guided/unconditional transition logp."""
        if not isinstance(step_info, dict):
            raise RuntimeError(
                "Importance sampling requires step_info to be a dictionary "
                "containing per-sample transition log probabilities."
            )

        validated: list[torch.Tensor] = []
        for key in ("logp_guided", "logp_uncond"):
            value = step_info.get(key)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"Importance sampling requires tensor step_info['{key}']."
                )
            if value.ndim != 1 or value.shape[0] != expected_batch_size:
                raise RuntimeError(
                    f"step_info['{key}'] must have shape "
                    f"[{expected_batch_size}], got {tuple(value.shape)}."
                )
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"Non-finite values in step_info['{key}']."
                )
            validated.append(value.detach())
        return validated[0], validated[1]
            
    def _is_logical_branch_node(self, node: DTSNode) -> bool:
        """Return whether ``node`` represents a conceptual DTS branch state."""
        if len(node.children) > 1 or node.t in self.expand_timesteps:
            return True
        return hasattr(self, "timesteps") and self._is_meta_root(node)

    def _branch_endpoints(
        self,
        node: DTSNode,
    ) -> tuple[list[DTSNode], torch.Tensor]:
        """Find logical child endpoints and their proposal-to-target log ratios.

        A selectable child is the first scalar diffusion step of a rollout, while
        DTS expansion only occurs at ``expand_timesteps``.  Follow each
        single-child rollout until the next branching or terminal node so the
        importance ratio covers the complete conceptual tree edge.

        If ``self.is_logeta_clip`` is set, each per-edge log importance weight
        ``edge_logp_uncond - edge_logp_guided`` is clamped to
        ``[-is_logeta_clip, is_logeta_clip]`` before being accumulated into the
        segment sum (truncated importance sampling), bounding unbounded growth
        of the chain-summed log-weight with depth.

        Args:
            node: Current branching node whose immediate children identify the
                rollout alternatives.

        Returns:
            A pair containing the endpoint for every immediate child and a
            tensor of branch-segment log importance ratios ``log(p / q)``.
        """
        endpoints: list[DTSNode] = []
        segment_log_weights: list[float] = []
        for child in node.children:
            endpoint = child
            segment_log_weight = 0.0
            while True:
                if (
                    endpoint.edge_logp_guided is None
                    or endpoint.edge_logp_uncond is None
                ):
                    raise RuntimeError(
                        "Importance-corrected selection encountered a tree edge "
                        "without exact guided/unconditional transition likelihoods."
                    )
                edge_log_eta = (
                    float(endpoint.edge_logp_uncond)
                    - float(endpoint.edge_logp_guided)
                )
                if not math.isfinite(edge_log_eta):
                    raise FloatingPointError(
                        "Non-finite per-edge importance weight during "
                        "branch-endpoint traversal."
                    )
                if self.is_logeta_clip is not None:
                    edge_log_eta = max(
                        -self.is_logeta_clip,
                        min(self.is_logeta_clip, edge_log_eta),
                    )
                segment_log_weight += edge_log_eta

                if (
                    endpoint.t <= 0
                    or endpoint.t in self.expand_timesteps
                    or len(endpoint.children) != 1
                ):
                    break
                endpoint = endpoint.children[0]

            endpoints.append(endpoint)
            segment_log_weights.append(segment_log_weight)

        log_weights = torch.tensor(
            segment_log_weights,
            device=self.device,
            dtype=torch.float32,
        )
        if not torch.isfinite(log_weights).all():
            raise FloatingPointError(
                "Non-finite branch-segment importance weight during selection."
            )
        return endpoints, log_weights

    def _compute_selection_logits(
        self,
        node: DTSNode,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute value and importance-adjusted child-selection logits.

        Implements branch-level importance resampling,
        ``logit_i = score_i / temperature + log(p_i / q_i)``.  The likelihood
        ratio is not tempered because it corrects the proposal measure itself.

        If ``use_is_in_selection`` is False, branch endpoints are still used
        to score children (``score_nodes``), but the ``log(p_i / q_i)`` term
        is omitted from the logits, so selection reduces to
        ``logit_i = score_i / temperature``. This decouples the *behavior
        policy* (selection, which decides where NFE budget goes) from the
        *target-distribution correctness* claim, which lives entirely in
        ``_backup`` and is unaffected by this flag.

        Args:
            node: Current branching node.
            weights: Optional objective scalarization weights.

        Returns:
            One unnormalized logit for each immediate child of ``node``.
        """
        score_nodes = node.children
        segment_log_weights = None
        if self.use_importance_sampling:
            score_nodes, segment_log_weights = self._branch_endpoints(node)

        scores = self._compute_child_scores(score_nodes, weights)
        temperature = max(1e-6, float(self.temperature))
        logits = scores / temperature
        if segment_log_weights is not None and self.use_is_in_selection:
            logits = logits + segment_log_weights.to(
                device=logits.device,
                dtype=logits.dtype,
            )
        return logits

    def _select_child_by_value(self, node, weights=None, use_exploration=True):
        """
        Select a child of 'node' proportional to its value.
        With probability p, select a random child if exploration is enabled.
        Return the selected child_node.
        """
        if not node.children:
            return None
        
        if len(node.children) == 1:
            return node.children[0]

        if self.exploration_type == 'epsilon':
            # Epsilon-greedy exploration
            if use_exploration and random.random() < self.current_exploration_prob:
                return random.choice(node.children)
                
            # Otherwise, select based on value and proposal correction (exploitation)
            logits = self._compute_selection_logits(node, weights)
            
            # Sample a child proportional to value using torch operations
            if self.maximize:
                idx = torch.argmax(logits).item()
            else:
                logits = logits - logits.max()
                probs = torch.softmax(logits, dim=0)
                idx = torch.multinomial(probs, 1).item()
            return node.children[idx]
        
        elif self.exploration_type == 'ucb':
            # UCB exploration
            if not use_exploration:
                # If exploration is disabled, sample from corrected value logits.
                logits = self._compute_selection_logits(node, weights)
                logits = logits - logits.max()
                probs = torch.softmax(logits, dim=0)
                idx = torch.multinomial(probs, 1).item()
                return node.children[idx]
            
            # Calculate UCB scores for each child
            parent_visits = max(1, node.visit_count)  # Avoid division by zero
            
            # Vectorized UCB calculation
            logits = self._compute_selection_logits(node, weights)
            logits = logits - logits.max()
            child_visits = torch.as_tensor(
                [max(1, child.visit_count) for child in node.children],
                device=logits.device,
                dtype=logits.dtype,
            )
            parent_visits_tensor = torch.tensor(
                float(parent_visits),
                device=logits.device,
                dtype=logits.dtype,
            )
            
            # UCB formula: value + c * sqrt(ln(parent_visits) / child_visits)
            exploration_term = self.c_ucb * torch.sqrt(
                torch.log(parent_visits_tensor) / child_visits
            )
            ucb_scores = torch.exp(logits) + exploration_term
            
            # Find the child with the highest UCB score
            if self.maximize:
                idx = torch.argmax(ucb_scores).item()
            else:
                probs = ucb_scores / ucb_scores.sum()
                idx = torch.multinomial(probs, 1).item()
            return node.children[idx]
        
        else:
            raise ValueError(f"Unknown exploration type: {self.exploration_type}")

    def _compute_child_scores(self, children, weights):
        """
        Compute scalarized scores for a list of children given optional weights.
        If a child has a vector value, compute the (weighted) sum; otherwise
        fall back to a small positive scalar based on value_est.
        """
        if self.use_tcheby:
            value_vecs = []
            fallback_scores = []
            for child in children:
                if getattr(child, "value_vec", None) is not None and child.value_vec is not None:
                    v = child.value_vec
                    if isinstance(v, np.ndarray):
                        v = torch.from_numpy(v).to(self.device)
                    value_vecs.append(v.to(torch.float32))
                    fallback_scores.append(None)
                else:
                    fallback_scores.append(
                        torch.tensor(float(max(1e-6, getattr(child, "value_est", 0.0)))).to(self.device)
                    )
            if value_vecs:
                stacked = torch.stack(value_vecs, dim=0)
                K = stacked.shape[1]
                if weights is None:
                    ww = torch.full((K,), 1.0 / K, device=self.device, dtype=stacked.dtype)
                else:
                    ww = weights
                    if isinstance(ww, np.ndarray):
                        ww = torch.from_numpy(ww).to(self.device)
                    ww = ww.to(stacked.dtype)
                    s = ww.sum()
                    ww = ww / s if s > 0 else torch.full((K,), 1.0 / K, device=self.device, dtype=stacked.dtype)
                ideal = None
                if self.tcheby_ideal_values is not None:
                    ideal_arr = np.asarray(self.tcheby_ideal_values, dtype=np.float32).reshape(-1)
                    if ideal_arr.size == K:
                        ideal = torch.from_numpy(ideal_arr).to(device=self.device, dtype=stacked.dtype)
                if ideal is None:
                    ideal = torch.max(stacked, dim=0).values
                diff = ideal[None, :] - stacked
                weighted = diff * ww[None, :]
                max_term = torch.max(weighted, dim=1).values
                tcheby_scores = -max_term

                out_scores = []
                vec_idx = 0
                for fallback in fallback_scores:
                    if fallback is None:
                        out_scores.append(tcheby_scores[vec_idx])
                        vec_idx += 1
                    else:
                        out_scores.append(fallback)
                return torch.stack(out_scores)
            return torch.stack([s for s in fallback_scores if s is not None])

        scores_list = []
        for child in children:
            if getattr(child, 'value_vec', None) is not None and child.value_vec is not None:
                value_vec = child.value_vec
                if isinstance(value_vec, np.ndarray):
                    value_vec = torch.from_numpy(value_vec).to(self.device)
                value_vec = value_vec.to(torch.float32)
                if weights is None:
                    score = value_vec.sum()
                else:
                    ww = weights
                    if isinstance(ww, np.ndarray):
                        ww = torch.from_numpy(ww).to(self.device)
                    ww = ww.to(value_vec.dtype)
                    score = (ww * value_vec).sum()
                scores_list.append(score)
            else:
                scores_list.append(torch.tensor(float(max(1e-6, getattr(child, 'value_est', 0.0)))).to(self.device))
        return torch.stack(scores_list)

    def _maybe_sample_weights(self, K: int) -> torch.Tensor:
        if self.coeff_mode == 'dirichlet':
            alpha = torch.full((K,), float(self.coeff_alpha))
            return torch.distributions.Dirichlet(alpha).sample().to(torch.float32).to(self.device)
        elif self.coeff_mode == 'fixed':
            if self.coeff_fixed is None:
                raise ValueError("coeff_mode='fixed' requires coeff_fixed")
            w = self.coeff_fixed
            if w.numel() != K:
                raise ValueError("coeff_fixed length must equal number of objectives")
            # Normalize to simplex if needed
            s = w.sum()
            return (w / s) if s > 0 else torch.full((K,), 1.0 / K).to(self.device)
        elif self.coeff_mode == 'grid':
            if not self.coeff_grid:
                raise ValueError("coeff_mode='grid' requires coeff_grid")
            w = random.choice(self.coeff_grid)
            if w.numel() != K:
                raise ValueError("Each coeff_grid vector must match number of objectives")
            s = w.sum()
            return (w / s) if s > 0 else torch.full((K,), 1.0 / K).to(self.device)
        else:
            raise ValueError(f"Unknown coeff_mode: {self.coeff_mode}")

    def _can_expand(self, node: DTSNode, reserved_children: int = 0) -> bool:
        """Check whether progressive widening permits another child.

        ``reserved_children`` accounts for expansions already scheduled by other
        paths in the current batched traversal but not yet attached to the tree.
        Counting committed and reserved children together makes the capacity
        check atomic at batch granularity.

        Args:
            node: Node whose progressive-widening capacity is being checked.
            reserved_children: Scheduled child creations not yet in
                ``node.children``.

        Returns:
            Whether one more child can be scheduled.
        """
        # Special case for meta-root
        if self._is_meta_root(node):
            effective_visits = max(1, node.visit_count)
            meta_root_alpha = self.progressive_width_alpha
            max_children = math.floor(self.progressive_width_const * (effective_visits ** meta_root_alpha))
            max_children = max(1, max_children)
            return len(node.children) + reserved_children < max_children

        # For normal nodes:
        if node.t <= 0:  # Can't expand terminal nodes
            return False

        if node.t not in self.expand_timesteps:  # Only expand at specified timesteps
            return False

        # Ensure at least 1 for the calculation
        effective_visits = max(1, node.visit_count)

        # Progressive widening: max_children = C * N(node)^alpha
        max_children = math.floor(self.progressive_width_const * (effective_visits ** self.progressive_width_alpha))

        # Ensure at least 1 child is allowed
        max_children = max(1, max_children)

        return len(node.children) + reserved_children < max_children

    def _try_reserve_expansion(
        self,
        node: DTSNode,
        reservations: dict[int, int],
    ) -> bool:
        """Atomically reserve one batched expansion slot for ``node``.

        Args:
            node: Node for which a child will be scheduled.
            reservations: Per-node reservations for the current selection phase.

        Returns:
            Whether a slot was available and reserved.
        """
        node_key = id(node)
        reserved_children = reservations.get(node_key, 0)
        if not self._can_expand(node, reserved_children=reserved_children):
            return False
        reservations[node_key] = reserved_children + 1
        return True

    def _backup(self, node: DTSNode) -> None:
        """Back up temperature-scaled per-objective soft values.

        At each logical branching state, for proposal endpoints ``x_i ~ q``,
        this applies, independently for each objective,
        ``V = tau * (logsumexp(V_i / tau + log(p_i / q_i)) - log(n))``, or,
        when ``self.is_self_normalize`` is True, the self-normalized variant
        ``V = tau * (logsumexp(V_i / tau + log_w_i) - logsumexp(log_w_i))``
        where ``log_w_i = log(p_i / q_i)``. Here ``tau`` is ``temperature``
        (and inverse temperature is ``lambda = 1 / tau``).  The importance
        ratio is deliberately not divided by ``tau`` because it corrects the
        proposal measure rather than changing the reward tilt.

        The likelihood ratio covers the complete rollout to the next logical
        branch endpoint. Physical unary nodes only forward endpoint values.

        Args:
            node: Descendant node from which to update every ancestor.
        """
        # Traverse up the tree
        current = node
        while current.parent is not None:
            parent = current.parent
            
            # Aggregate only over logical endpoints that have vector values.
            if parent.children:
                use_endpoint_importance = (
                    self.use_importance_sampling
                    and self._is_logical_branch_node(parent)
                )
                value_nodes = parent.children
                endpoint_log_weights = None
                if use_endpoint_importance:
                    value_nodes, endpoint_log_weights = self._branch_endpoints(parent)

                child_vecs = []
                log_imp_weights = []
                for child_idx, ch in enumerate(value_nodes):
                    if getattr(ch, 'value_vec', None) is not None and ch.value_vec is not None:
                        v = ch.value_vec
                        if isinstance(v, np.ndarray):
                            v = torch.from_numpy(v).to(self.device)
                        child_vecs.append(v.to(torch.float32))
                        if endpoint_log_weights is not None:
                            log_imp_weights.append(endpoint_log_weights[child_idx])
                if child_vecs:
                    stacked = torch.stack(child_vecs, dim=0)
                    temperature = max(1e-6, float(self.temperature))
                    scaled_values = stacked / temperature
                    log_num_samples = torch.log(
                        torch.tensor(
                            float(stacked.shape[0]),
                            device=stacked.device,
                            dtype=stacked.dtype,
                        )
                    )
                    if use_endpoint_importance and log_imp_weights:
                        log_w = torch.stack(log_imp_weights, dim=0).to(
                            device=stacked.device,
                            dtype=stacked.dtype,
                        )
                        scaled_values = scaled_values + log_w[:, None]
                        if self.is_self_normalize:
                            log_norm = torch.logsumexp(log_w, dim=0)
                        else:
                            log_norm = log_num_samples
                        agg_vec = temperature * (
                            torch.logsumexp(scaled_values, dim=0) - log_norm
                        )
                    else:
                        agg_vec = temperature * (
                            torch.logsumexp(scaled_values, dim=0) - log_num_samples
                        )
                    parent.update_value_vec(agg_vec)
            
            # Move up the tree
            current = parent

    def _update_exploration_probability(self):
        """
        Update the exploration probability based on current NFE.
        Linearly decay from initial_exploration_prob to final_exploration_prob.
        """
        if self.nfes >= self.num_nfes:
            self.current_exploration_prob = self.final_exploration_prob
        else:
            # Linear decay
            decay_rate = (self.initial_exploration_prob - self.final_exploration_prob) / self.num_nfes
            self.current_exploration_prob = self.initial_exploration_prob - decay_rate * self.nfes

    def _simulate_down_full_batched(self, nodes_for_batch_start):
        """
        Batched version of full DTS simulation that processes multiple nodes in parallel.
        Each element in `nodes_for_batch_start` is the starting DTSNode for an independent simulation path.
        Typically, this will be [self.meta_root] * self.batch_size.
        Similar to the original _simulate_down_batched but continues through
        countdown step t=1 to the terminal node t=0 after expansion.
        """
        self._rollout_call_id += 1
        # current_path_nodes[i] is the DTSNode for the i-th simulation path.
        current_path_nodes = list(nodes_for_batch_start) # Make a mutable copy
        initial_timestep_idx = len(self.timesteps)
        # Accumulate all terminal nodes across the entire traversal to batch decode once at the end
        all_terminal_items: list[tuple[int, DTSNode]] = []
        # Per-path scalarization weights (sampled once per traversal path)
        path_weights: dict[int, torch.Tensor] = {}

        while True:
            # 1) Build the list of active paths (countdown indices t > 0)
            active_path_tuples_for_step: list[tuple[int, DTSNode]] = []
            for i in range(len(current_path_nodes)):
                node = current_path_nodes[i]
                if node is not None and node.t > 0:
                    active_path_tuples_for_step.append((i, node))

            if not active_path_tuples_for_step:
                break  # All paths are at terminal t=0.

            if self.enable_diagnostics:
                self._diag_selection_passes += 1

            # 2) Selection-only traversal to frontier; collect units needing diffusion
            # Each unit: { 'path_idx', 'parent', 'x_t', 't' }
            units_by_t: dict[int, list[dict]] = defaultdict(list)
            terminal_paths_map: dict[int, DTSNode] = {}
            expansion_reservations: dict[int, int] = {}

            for path_idx, node in active_path_tuples_for_step:
                # Deepest expansion node passed while descending this path. Used
                # to rescue the batch fill: if the greedy path bottoms out on an
                # already-computed terminal (no diffusion work), we branch a
                # fresh child from here instead.
                last_expandable: Optional[DTSNode] = None
                # Follow existing children until we hit an expansion point or a leaf/terminal
                while True:
                    if node.t <= 0:
                        if (self.force_expand_to_fill_batch
                                and last_expandable is not None):
                            # Batch-fill rescue: the path reached a cached
                            # terminal via pure selection, contributing nothing
                            # to p_sample. Branch a new child from the deepest
                            # expansion node it passed, exceeding the C*N^alpha
                            # cap, so the rollout batch stays full and the
                            # frontier deepens rather than the root widening.
                            self._append_expansion_unit(
                                last_expandable,
                                path_idx,
                                self._is_meta_root(last_expandable),
                                True,
                                path_weights,
                                units_by_t,
                                initial_timestep_idx,
                            )
                        else:
                            terminal_paths_map[path_idx] = node
                        break

                    is_meta = self._is_meta_root(node)
                    is_expansion_node = is_meta or node.t in self.expand_timesteps
                    if is_expansion_node:
                        last_expandable = node
                    reserved = (
                        is_expansion_node
                        and self._try_reserve_expansion(
                            node,
                            expansion_reservations,
                        )
                    )
                    # Batch-fill guarantee (part 1): a path that reaches an
                    # expandable node with no children to select, whose every
                    # progressive-widening slot is already reserved by sibling
                    # paths in this selection pass, would otherwise stall and
                    # shrink the rollout batch. When ``force_expand_to_fill_batch``
                    # is set we expand anyway, exceeding the C*N^alpha cap so the
                    # batch stays full. Nodes with existing children do not force
                    # here -- surplus paths select among them and, if they bottom
                    # out on a cached terminal, are rescued at the terminal branch
                    # above (batch-fill part 2) so the frontier deepens.
                    forced = (
                        self.force_expand_to_fill_batch
                        and is_expansion_node
                        and not reserved
                        and not node.children
                    )
                    if forced:
                        expansion_reservations[id(node)] = (
                            expansion_reservations.get(id(node), 0) + 1
                        )
                    can_expand_node = reserved or forced

                    # Expansion point (meta or normal) -> stop traversal here
                    if can_expand_node:
                        self._append_expansion_unit(
                            node,
                            path_idx,
                            is_meta,
                            forced,
                            path_weights,
                            units_by_t,
                            initial_timestep_idx,
                        )
                        break

                    if is_expansion_node and not node.children:
                        # Reached only when force_expand_to_fill_batch is False.
                        # Other paths in this batch hold every available slot,
                        # but their children are not attached until denoising.
                        # Leave this path at the node; the next selection phase
                        # will use the newly committed children.
                        break

                    # Not an expansion point
                    if node.children:
                        # Selection among existing children only
                        weights_for_path = self._get_selection_weights_for_path(path_idx, node, path_weights)
                        chosen_child = self._select_child_by_value(
                            node,
                            weights=weights_for_path,
                            use_exploration=self.use_exploration,
                        )
                        chosen_child.visit_count += 1
                        node = chosen_child
                        continue
                    else:
                        # Leaf at non-expand timestep -> needs rollout, treat as a unit
                        weights_for_path = self._get_selection_weights_for_path(path_idx, node, path_weights)
                        cond_for_node, cond_key = self._condition_for_preference(weights_for_path)
                        x_t = node.x_t
                        if self.use_dynamic_conditioning or self.condition_dict:
                            cond_dict = cond_for_node if cond_for_node is not None else self.condition_dict
                            cond_key = cond_key if cond_for_node is not None else self._make_cond_key(self.condition_dict)
                            if cond_dict:
                                x_t = self._apply_conditioning(x_t, cond_dict)
                        units_by_t[node.t].append({
                            'path_idx': path_idx,
                            'parent': node,
                            'x_t': x_t,
                            't': node.t,
                            'meta': False,
                            'cond_dict': cond_for_node,
                            'cond_key': cond_key,
                            'weights': weights_for_path,
                        })
                        break

            # Early exit if nothing to roll out (all paths already terminal by traversal)
            if not units_by_t and terminal_paths_map:
                for p_idx, tnode in terminal_paths_map.items():
                    current_path_nodes[p_idx] = tnode
                continue

            # 3) Fill in meta units' x_T by sampling noise in one batch
            if initial_timestep_idx in units_by_t:
                meta_units = [u for u in units_by_t[initial_timestep_idx] if u.get('meta', False)]
                if meta_units:
                    if self.use_dynamic_conditioning or self.use_conditional_model:
                        x_T_list = []
                        for unit in meta_units:
                            cond_dict = unit.get('cond_dict') or self.condition_dict
                            x_T = self.sample_noise(1, properties=cond_dict)
                            x_T = self._apply_conditioning(x_T, cond_dict)
                            x_T_list.append(x_T.to_data_list()[0])
                        x_T_batch = pyg_data.Batch.from_data_list(x_T_list)
                        for i, unit in enumerate(meta_units):
                            unit['x_t'] = pyg_data.Batch.from_data_list(x_T_batch[i:i+1])
                    else:
                        x_T_batch = self.model.sample_prior(len(meta_units))
                        for i, unit in enumerate(meta_units):
                            unit['x_t'] = pyg_data.Batch.from_data_list(x_T_batch[i:i+1])

            # 4) Segmented batched rollout through countdown t=1 to terminal t=0.
            unique_ts = sorted(list(units_by_t.keys()), reverse=True)
            if not unique_ts:
                # Nothing to do
                for p_idx, tnode in terminal_paths_map.items():
                    current_path_nodes[p_idx] = tnode
                continue
            if unique_ts[-1] != 0:
                unique_ts.append(0)

            # Iterate over boundaries [t_cur -> t_next]
            for b in range(len(unique_ts) - 1):
                t_cur = unique_ts[b]
                t_next = unique_ts[b + 1]

                # Pop the current bucket so it cannot be reprocessed later in this call
                units_at_t = units_by_t.pop(t_cur, [])
                # Step units at t_cur down to t_next, one scalar timestep at a time
                while units_at_t and t_cur > t_next:
                    for unit in units_at_t:
                        cond_dict = unit.get('cond_dict') or self.condition_dict
                        if cond_dict:
                            unit['x_t'] = self._apply_conditioning(unit['x_t'], cond_dict)
                    # Batch current x_t
                    x_t_batch = pyg_data.Batch.from_data_list([u['x_t'] for u in units_at_t])
                    start_time = time.time()

                    if self.enable_diagnostics:
                        self._diag_psample_units += len(units_at_t)
                        self._diag_psample_count += 1
                        self._diag_psample_max = max(self._diag_psample_max, len(units_at_t))

                    # Single-timestep UNet call for this t
                    with torch.no_grad():
                        x_tm1_batch, step_info = self.p_sample(x_t_batch, t_cur)

                    step_logp_guided = None
                    step_logp_uncond = None
                    if self.use_importance_sampling:
                        step_logp_guided, step_logp_uncond = (
                            self._validated_step_log_probs(
                                step_info,
                                len(units_at_t),
                            )
                        )


                    # Update each unit: create child node, attach to tree, advance state
                    next_units: list[dict] = []
                    for i, unit in enumerate(units_at_t):
                        parent_node = unit['parent']
                        child_x_tm1 = pyg_data.Batch.from_data_list(x_tm1_batch[i:i+1])
                        edge_logp_guided = None
                        edge_logp_uncond = None
                        if self.use_importance_sampling:
                            edge_logp_guided = float(
                                step_logp_guided[i].detach().cpu().item()
                            )
                            edge_logp_uncond = float(
                                step_logp_uncond[i].detach().cpu().item()
                            )
                        child_log_weight = getattr(parent_node, "log_weight", 0.0)
                        if self.use_importance_sampling and edge_logp_guided is not None and edge_logp_uncond is not None:
                            child_log_weight = float(child_log_weight) + (edge_logp_uncond - edge_logp_guided)
                        child_cond_key = unit.get('cond_key', None)
                        if child_cond_key is None:
                            child_cond_key = getattr(parent_node, "cond_key", None)
                        child_node = DTSNode(
                            child_x_tm1,
                            t_cur - 1,
                            parent=parent_node,
                            edge_logp_guided=edge_logp_guided,
                            edge_logp_uncond=edge_logp_uncond,
                            log_weight=child_log_weight,
                            cond_key=child_cond_key,
                            nfe_created=self.nfes,
                            preference=unit.get('weights'),
                        )
                        parent_node.children.append(child_node)

                        # Free GPU memory for intermediate nodes that won't be
                        # revisited (not at an expand timestep).  We keep x_t
                        # only on expand-timestep nodes (needed for branching)
                        # and terminal nodes (t=0, needed for reward).
                        if (not self._is_meta_root(parent_node)
                                and parent_node.t not in self.expand_timesteps):
                            parent_node.x_t = None

                        # Advance unit
                        unit['parent'] = child_node
                        unit['x_t'] = child_node.x_t
                        unit['t'] = child_node.t

                        if child_node.t <= 0:
                            # Reached terminal; record for later decode+backup
                            terminal_paths_map[unit['path_idx']] = child_node
                        else:
                            next_units.append(unit)

                    # Prepare for next inner step at t_cur - 1
                    units_at_t = next_units
                    t_cur -= 1

                # Move remaining units (now all at t_next) into that bucket
                if units_at_t:
                    if t_next not in units_by_t:
                        units_by_t[t_next] = []
                    units_by_t[t_next].extend(units_at_t)

            # 5) Record terminal nodes and update current path positions; defer decode+backup until after loop
            if terminal_paths_map:
                for path_idx, terminal_node in terminal_paths_map.items():
                    current_path_nodes[path_idx] = terminal_node
                    all_terminal_items.append((path_idx, terminal_node))

            # 6) Persist non-terminal paths at their latest reached nodes so we don't restart from meta-root
            for bucket in units_by_t.values():
                for unit in bucket:
                    p_idx = unit['path_idx']
                    if p_idx not in terminal_paths_map:
                        current_path_nodes[p_idx] = unit['parent']
        # End while (all paths have reached terminal t=0)

        # Single batched reward for all terminal nodes collected across iterations (ChemGraphBatch)
        if all_terminal_items:
            final_xts = [self.model.final_output(node.x_t) for _, node in all_terminal_items]
            materials_batch = pyg_data.Batch.from_data_list(final_xts)
            rewards = []
            for i in range(len(self.reward_fns)):
                reward_i = torch.Tensor(self.reward_fns[i](x=materials_batch)).to(self.device)
                rewards.append(reward_i)
            rewards = torch.stack(rewards, dim=1)
            if rewards.ndim == 1:
                rewards = rewards.unsqueeze(1)
            # Establish number of objectives
            self.num_objectives = rewards.shape[1]
            rewards_norm = self._normalize_rewards(rewards)
            rewards_norm_np = rewards_norm.detach().cpu().numpy()

            # Get pareto metrics
            if self.pareto_metrics:
                hv = self.pareto_metrics.hv(-rewards_norm_np)
                gd_plus = self.pareto_metrics.gd_plus(-rewards_norm_np)
                igd_plus = self.pareto_metrics.igd_plus(-rewards_norm_np)
            else:
                hv = float('nan')
                gd_plus = float('nan')
                igd_plus = float('nan')

            if self.wandb_run is not None:
                max_rewards = rewards_norm_np.max(axis=0)
                mean_rewards = rewards_norm_np.mean(axis=0)
                std_rewards = rewards_norm_np.std(axis=0)
                for k, mx in enumerate(max_rewards):
                    self.wandb_run.log({f"dts/terminal_max_reward_{k}": float(mx)}, step=self.nfes)
                for k, m in enumerate(mean_rewards):
                    self.wandb_run.log({f"dts/terminal_mean_reward_{k}": float(m)}, step=self.nfes)
                for k, s in enumerate(std_rewards):
                    self.wandb_run.log({f"dts/terminal_std_reward_{k}": float(s)}, step=self.nfes)
                # Mean terminal reward under both scalarizations, using each path's own
                # preference vector (the same weights the traversal used for selection).
                # Mirrors `_compute_child_scores`: linear = w . r, Tchebycheff =
                # -max_k w_k * (ideal_k - r_k) with the configured ideal point.
                K = rewards_norm_np.shape[1]
                uniform_w = np.full((K,), 1.0 / K, dtype=np.float32)
                w_rows = []
                for path_idx, _ in all_terminal_items:
                    w = path_weights.get(path_idx, None)
                    if isinstance(w, torch.Tensor):
                        w = w.detach().cpu().numpy()
                    if w is None:
                        w_rows.append(uniform_w)
                        continue
                    w = np.asarray(w, dtype=np.float32).reshape(-1)
                    s = float(w.sum())
                    w_rows.append(w / s if (w.size == K and s > 0) else uniform_w)
                w_np = np.stack(w_rows, axis=0)  # [num_terminal, K]

                lin_scores = (rewards_norm_np * w_np).sum(axis=1)  # [num_terminal]

                ideal_np = None
                if self.tcheby_ideal_values is not None:
                    ideal_arr = np.asarray(self.tcheby_ideal_values, dtype=np.float32).reshape(-1)
                    if ideal_arr.size == K:
                        ideal_np = ideal_arr
                if ideal_np is None:
                    ideal_np = rewards_norm_np.max(axis=0)
                tcheb_scores = -np.max(w_np * (ideal_np[None, :] - rewards_norm_np), axis=1)

                # Same two scalarizations with the preference fixed to the simplex
                # midpoint (uniform 1/K), giving a preference-independent scalar that is
                # comparable across runs with different preference distributions.
                lin_scores_mid = (rewards_norm_np * uniform_w[None, :]).sum(axis=1)
                tcheb_scores_mid = -np.max(
                    uniform_w[None, :] * (ideal_np[None, :] - rewards_norm_np), axis=1
                )

                self.wandb_run.log(
                    {"dts/terminal_scalarized_lin": float(lin_scores.mean())}, step=self.nfes
                )
                self.wandb_run.log(
                    {"dts/terminal_scalarized_tcheb": float(tcheb_scores.mean())}, step=self.nfes
                )
                self.wandb_run.log(
                    {"dts/terminal_scalarized_lin_mid": float(lin_scores_mid.mean())},
                    step=self.nfes,
                )
                self.wandb_run.log(
                    {"dts/terminal_scalarized_tcheb_mid": float(tcheb_scores_mid.mean())},
                    step=self.nfes,
                )
                self.wandb_run.log({f"dts/terminal_hypervolume": float(hv)}, step=self.nfes)
                self.wandb_run.log({f"dts/terminal_gd_plus": float(gd_plus)}, step=self.nfes)
                self.wandb_run.log({f"dts/terminal_igd_plus": float(igd_plus)}, step=self.nfes)
            if self.enable_diagnostics:
                self._log_batch_diagnostics()
            for i, (path_idx, terminal_node) in enumerate(all_terminal_items):
                terminal_node.update_value_vec(rewards_norm[i])
                self._backup(terminal_node)

    def _get_selection_weights_for_path(self, path_idx: int, node: DTSNode, path_weights: dict[int, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Decide which scalarization weights to use for a given traversal path according to
        the configured resampling policy. Maintains root- or traversal-level caches as needed.
        """
        K = self.num_objectives
        if K is None and self.cond_front_value_cols is not None:
            K = len(self.cond_front_value_cols)
        if K is None:
            return None
        if self.use_dynamic_conditioning:
            if path_idx not in path_weights:
                path_weights[path_idx] = self._maybe_sample_weights(K)
            return path_weights[path_idx]
        if self.coeff_resample == 'per_node':
            return self._maybe_sample_weights(K)
        else:  # per_traversal
            if path_idx not in path_weights:
                path_weights[path_idx] = self._maybe_sample_weights(K)
            return path_weights[path_idx]

    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw reward matrix according to `self.reward_norm` policy.
        Returns a float32 tensor with the same shape as rewards.
        """
        if self.reward_norm == 'zscore':
            mean = rewards.mean(dim=0)
            std = rewards.std(dim=0)
            std = torch.clamp(std, min=self.reward_norm_eps)
            return (rewards - mean) / std
        elif self.reward_norm == 'none':
            return rewards.to(torch.float32)
        else:
            raise ValueError(f"Unknown reward_norm: {self.reward_norm}")

    
    @torch.no_grad()
    def initialize_root_nodes(self):
        self.start_time = time.time()
        self.meta_root = self._initialize_meta_root()

    @torch.no_grad()
    def build_search_tree(self, num_nfes=None):
        """
        Perform multiple DTS iterations starting from the meta-root node.
        The meta-root allows dynamic generation of tree roots during the search.
        """
        # Initialize meta-root if not provided
        if self.meta_root is None:
            self.start_time = time.time()
            self.meta_root = self._initialize_meta_root()
        
        # Set up iteration range and possibly wrap with tqdm
        max_nfes = num_nfes
        if self.use_progress_bar:
            pbar = tqdm.tqdm(total=max_nfes, desc="Building search tree")
            pbar.update(self.nfes)
        
        # Main DTS loop
        while self.nfes < max_nfes:
            self.meta_root.visit_count += self.batch_size

            # Update exploration probability
            self._update_exploration_probability()
            
            # Use batched simulation
            self._simulate_down_full_batched([self.meta_root] * self.batch_size)
            
            # Update progress bar or print progress
            if self.use_progress_bar:
                if self.nfes < num_nfes:
                    pbar.update(self.nfes - pbar.n)
                else:
                    pbar.update(num_nfes - pbar.n)
            if self.wandb_run is not None:
                self.wandb_run.log({"dts/nfes": int(self.nfes),
                                    "dts/meta_root_visits": int(self.meta_root.visit_count)}, step=int(self.nfes))
            elif (self.nfes + 1) % (num_nfes // 10) == 0:
                progress = (self.nfes + 1) / num_nfes * 100
                print(f"Tree building progress: {progress:.1f}% (NFE {self.nfes+1}/{num_nfes})")
                sys.stdout.flush()


        if self.use_progress_bar:
            pbar.close()
        
        # # Visualize the tree for debugging
        
        print(f"DTS tree building {self.num_nfes} NFEs in {time.time() - self.start_time:.2f} seconds")


    def _step_meta_nodes_for_sampling(self, nodes_without_children_meta, max_timestep_idx: int) -> dict[int, DTSNode]:
        """
        Advance meta-root nodes by one step T -> T-1 and return mapping path_idx -> next_node.
        """
        path_indices = [idx for idx, _ in nodes_without_children_meta]
        if self.use_dynamic_conditioning or self.use_conditional_model:
            x_T_list = []
            cond_dicts = []
            cond_keys = []
            for _ in path_indices:
                if self.use_dynamic_conditioning:
                    cond_dict, cond_key = self._condition_for_preference()
                else:
                    cond_dict, cond_key = self.condition_dict, self._make_cond_key(self.condition_dict)
                cond_dicts.append(cond_dict)
                cond_keys.append(cond_key)
                x_T = self.sample_noise(1, properties=cond_dict)
                x_T = self._apply_conditioning(x_T, cond_dict)
                x_T_list.append(x_T.to_data_list()[0])
            x_T_batch = pyg_data.Batch.from_data_list(x_T_list)
        else:
            x_T_batch = self.model.sample_prior(len(path_indices))
        t_tensor = torch.tensor([max_timestep_idx] * len(path_indices), device=self.device).long()
        with torch.no_grad():
            x_Tm1_batch, step_info = self.p_sample(x_T_batch, t_tensor)
        step_logp_guided = None
        step_logp_uncond = None
        if self.use_importance_sampling:
            step_logp_guided, step_logp_uncond = self._validated_step_log_probs(
                step_info,
                len(path_indices),
            )
        next_t_idx = max_timestep_idx - 1
        result = {}
        for i, (path_idx, parent_node) in enumerate(nodes_without_children_meta):
            x_Tm1 = pyg_data.Batch.from_data_list(x_Tm1_batch[i:i+1])
            edge_logp_guided = None
            edge_logp_uncond = None
            if self.use_importance_sampling:
                edge_logp_guided = float(
                    step_logp_guided[i].detach().cpu().item()
                )
                edge_logp_uncond = float(
                    step_logp_uncond[i].detach().cpu().item()
                )
            child_log_weight = getattr(parent_node, "log_weight", 0.0)
            if self.use_importance_sampling and edge_logp_guided is not None and edge_logp_uncond is not None:
                child_log_weight = float(child_log_weight) + (edge_logp_uncond - edge_logp_guided)
            cond_key = None
            if self.use_dynamic_conditioning or self.condition_dict:
                cond_key = cond_keys[i] if self.use_dynamic_conditioning else self._make_cond_key(self.condition_dict)
            next_node = DTSNode(
                x_Tm1,
                next_t_idx,
                parent=parent_node,
                timestep_value=None,
                edge_logp_guided=edge_logp_guided,
                edge_logp_uncond=edge_logp_uncond,
                log_weight=child_log_weight,
                cond_key=cond_key,
            )
            result[path_idx] = next_node
        return result

    def _step_normal_leaf_nodes_for_sampling(self, t_idx: int, nodes_without_children_normal) -> dict[int, DTSNode]:
        """
        Advance normal leaf nodes by one step t -> t-1 and return mapping path_idx -> next_node.
        """
        if not nodes_without_children_normal:
            return {}
        normal_nodes = [n for _, n in nodes_without_children_normal]
        path_indices = [idx for idx, _ in nodes_without_children_normal]
        x_t_list = []
        cond_keys = []
        for node in normal_nodes:
            if self.use_dynamic_conditioning:
                cond_dict, cond_key = self._condition_for_preference()
            else:
                cond_dict, cond_key = self.condition_dict, self._make_cond_key(self.condition_dict)
            x_t = self._apply_conditioning(node.x_t, cond_dict) if (self.use_dynamic_conditioning or cond_dict) else node.x_t
            x_t_list.append(x_t)
            cond_keys.append(cond_key)
        x_t_batch = pyg_data.Batch.from_data_list(x_t_list)
        t_tensor = torch.tensor([t_idx] * len(path_indices), device=self.device).long()
        with torch.no_grad():
            x_tm1_batch, step_info = self.p_sample(x_t_batch, t_tensor)
        step_logp_guided = None
        step_logp_uncond = None
        if self.use_importance_sampling:
            step_logp_guided, step_logp_uncond = self._validated_step_log_probs(
                step_info,
                len(path_indices),
            )
        next_t_idx = t_idx - 1
        result = {}
        for i, (path_idx, parent_node) in enumerate(nodes_without_children_normal):  
            x_tm1 = pyg_data.Batch.from_data_list(x_tm1_batch[i:i+1])
            edge_logp_guided = None
            edge_logp_uncond = None
            if self.use_importance_sampling:
                edge_logp_guided = float(
                    step_logp_guided[i].detach().cpu().item()
                )
                edge_logp_uncond = float(
                    step_logp_uncond[i].detach().cpu().item()
                )
            child_log_weight = getattr(parent_node, "log_weight", 0.0)
            if self.use_importance_sampling and edge_logp_guided is not None and edge_logp_uncond is not None:
                child_log_weight = float(child_log_weight) + (edge_logp_uncond - edge_logp_guided)
            next_node = DTSNode(
                x_tm1,
                next_t_idx,
                parent=parent_node,
                timestep_value=None,
                edge_logp_guided=edge_logp_guided,
                edge_logp_uncond=edge_logp_uncond,
                log_weight=child_log_weight,
                cond_key=cond_keys[i] if (self.use_dynamic_conditioning or self.condition_dict) else None,
            )
            result[path_idx] = next_node
        return result

    def _select_children_for_sampling(self, nodes_with_children, weights: Optional[torch.Tensor]) -> dict[int, DTSNode]:
        """
        Select pre-expanded children for nodes that already have children according to scores.
        Returns mapping path_idx -> chosen_child.
        """
        result = {}
        for path_idx, node in nodes_with_children:
            chosen_child = self._select_child_by_value(
                node,
                weights=weights,
                use_exploration=False,
            )
            result[path_idx] = chosen_child
        return result

    def get_all_nodes(self):
        '''
        breadth-first search of all nodes in tree
        '''
        list_of_nodes = []
        current_level = [self.meta_root]
        while current_level:
            next_level = []
            for node in current_level:
                list_of_nodes.append(node)
                next_level.extend(node.children)
            current_level = next_level
        return list_of_nodes
    
    def get_tree_stats(self):
        all_nodes = self.get_all_nodes()
        num_nodes = len(all_nodes)
        num_children_per_level = Counter([(n.t, len(n.children)) for n in all_nodes])
        num_nodes_per_level = Counter([n.t for n in all_nodes])
        return {
            'num_nodes': num_nodes,
            'num_children_per_level': num_children_per_level,
            'num_nodes_per_level': num_nodes_per_level,
        }

