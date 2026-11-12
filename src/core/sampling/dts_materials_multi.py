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
from typing import Callable, List, Optional
import torch_geometric.data as pyg_data

logging.getLogger("torch_tensorrt.dynamo.conversion._TRTInterpreter").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Both operands of the binary elementwise op .* are constant")



class DTSNode:
    """
    Each node represents a latent state x_t at diffusion timestep t.
    Children are possible next states x_{t-1}.
    """

    def __init__(self, x_t, t, parent=None, timestep_value=None, nfe_created: int = 0):
        """
        Args:
          x_t (torch.Tensor): The noisy latent at diffusion timestep t.
          t (int): The current diffusion timestep index.
          parent (DTSNode or None): The parent node. None if root.
          timestep_value (float or None): The actual scheduler timestep value if known.
          nfe_created (int): NFE count at the time this node was created.
        """
        self.x_t = x_t          # ChemGraphBatch
        self.t = t               # This is the timestep index
        self.timestep_value = timestep_value  # This is the actual scheduler timestep value
        self.parent = parent
        self.children = []      # list of child nodes
        self.visit_count = 1    # N(x_t) in the algorithm
        self.value_est = 0.0    # Optional scalar estimate for visualization only
        self.value_vec = None   # torch.Tensor[K], log-domain per-objective values
        self.nfe_created = nfe_created  # NFE count when this node was created

        
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
                 # Materials specific
                 wandb_run=None,
                 pareto_metrics=None):
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

    def p_sample(self, x_t, t, predict_x0=False, eta=None, print_batch=False):
        """
        Perform one step of the diffusion process using the model's methods.
        Args:
            x_t (ChemGraphBatch): Current noisy data graph.
            t_idx (Union[int, torch.Tensor, list]): DTS timestep *index* (0 to T-1).
            predict_x0 (bool): If True, use ddim_denoise to predict x0.
        Returns:
            Tensor: Either x_{t-1} or predicted x0.
        """
        batch_size = len(x_t)
        if print_batch:
            print("BATCH SIZE: ", str(batch_size), ' TIMESTEP: ', str(t))
        if eta is None:
            eta = self.eta

        self.nfes += batch_size

        # Call method from model
        with torch.no_grad():
            if predict_x0:
                raise NotImplementedError("predict_x0 not implemented for mattergen model")
            else:
                return self.model.p_sample(x_t, t, predict_x0, eta, print_batch)

        return pred
            
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
                
            # Otherwise, select based on value (exploitation)
            values = self._compute_child_scores(node.children, weights)
            
            # Sample a child proportional to value using torch operations
            if self.maximize:
                idx = torch.argmax(values).item()
            else:
                temp = max(1e-6, float(self.temperature))
                logits = values / temp
                logits = logits - logits.max()
                probs = torch.softmax(logits, dim=0)
                idx = torch.multinomial(probs, 1).item()
            return node.children[idx]
        
        elif self.exploration_type == 'ucb':
            # UCB exploration
            if not use_exploration:
                # If exploration is disabled, just use values
                values = self._compute_child_scores(node.children, weights)
                temp = max(1e-6, float(self.temperature))
                logits = values / temp
                logits = logits - logits.max()
                probs = torch.softmax(logits, dim=0)
                idx = torch.multinomial(probs, 1).item()
                return node.children[idx]
            
            # Calculate UCB scores for each child
            parent_visits = max(1, node.visit_count)  # Avoid division by zero
            
            # Vectorized UCB calculation
            child_visits = torch.tensor(np.array([max(1, child.visit_count) for child in node.children])).to(self.device)
            child_values = self._compute_child_scores(node.children, weights)
            temp = max(1e-6, float(self.temperature))
            logits = child_values / temp
            logits = logits - logits.max()
            
            # UCB formula: value + c * sqrt(ln(parent_visits) / child_visits)
            exploration_term = self.c_ucb * torch.sqrt(torch.log(torch.tensor(parent_visits)) / child_visits)
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

    def _can_expand(self, node):
        """
        Check if a node can be expanded based on progressive widening criteria.
        Also handles the special case of the meta-root node.
        """
        # Special case for meta-root
        if self._is_meta_root(node):
            effective_visits = max(1, node.visit_count)
            meta_root_alpha = self.progressive_width_alpha
            max_children = math.floor(self.progressive_width_const * (effective_visits ** meta_root_alpha))
            return len(node.children) < max_children
        
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
        
        return len(node.children) < max_children

    def _backup(self, node):
        """
        Vectorized backup: for each ancestor, set value_vec to log-sum-exp of children's value_vec elementwise.
        """
        # Traverse up the tree
        current = node
        while current.parent is not None:
            parent = current.parent
            
            # Aggregate only over children that have vector values
            if parent.children:
                child_vecs = []
                for ch in parent.children:
                    if getattr(ch, 'value_vec', None) is not None and ch.value_vec is not None:
                        v = ch.value_vec
                        if isinstance(v, np.ndarray):
                            v = torch.from_numpy(v).to(self.device)
                        child_vecs.append(v.to(torch.float32))
                if child_vecs:
                    stacked = torch.stack(child_vecs, dim=0)
                    # Log-sum-exp along children dimension, per objective
                    tau = max(1e-6, float(self.temperature))
                    agg_vec = tau * (torch.logsumexp(stacked / tau, dim=0) - torch.log(torch.tensor(len(parent.children)).to(self.device)))
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
        Similar to the original _simulate_down_batched but continues rollout to t=0 after expansion.
        """
        self._rollout_call_id += 1
        # current_path_nodes[i] is the DTSNode for the i-th simulation path.
        current_path_nodes = list(nodes_for_batch_start) # Make a mutable copy
        max_timestep_idx = len(self.timesteps) - 1
        # Accumulate all terminal nodes across the entire traversal to batch decode once at the end
        all_terminal_items: list[tuple[int, DTSNode]] = []
        # Per-path scalarization weights (sampled according to policy)
        path_weights: dict[int, torch.Tensor] = {}

        while True:
            # 1) Build the list of active paths (t > 0)
            active_path_tuples_for_step: list[tuple[int, DTSNode]] = []
            for i in range(len(current_path_nodes)):
                node = current_path_nodes[i]
                if node is not None and node.t > 0:
                    active_path_tuples_for_step.append((i, node))

            if not active_path_tuples_for_step:
                break  # All paths are at t <= 0

            # 2) Selection-only traversal to frontier; collect units needing diffusion
            # Each unit: { 'path_idx', 'parent', 'x_t', 't' }
            units_by_t: dict[int, list[dict]] = defaultdict(list)
            terminal_paths_map: dict[int, DTSNode] = {}

            for path_idx, node in active_path_tuples_for_step:
                # Follow existing children until we hit an expansion point or a leaf/terminal
                while True:
                    if node.t <= 0:
                        terminal_paths_map[path_idx] = node
                        break

                    is_meta = self._is_meta_root(node)
                    can_expand_node = self._can_expand(node)

                    # Expansion point (meta or normal) -> stop traversal here
                    if (is_meta and can_expand_node) or (not is_meta and node.t in self.expand_timesteps and can_expand_node):
                        # Create a processing unit for this path
                        if is_meta:
                            # Meta expansion starts from x_T at t = max_timestep_idx
                            # We'll sample actual x_T for all meta units together below; for now, mark placeholder
                            units_by_t[max_timestep_idx].append({
                                'path_idx': path_idx,
                                'parent': node,  # meta root instance
                                'x_t': None,     # to be filled with sampled noise
                                't': max_timestep_idx,
                                'meta': True,
                            })
                        else:
                            units_by_t[node.t].append({
                                'path_idx': path_idx,
                                'parent': node,
                                'x_t': node.x_t,
                                't': node.t,
                                'meta': False,
                            })
                        break

                    # Not an expansion point
                    if node.children:
                        # Selection among existing children only
                        # Determine weights per resample policy
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
                        units_by_t[node.t].append({
                            'path_idx': path_idx,
                            'parent': node,
                            'x_t': node.x_t,
                            't': node.t,
                            'meta': False,
                        })
                        break

            # Early exit if nothing to roll out (all paths already terminal by traversal)
            if not units_by_t and terminal_paths_map:
                for p_idx, tnode in terminal_paths_map.items():
                    current_path_nodes[p_idx] = tnode
                continue

            # 3) Fill in meta units' x_T by sampling noise in one batch
            if max_timestep_idx in units_by_t:
                meta_units = [u for u in units_by_t[max_timestep_idx] if u.get('meta', False)]
                if meta_units:
                    x_T_batch = self.model.sample_prior(len(meta_units))
                    for i, unit in enumerate(meta_units):
                        unit['x_t'] = pyg_data.Batch.from_data_list(x_T_batch[i:i+1])

            # 4) Segmented batched rollout across unique timesteps down to t=0
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
                    # Batch current x_t
                    x_t_batch = pyg_data.Batch.from_data_list([u['x_t'] for u in units_at_t])
                    start_time = time.time()

                    # Single-timestep UNet call for this t
                    with torch.no_grad():
                        x_tm1_batch = self.p_sample(x_t_batch, t_cur)


                    # Update each unit: create child node, attach to tree, advance state
                    next_units: list[dict] = []
                    for i, unit in enumerate(units_at_t):
                        parent_node = unit['parent']
                        child_x_tm1 = pyg_data.Batch.from_data_list(x_tm1_batch[i:i+1])
                        child_node = DTSNode(child_x_tm1, t_cur - 1, parent=parent_node, nfe_created=self.nfes)
                        parent_node.children.append(child_node)

                        # Free GPU memory for intermediate nodes that won't be
                        # revisited (not at an expand timestep).  We keep x_t
                        # only on expand-timestep nodes (needed for branching)
                        # and terminal nodes (t<=0, needed for reward).
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
        # End while (all paths have reached t <= 0)

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
                self.wandb_run.log({f"dts/terminal_hypervolume": float(hv)}, step=self.nfes)
                self.wandb_run.log({f"dts/terminal_gd_plus": float(gd_plus)}, step=self.nfes)
                self.wandb_run.log({f"dts/terminal_igd_plus": float(igd_plus)}, step=self.nfes)
            for i, (path_idx, terminal_node) in enumerate(all_terminal_items):
                terminal_node.update_value_vec(rewards_norm[i])
                self._backup(terminal_node)

    def _get_selection_weights_for_path(self, path_idx: int, node: DTSNode, path_weights: dict[int, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Decide which scalarization weights to use for a given traversal path according to
        the configured resampling policy. Maintains root- or traversal-level caches as needed.
        """
        K = self.num_objectives
        if K is None:
            return None
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
        self.start_time = time.time()
        # Initialize meta-root if not provided
        if self.meta_root is None:
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
        x_T_batch = self.model.sample_prior(len(path_indices))
        t_tensor = torch.tensor([max_timestep_idx] * len(path_indices), device=self.device).long()
        with torch.no_grad():
            x_Tm1_batch = self.p_sample(x_T_batch, t_tensor)
        next_t_idx = max_timestep_idx - 1
        next_t_value = self.get_timestep_from_index(next_t_idx)
        result = {}
        for i, (path_idx, parent_node) in enumerate(nodes_without_children_meta):
            x_Tm1 = pyg_data.Batch.from_data_list(x_Tm1_batch[i:i+1])
            next_node = DTSNode(x_Tm1, next_t_idx, parent=parent_node, timestep_value=next_t_value)
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
        x_t_batch = pyg_data.Batch.from_data_list([node.x_t for node in normal_nodes])
        t_tensor = torch.tensor([t_idx] * len(path_indices), device=self.device).long()
        with torch.no_grad():
            x_tm1_batch = self.p_sample(x_t_batch, t_tensor)
        next_t_idx = t_idx - 1
        next_t_value = self.get_timestep_from_index(next_t_idx)
        result = {}
        for i, (path_idx, parent_node) in enumerate(nodes_without_children_normal):  
            x_tm1 = pyg_data.Batch.from_data_list(x_tm1_batch[i:i+1])
            next_node = DTSNode(x_tm1, next_t_idx, parent=parent_node, timestep_value=next_t_value)
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
    
