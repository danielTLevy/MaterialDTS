"""
DTS Checkpointing Utilities

Provides functionality to save and restore the complete state of an DTS tree,
including all nodes, their relationships, and the diffusion model state.
"""

import gzip
import math
import pickle
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, asdict
import signal
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DTSCheckpointMetadata:
    """Metadata about the checkpoint"""
    nfes: int
    meta_root_visits: int
    total_nodes: int
    terminal_nodes: int
    exploration_prob: float
    timestamp: str
    checkpoint_version: str = "1.0"


class DTSCheckpointer:
    """
    Handles saving and loading DTS tree state with preemption handling.
    
    The checkpoint includes:
    - All node states (x_t tensors, visit counts, values)
    - Tree structure (parent-child relationships)
    - Sampler state (NFE count, exploration parameters)
    - Model state (if needed)
    """
    
    def __init__(self,
                 checkpoint_dir: str,
                 checkpoint_interval: int = 100,
                 max_checkpoints: int = 3,
                 use_compression: bool = True,
                 compression_level: int = 6,
                 checkpoint_nfes: Optional[list] = None):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            checkpoint_interval: Save every N NFEs (ignored when
                checkpoint_nfes is set)
            max_checkpoints: Maximum number of checkpoints to keep (None = keep all)
            use_compression: Whether to use gzip compression
            compression_level: Gzip compression level (1-9, 6 is good default)
            checkpoint_nfes: Explicit list of NFE values at which to save.
                When provided, exactly one checkpoint is written per target
                (no fixed-interval saving).
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.use_compression = use_compression
        self.compression_level = compression_level
        self.checkpoint_nfes = sorted(checkpoint_nfes) if checkpoint_nfes else None
        self._saved_targets: set = set()
        self.last_checkpoint_nfe = 0

        # Setup preemption signal handler
        self.preemption_triggered = False
        self.preemption_signum: Optional[int] = None
        self._setup_preemption_handler()

    def _setup_preemption_handler(self):
        """Setup signal handler for SLURM preemption (SIGTERM)"""
        def preemption_handler(signum, frame):
            logger.warning(f"Received preemption signal {signum}. Triggering checkpoint...")
            self.preemption_triggered = True
            self.preemption_signum = signum

        # SLURM sends SIGTERM before killing the job
        signal.signal(signal.SIGTERM, preemption_handler)
        signal.signal(signal.SIGUSR1, preemption_handler)  # Also handle SIGUSR1

    def should_checkpoint(self, current_nfe: int, force: bool = False) -> bool:
        """Determine if checkpoint should be saved"""
        if force or self.preemption_triggered:
            return True

        if self.checkpoint_nfes is not None:
            return any(t <= current_nfe and t not in self._saved_targets
                       for t in self.checkpoint_nfes)

        return (current_nfe - self.last_checkpoint_nfe) >= self.checkpoint_interval

    def _mark_targets_saved(self, current_nfe: int):
        """Record which checkpoint_nfes targets were covered by this save."""
        if self.checkpoint_nfes is None:
            return
        for t in self.checkpoint_nfes:
            if t <= current_nfe:
                self._saved_targets.add(t)
    
    def save_checkpoint(self, sampler, force: bool = False, 
                       checkpoint_name: Optional[str] = None) -> Path:
        """
        Save checkpoint with optional compression and automatic cleanup.
        """
        if not self.should_checkpoint(sampler.nfes, force):
            return None
        
        logger.info(f"Saving checkpoint at NFE={sampler.nfes}...")
        
        # Generate checkpoint filename
        if checkpoint_name is None:
            ext = '.pkl.gz' if self.use_compression else '.pkl'
            checkpoint_name = f"dts_checkpoint_nfe_{sampler.nfes}{ext}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Collect all nodes via BFS
        node_data = self._serialize_tree(sampler.meta_root)
        
        # Prepare checkpoint data
        checkpoint_data = self._prepare_checkpoint_data(sampler, node_data)
        
        # Save with or without compression
        if self.use_compression:
            with gzip.open(checkpoint_path, 'wb', compresslevel=self.compression_level) as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Log size
        size_mb = checkpoint_path.stat().st_size / (1024**2)
        logger.info(f"Checkpoint saved: {size_mb:.1f} MB")
        
        if size_mb > 5000:  # 5 GB
            logger.warning(f"⚠️  Large checkpoint: {size_mb:.1f} MB")
        
        # Save metadata JSON
        self._save_metadata_json(checkpoint_data, checkpoint_path)
        
        # Update latest symlink
        self._update_latest_symlink(checkpoint_path)
        
        # Cleanup old checkpoints
        if self.max_checkpoints is not None:
            self._cleanup_old_checkpoints()
        
        self.last_checkpoint_nfe = sampler.nfes
        self._mark_targets_saved(sampler.nfes)
        return checkpoint_path

    def _prepare_checkpoint_data(self, sampler, node_data):
        """Prepare checkpoint data dict."""
        return {
            'metadata': asdict(DTSCheckpointMetadata(
                nfes=sampler.nfes,
                meta_root_visits=sampler.meta_root.visit_count,
                total_nodes=len(node_data['nodes']),
                terminal_nodes=len([n for n in sampler.get_all_nodes() if n.t == 0]),
                exploration_prob=sampler.current_exploration_prob,
                timestamp=datetime.now().isoformat(),
            )),
            'tree_structure': node_data,
            'sampler_state': {
                'nfes': sampler.nfes,
                'current_exploration_prob': sampler.current_exploration_prob,
                'num_nfes': sampler.num_nfes,
                'expand_timesteps': sampler.expand_timesteps,
                'exploration_const': sampler.c_ucb,
                'progressive_width_const': sampler.progressive_width_const,
                'progressive_width_alpha': sampler.progressive_width_alpha,
                'initial_exploration_prob': sampler.initial_exploration_prob,
                'final_exploration_prob': sampler.final_exploration_prob,
                'temperature': sampler.temperature,
                'batch_size': sampler.batch_size,
                'use_rollout': sampler.use_rollout,
                'maximize': sampler.maximize,
                'exploration_type': sampler.exploration_type,
                'use_exploration': sampler.use_exploration,
                # Conditional DTS flag
                'use_conditional_dts': getattr(sampler, 'use_conditional_model', False),
                'use_importance_sampling': getattr(
                    sampler,
                    'use_importance_sampling',
                    False,
                ),
                'use_is_in_selection': getattr(
                    sampler,
                    'use_is_in_selection',
                    True,
                ),
                'use_lattice_is_correction': getattr(
                    sampler,
                    'use_lattice_is_correction',
                    False,
                ),
                'reward_names': sampler.reward_names,
                'num_objectives': sampler.num_objectives,
                'coeff_mode': sampler.coeff_mode,
                'coeff_alpha': sampler.coeff_alpha,
                'coeff_fixed': sampler.coeff_fixed.tolist() if sampler.coeff_fixed is not None else None,
                'coeff_resample': sampler.coeff_resample,
                'reward_norm': sampler.reward_norm,
                'reward_norm_eps': sampler.reward_norm_eps,
            },
        }

    def _save_metadata_json(self, checkpoint_data, checkpoint_path):
        """Save metadata as JSON for easy inspection."""
        metadata_path = self.checkpoint_dir / f"{checkpoint_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_data['metadata'], f, indent=2)

    def _update_latest_symlink(self, checkpoint_path):
        """Update 'latest' symlink to point to most recent checkpoint."""
        if self.use_compression:
            latest_link = self.checkpoint_dir / "latest_checkpoint.pkl.gz"
        else:
            latest_link = self.checkpoint_dir / "latest_checkpoint.pkl"
        
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_path.name)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent max_checkpoints."""
        if self.max_checkpoints is None:
            return
        
        # Find all checkpoint files
        pattern = "*.pkl.gz" if self.use_compression else "*.pkl"
        all_checkpoints = sorted(
            [p for p in self.checkpoint_dir.glob(f"dts_checkpoint_{pattern}")],
            key=lambda p: p.stat().st_mtime
        )
        
        # Remove old checkpoints
        if len(all_checkpoints) > self.max_checkpoints:
            checkpoints_to_remove = all_checkpoints[:-self.max_checkpoints]
            
            for old_checkpoint in checkpoints_to_remove:
                logger.info(f"Removing old checkpoint: {old_checkpoint.name}")
                old_checkpoint.unlink()
                
                # Remove associated metadata JSON
                metadata_file = self.checkpoint_dir / f"{old_checkpoint.stem}_metadata.json"
                if metadata_file.exists():
                    metadata_file.unlink()
    
    def _serialize_tree(self, root_node) -> Dict[str, Any]:
        """
        Serialize the entire tree structure.
        
        Returns a dictionary containing:
        - nodes: List of serialized node data
        - edges: Parent-child relationships (as indices)
        - node_map: Mapping from node id to list index
        """
        nodes = []
        edges = []
        node_to_idx = {}
        
        # BFS traversal
        queue = [root_node]
        idx = 0
        
        while queue:
            node = queue.pop(0)
            node_id = id(node)
            
            # Skip if already processed
            if node_id in node_to_idx:
                continue
            
            # Add to mapping
            node_to_idx[node_id] = idx
            
            # Serialize node data
            node_dict = {
                'idx': idx,
                'node_id': node_id,
                't': node.t,
                'timestep_value': node.timestep_value,
                'visit_count': node.visit_count,
                'value_est': float(node.value_est) if node.value_est is not None else None,
                'x_t': node.x_t.to_data_list() if node.x_t is not None else None,  # ChemGraphBatch serialization
                'parent_id': id(node.parent) if node.parent is not None else None,
                'num_children': len(node.children),
                'nfe_created': node.nfe_created,
            }
            
            # Store value vector if present
            if hasattr(node, 'value_vec') and node.value_vec is not None:
                node_dict['value_vec'] = node.value_vec.cpu().numpy().tolist()

            # Store conditional DTS fields if present (dts_materials_multi_cond nodes)
            for attr in ('edge_logp_guided', 'edge_logp_uncond', 'log_weight', 'cond_key', 'preference'):
                if hasattr(node, attr):
                    node_dict[attr] = getattr(node, attr)

            nodes.append(node_dict)
            
            # Add children to queue and record edges
            for child in node.children:
                child_id = id(child)
                queue.append(child)
                # Will add edge after child is processed
            
            idx += 1
        
        # Build edges after all nodes are indexed
        for node_dict in nodes:
            if node_dict['parent_id'] is not None:
                parent_idx = node_to_idx[node_dict['parent_id']]
                child_idx = node_dict['idx']
                edges.append((parent_idx, child_idx))
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_map': {str(k): v for k, v in node_to_idx.items()},
            'root_idx': 0,
        }
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> dict:
        """Load checkpoint with automatic decompression."""
        if checkpoint_path is None:
            # Try compressed version first, then uncompressed
            latest_compressed = self.checkpoint_dir / "latest_checkpoint.pkl.gz"
            latest_uncompressed = self.checkpoint_dir / "latest_checkpoint.pkl"
            
            if latest_compressed.exists():
                checkpoint_path = latest_compressed
            elif latest_uncompressed.exists():
                checkpoint_path = latest_uncompressed
            else:
                raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading DTS checkpoint from {checkpoint_path}...")
        
        # Load with or without decompression
        if checkpoint_path.suffix == '.gz':
            with gzip.open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        else:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        
        logger.info(f"Loaded checkpoint: NFE={checkpoint_data['metadata']['nfes']}, "
                   f"Total nodes={checkpoint_data['metadata']['total_nodes']}")
        
        return checkpoint_data
    
    def restore_tree(self, checkpoint_data: Dict[str, Any], sampler) -> None:
        """
        Restore DTS tree structure from checkpoint data.

        Args:
            checkpoint_data: Data loaded from checkpoint
            sampler: MaterialDTSDiffusionSampler instance to restore into
        """
        import torch_geometric.data as pyg_data

        sampler_state = checkpoint_data.get('sampler_state', {})
        checkpoint_uses_conditional = sampler_state.get(
            'use_conditional_dts',
            False,
        )
        current_uses_conditional = getattr(
            sampler,
            'use_conditional_model',
            False,
        )
        if checkpoint_uses_conditional != current_uses_conditional:
            raise RuntimeError(
                "Cannot resume a checkpoint with a different conditional-DTS "
                "mode. Start a fresh tree."
            )

        current_uses_importance = getattr(
            sampler,
            'use_importance_sampling',
            False,
        )
        checkpoint_uses_importance = sampler_state.get(
            'use_importance_sampling'
        )
        if bool(checkpoint_uses_importance) != bool(current_uses_importance):
            raise RuntimeError(
                "Cannot resume a checkpoint with a different "
                "importance-sampling mode. Start a fresh tree."
            )

        if current_uses_conditional:
            checkpoint_temperature = sampler_state.get('temperature')
            current_temperature = float(sampler.temperature)
            if (
                checkpoint_temperature is None
                or not math.isclose(
                    float(checkpoint_temperature),
                    current_temperature,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError(
                    "Cannot resume a conditional DTS tree with a different "
                    "temperature because its stored soft values use the original "
                    "temperature. Start a fresh tree."
                )

            checkpoint_expand_timesteps = sampler_state.get('expand_timesteps')
            current_expand_timesteps = list(sampler.expand_timesteps)
            if (
                checkpoint_expand_timesteps is None
                or list(checkpoint_expand_timesteps) != current_expand_timesteps
            ):
                raise RuntimeError(
                    "Cannot resume a conditional DTS tree with different "
                    "expand_timesteps because existing rollout segments use the "
                    "original expansion schedule. Start a fresh tree."
                )

        use_cond = sampler_state['use_conditional_dts']
        if use_cond:
            from src.core.sampling.dts_materials_multi_cond import DTSNode, MaterialMetaRootNode
        else:
            from src.core.sampling.dts_materials_multi import DTSNode, MaterialMetaRootNode

        logger.info("Restoring DTS tree structure...")

        tree_data = checkpoint_data['tree_structure']
        nodes_data = tree_data['nodes']
        edges = tree_data['edges']

        # First pass: Create all node objects
        idx_to_node = {}
        for node_dict in nodes_data:
            idx = node_dict['idx']

            # Reconstruct x_t (ChemGraphBatch)
            if node_dict['x_t'] is not None:
                from mattergen.common.data.chemgraph import ChemGraph
                chem_graphs = [ChemGraph.from_dict(d) for d in node_dict['x_t']]
                x_t = pyg_data.Batch.from_data_list(chem_graphs)
            else:
                x_t = None

            # Create node (without parent reference yet)
            if idx == 0:  # Root node
                node = MaterialMetaRootNode(sampler, device=sampler.device)
                node.x_t = x_t
            else:
                # Build kwargs for conditional fields when using cond DTSNode
                extra_kwargs = {}
                if use_cond:
                    for attr in ('edge_logp_guided', 'edge_logp_uncond', 'log_weight', 'cond_key', 'preference'):
                        if attr in node_dict:
                            extra_kwargs[attr] = node_dict[attr]

                node = DTSNode(
                    x_t=x_t,
                    t=node_dict['t'],
                    parent=None,  # Will be set in second pass
                    timestep_value=node_dict['timestep_value'],
                    nfe_created=node_dict.get('nfe_created', 0),
                    **extra_kwargs,
                )

            # Restore state
            node.visit_count = node_dict['visit_count']
            node.value_est = node_dict['value_est'] if node_dict['value_est'] is not None else 0.0

            # Restore value vector if present
            if 'value_vec' in node_dict and node_dict['value_vec'] is not None:
                node.value_vec = torch.tensor(node_dict['value_vec'], dtype=torch.float32).to(sampler.device)

            idx_to_node[idx] = node
        
        # Second pass: Connect parent-child relationships
        for parent_idx, child_idx in edges:
            parent_node = idx_to_node[parent_idx]
            child_node = idx_to_node[child_idx]
            
            child_node.parent = parent_node
            parent_node.children.append(child_node)
        
        # Restore sampler state
        sampler_state = checkpoint_data['sampler_state']
        sampler.nfes = sampler_state['nfes']
        sampler.current_exploration_prob = sampler_state['current_exploration_prob']
        sampler.meta_root = idx_to_node[0]

        # Restore multi-objective state
        sampler.num_objectives = sampler_state.get('num_objectives')
        if sampler_state.get('coeff_fixed') is not None:
            sampler.coeff_fixed = torch.tensor(sampler_state['coeff_fixed'], dtype=torch.float32)

        # Initialize start_time for timing resumed tree building
        import time
        sampler.start_time = time.time()

        logger.info(f"Tree restored: {len(idx_to_node)} nodes, NFE={sampler.nfes}")


def checkpoint_aware_training(sampler, checkpoint_dir: str,
                              target_nfes: int,
                              checkpoint_interval: int = 1000,
                              max_checkpoints: int = 3,
                              use_compression: bool = True,
                              compression_level: int = 6,
                              load_dir: Optional[str] = None,
                              wandb_run=None,
                              checkpoint_nfes: Optional[list] = None) -> None:
    """
    Run DTS with periodic checkpointing (compression + cleanup).

    Args:
        sampler: MaterialDTSDiffusionSampler instance
        checkpoint_dir: Directory for checkpoints (save location)
        target_nfes: Target number of NFEs
        checkpoint_interval: NFEs between checkpoints (fixed cadence). Ignored
            when checkpoint_nfes is set.
        max_checkpoints: Maximum checkpoints to keep (None = keep all)
        use_compression: Use gzip compression (reduces size by ~70%)
        compression_level: Gzip level (1-9, 6 is good default)
        load_dir: If set, load initial tree from here instead of checkpoint_dir (for branching)
        checkpoint_nfes: Optional explicit list of NFE values at which to
            save. When set, the loop advances directly to the next target
            instead of using a fixed cadence — exactly one checkpoint per
            target value ends up on disk.
    """

    # In list mode we want all target checkpoints preserved; auto-disable
    # the fixed-count cleanup so the earliest saves survive to the end.
    if checkpoint_nfes is not None and max_checkpoints is not None:
        logger.info("checkpoint_nfes set; disabling max_checkpoints cleanup to preserve all targets.")
        max_checkpoints = None

    # Checkpointer for SAVING always uses checkpoint_dir
    checkpointer = DTSCheckpointer(
        checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        max_checkpoints=max_checkpoints,
        use_compression=use_compression,
        compression_level=compression_level,
        checkpoint_nfes=checkpoint_nfes,
    )

    # Loader uses load_dir when branching, otherwise checkpoint_dir.
    # Construct after checkpointer so its __init__ overwrites the signal handler,
    # then immediately re-register checkpointer's handler so the loop sees the right flag.
    load_path = load_dir if load_dir else checkpoint_dir
    loader = DTSCheckpointer(load_path, checkpoint_interval=checkpoint_interval,
                             use_compression=use_compression)
    # Re-register so signals set checkpointer.preemption_triggered, not loader's.
    checkpointer._setup_preemption_handler()

    # Wrap the signal handler so wandb is notified immediately on preemption,
    # before SLURM's grace period expires (the loop body may not run in time).
    if wandb_run is not None:
        def _preemption_with_wandb(signum, frame):
            logger.warning(f"Received preemption signal {signum}. Notifying wandb immediately...")
            checkpointer.preemption_triggered = True
            checkpointer.preemption_signum = signum
            try:
                wandb_run.mark_preempting()
            except Exception as e:
                logger.warning(f"Failed to mark wandb run as preempting: {e}")

        signal.signal(signal.SIGTERM, _preemption_with_wandb)
        signal.signal(signal.SIGUSR1, _preemption_with_wandb)

    # Try to load existing checkpoint
    try:
        checkpoint_data = loader.load_checkpoint()
        logger.info(f"Resuming from {'load_dir' if load_dir else 'checkpoint_dir'}...")
        checkpointer.restore_tree(checkpoint_data, sampler)
    except FileNotFoundError:
        logger.info("No existing checkpoint found, starting fresh...")
        sampler.initialize_root_nodes()

    # Keep only targets that lie ahead of the (possibly restored) current NFE.
    remaining_targets = (
        [n for n in checkpointer.checkpoint_nfes if n > sampler.nfes]
        if checkpointer.checkpoint_nfes is not None else None
    )

    # Build tree with periodic checkpointing
    while sampler.nfes < target_nfes:
        # Check for preemption
        if checkpointer.preemption_triggered:
            logger.warning("Preemption detected! Saving checkpoint and exiting...")
            checkpointer.save_checkpoint(sampler, force=True)
            if wandb_run is not None:
                wandb_run.mark_preempting()
            signum = checkpointer.preemption_signum or signal.SIGTERM
            sys.exit(128 + signum)

        # Decide batch size: either aim for the next explicit target, or
        # take a fixed-interval step.
        if remaining_targets:
            next_target = remaining_targets[0]
            batch_nfes = min(next_target, target_nfes) - sampler.nfes
        else:
            batch_nfes = min(checkpoint_interval, target_nfes - sampler.nfes)

        sampler.build_search_tree(num_nfes=sampler.nfes + batch_nfes)

        # Save checkpoint
        checkpointer.save_checkpoint(sampler)

        # Drop targets we've now passed (they were marked saved above).
        if remaining_targets:
            remaining_targets = [n for n in remaining_targets if n > sampler.nfes]

    # Final checkpoint — skip when we already saved at target_nfes via the
    # explicit-targets path to avoid a redundant duplicate.
    if checkpoint_nfes is not None and target_nfes in checkpointer._saved_targets:
        logger.info(f"Final NFE {target_nfes} already in checkpoint_nfes; skipping redundant final save.")
    else:
        checkpoint_name = "final_checkpoint.pkl.gz" if use_compression else "final_checkpoint.pkl"
        checkpointer.save_checkpoint(sampler, force=True, checkpoint_name=checkpoint_name)

    logger.info(f"DTS training complete! Final NFE: {sampler.nfes}")

