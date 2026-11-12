from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union


import torch
import torch_geometric.data as pyg_data


import numpy as np

from mattergen.common.data.chemgraph import ChemGraphBatch


class DiffusionModelAdapter(ABC):
    """
    Adapter interface that wraps a concrete diffusion model (MatterGen, DiffCSP, etc.)
    and exposes a small, stable API that the DTS code consumes.

    Implementations must ensure device consistency (i.e., model and returned tensors
    should be placed on the requested device) and must provide conversion to the
    PyG/ChemGraph-style objects the DTS sampler expects.

    Recommended return types:
      - Per-sample data = torch_geometric.data.Data (or subclass like ChemGraph)
      - Batched outputs may be returned as a Python list[Data] (preferred) or
        a torch_geometric.data.Batch. DTS expects to be able to do
        `pyg_data.Batch.from_data_list(x_list)` and also slice/list-slice
        the returned object, therefore list[Data] is the safest choice.
    """

    @property
    def N(self) -> int:
        return self._N

    @abstractmethod
    def to(self, device: Union[str, torch.device]) -> DiffusionModelAdapter:
        """Move model parameters/state to `device`. Return self for chaining.

        Must ensure subsequent sampling calls return tensors on `device`.
        """

    @abstractmethod
    def sample_prior(
        self,
        num_samples: int = 1,
        conditioning: Optional[Any] = None,
        mask: Optional[Any] = None,
        step_lr: Optional[float] = None,
    ) -> List[pyg_data.Data]:
        """
        Produce `num_samples` samples from the prior at timestep T (i.e. x_T).

        Args:
            num_samples: number of independent prior samples to produce.
            conditioning: optional conditioning data (library-specific).
            mask: optional mask or condition tensor.
            step_lr: optional step size / guidance parameter used by some models.

        Returns:
            A list of per-sample `torch_geometric.data.Data` objects (ChemGraph or Data).

        Notes:
            - Returning a plain list[Data] is recommended because DTS slices
              returned values by index and then converts them with
              `pyg_data.Batch.from_data_list`.
        """

    @abstractmethod
    def p_sample(
        self,
        x_t: Union[pyg_data.Batch, List[pyg_data.Data], ChemGraphBatch],
        t: Union[int, torch.Tensor],
        predict_x0: bool = False,
        eta: Optional[float] = None,
        step_lr: Optional[float] = None,
    ) -> Tuple[Union[List[pyg_data.Data], pyg_data.Batch], Optional[Any]]:
        """
        Perform one denoising step: x_t -> x_{t-1} for a batch of inputs.

        Args:
            x_t: Batched noisy inputs. May be a PyG Batch, a ChemGraphBatch, or a
                 Python list of Data objects.
            t: DTS timestep index (integer or tensor). The adapter is responsible
               for mapping DTS's `t` semantics to the underlying model's index
               convention if they differ (e.g., reversed indexing).
            predict_x0: If True, return a predicted x0 when supported (may raise
                NotImplementedError otherwise).
            eta: optional stochasticity parameter (ddim/ddpm style).
            step_lr: optional step-size / guidance value for models that use it.

        Returns:
            (x_tm1_batch, aux) where x_tm1_batch is a list[Data] or a PyG Batch of
            per-sample denoised objects. `aux` is optional model-specific info
            (trajectory dict, diagnostics) and may be None.
        """


    @abstractmethod
    def final_output(self, x):
        pass


from mattergen.common.data.condition_factory import get_number_of_atoms_condition_loader
from mattergen.diffusion.sampling.pc_sampler import _sample_prior
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch


class MattergenAdapter(DiffusionModelAdapter):
    """Adapter for wrapping an existing MatterGen sampler object.
    """

    def __init__(self, model: Any, device: Union[str, torch.device] = "cpu", num_atoms_loader=None, condition_dict=None):
        self.model = model
        # Force eval mode: DTS calls _denoise_one_step directly, bypassing
        # PredictorCorrector._sample_maybe_record which normally calls .eval().
        # In train mode the denoiser skips element_mask_func, allowing
        # disallowed elements (e.g. U, Pa) to be sampled.
        diffusion_module = getattr(model, '_diffusion_module', None)
        if isinstance(diffusion_module, torch.nn.Module):
            diffusion_module.eval()
        if hasattr(model, '_multi_corruption'):
            self._multi_corruption = model._multi_corruption
        self.device = torch.device(device)
        self._N = model.N
        self.num_atoms_loader = num_atoms_loader
        self.condition_dict = condition_dict

    def set_num_atoms_loader(self, num_atoms_loader):
        self.num_atoms_loader = num_atoms_loader

    def set_condition_dict(self, condition_dict):
        self.condition_dict = condition_dict


    def to(self, device: Union[str, torch.device]) -> "MattergenAdapter":
        self.device = torch.device(device)
        if hasattr(self.model, '_diffusion_module'):
            self.model._diffusion_module.to(self.device)
        elif hasattr(self.model, 'diffusion_module'):
            self.model.diffusion_module.to(self.device)
        elif hasattr(self.model, 'to'):
            self.model.to(self.device)
        return self


    def _data_list_to_chemgraph_batch(self, data_list):
        """Convert list of ChemGraph objects back to ChemGraphBatch for model input."""
        
        # Use PyG's Batch.from_data_list with ChemGraph as base class
        # This creates a proper ChemGraphBatch with get_batch_size() method
        batch = pyg_data.Batch.from_data_list(data_list, follow_batch=['pos', 'atomic_numbers'])
        
        # Move to device
        batch = batch.to(self.device)
        
        return batch

    def sample_prior(self, batch_size=1):
        num_atoms_loader = get_number_of_atoms_condition_loader(num_atoms_distribution='ALEX_MP_20',
                                                                     num_samples=batch_size,
                                                                     batch_size=batch_size,
                                                                     shuffle=True,
                                                                     properties=self.condition_dict)
        conditioning_data, mask = next(iter(num_atoms_loader))
        return _sample_prior(self.model._multi_corruption, conditioning_data, mask).to(self.device)

    
    def p_sample(
        self,
        x_t,  # Can be list of Data or ChemGraphBatch
        t: Union[int, torch.Tensor],
        predict_x0: bool = False,
        eta: Optional[float] = None,
        step_lr: Optional[float] = None,
    ):
        # Convert list of Data to ChemGraphBatch if needed

        if isinstance(x_t, list):
            x_t = self._data_list_to_chemgraph_batch(x_t)
        timestep_i = self.N - t
        
        
        if t > 0:
            pred, _, _ = self.model._denoise_one_step(x_t, mask={}, timestep_i=timestep_i, record=False)
        else:
            _, pred, _ = self.model._denoise_one_step(x_t, mask={}, timestep_i=timestep_i, record=False)
        return pred

    def final_output(self, x):
        return x

    def _denoise_one_step(self, *args, **kwargs):
        return self.model._denoise_one_step(*args, **kwargs)

    def _denoise_one_step_with_logp_pair(self, *args, **kwargs):
        return self.model._denoise_one_step_with_logp_pair(*args, **kwargs)

from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset

from src.models import diffcsp_ext

train_dist = {
    'mp_20' : [0.0,
            0.0021742334905660377,
            0.021079009433962265,
            0.019826061320754717,
            0.15271226415094338,
            0.047132959905660375,
            0.08464770047169812,
            0.021079009433962265,
            0.07808814858490566,
            0.03434551886792453,
            0.0972877358490566,
            0.013303360849056603,
            0.09669811320754718,
            0.02155807783018868,
            0.06522700471698113,
            0.014372051886792452,
            0.06703272405660378,
            0.00972877358490566,
            0.053176591981132074,
            0.010576356132075472,
            0.08995430424528301]
}


class DiffCSPChemGraph(Data):
    def __cat__dim__(self, key, value, *args, **kwargs):
        if key == 'lengths' or key == 'angles' or key == 'lattice':
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
# Retrieve a pointer for the DynamicInheritance-based PYG Batch class.
# For typing reasons only, use isinstance(pyg_data.Batch) for runtime checks.
DiffCSPChemGraphBatch = pyg_data.Batch(_base_cls=DiffCSPChemGraph).__class__


class SampleDataset(Dataset):

    def __init__(self, dataset, total_num):
        super().__init__()
        self.total_num = total_num
        self.distribution = train_dist[dataset]
        self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        num_atom = self.num_atoms[index]
        data = DiffCSPChemGraph(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
            frac_coords=torch.zeros((num_atom, 3)),
            lattice=torch.eye(3).view(1, 3, 3),
            atom_types=torch.zeros((num_atom, 100))
        )
        return data


class DiffCSPAdapter(DiffusionModelAdapter):
    """Adapter for wrapping a DiffCSP-style sampler/model.

    This shim expects the underlying sampler to provide:
      - `sample_prior(num_samples, step_lr=...)` -> list[Data] or similar
      - `denoise_one_step(batch, t=..., step_lr=...)` -> pred
      - Optional: `.N`, `.beta_scheduler.timesteps`

    The adapter maps DTS's `t` to the model's expected timestep semantics
    and returns list[Data] objects suitable for PyG batching.
    """

    def __init__(self, model: Any, device: Union[str, torch.device] = "cpu", batch_size:int=None, n_samples=1000, mask_elements:bool=False):
        self.model = model
        self.device = torch.device(device)
        self._N = model.beta_scheduler.timesteps
        self.sample_dataset = SampleDataset('mp_20', n_samples)
        self.batch_size = batch_size
        self.loader = DataLoader(self.sample_dataset, batch_size=batch_size, shuffle=True)
        self.step_lr = 5e-6
        self.mask_elements = mask_elements

    def to(self, device: Union[str, torch.device]) -> "DiffCSPAdapter":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def sample_prior(
        self,
        num_samples: int = 1,
        conditioning: Optional[Any] = None,
        mask: Optional[Any] = None,
        step_lr: Optional[float] = None,
    ) -> List[pyg_data.Data]:
        batch = next(iter(self.loader))
        batch = batch.to(self.device)
        prior = diffcsp_ext.sample_prior(self.model, batch)

        return prior


    def p_sample(
            self,
        x_t: Union[pyg_data.Batch, List[pyg_data.Data], ChemGraphBatch],
        t: Union[int, torch.Tensor],
        predict_x0: bool = False,
        eta: Optional[float] = None,
        step_lr: Optional[float] = None,
    ) -> Tuple[Union[List[pyg_data.Data], pyg_data.Batch], Optional[Any]]:

        x_tm1 = diffcsp_ext.denoise_one_step(self.model, x_t, t=t, step_lr=self.step_lr, mask_elements=self.mask_elements)
        return x_tm1


    def final_output(self, x):
        # Replaces atom types with integers
        atomic_numbers = x.atom_types.argmax(dim=-1) + 1
        lengths, angles = lattice_matrix_to_params_torch(x.lattice)
        return ChemGraphBatch(
            pos=x.frac_coords,
            atomic_numbers=atomic_numbers,
            lengths=lengths,
            angles=angles,
            num_atoms=x.num_atoms,
            cell=x.lattice,
            num_graphs=x.num_atoms.size(0)
        )

