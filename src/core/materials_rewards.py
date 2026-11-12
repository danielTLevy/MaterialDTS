from mattergen.common.data.chemgraph import ChemGraphBatch
from mattergen.generator import structure_from_model_output
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
from alignn.pretrained import get_figshare_model
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
from pymatgen.core import Element, Structure
from mattersim.forcefield.potential import Potential
from mattersim.datasets.utils.build import build_dataloader
from mattergen.common.utils.globals import get_device
from pymatgen.io.ase import AseAtomsAdaptor
from mattergen.evaluation.reference.presets import ReferenceMP2020Correction
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from mattergen.evaluation.utils.vasprunlike import VasprunLike
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from functools import lru_cache

import torch
import numpy as np




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


def x_to_atoms(x: ChemGraphBatch):
    x_out_dict = dict()
    x_out_dict['lattice_mat'] = x.cell.cpu().squeeze().numpy()
    lengths, angles = lattice_matrix_to_params_torch(x.cell)
    x = x.replace(lengths=lengths, angles=angles)
    x_out_dict['elements'] = [Element.from_Z(z).symbol for z in x.atomic_numbers.cpu().numpy()]
    x_out_dict['coords'] = x.pos.cpu().numpy()
    x_out_dict['abc'] = x.lengths.cpu().squeeze().numpy()
    x_out_dict['angles'] = x.angles.cpu().squeeze().numpy()
    x_out_dict['cartesian'] = False
    x_out_dict['props'] = ['']*len(x_out_dict['elements'])
    return Atoms.from_dict(x_out_dict)

def structure_to_atoms(x: Structure):
    x_out_dict = dict()
    x_out_dict['lattice_mat'] = x.lattice.matrix
    x_out_dict['elements'] = [str(site.specie) for site in x]
    x_out_coords = [site.frac_coords for site in x]
    x_out_dict['coords'] = np.array(x_out_coords)
    x_out_dict['abc'] = x.lattice.abc
    x_out_dict['angles'] = x.lattice.angles
    x_out_dict['cartesian'] = False
    x_out_dict['props'] = ['']*len(x_out_dict['elements'])
    return Atoms.from_dict(x_out_dict)
    

class Property:
    def compute(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, x):
        return self.compute(x)

class Reward:
    def compute(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, x):
        return self.compute(x)

class Density(Property):
    def compute(self, x: list[Structure] | ChemGraphBatch):
        if isinstance(x, ChemGraphBatch):
            strucs = x_to_struc(x)
        else:
            strucs = x
        densities = np.array([struc.density for struc in strucs])
        return densities

class AlignnProperty(Property):
    def __init__(self, model_name, device):
        self.model = get_figshare_model(model_name)
        self.cutoff = 8.0
        self.max_neighbors = 12
        self.device = device

    def compute(self, x):
        outs = []
        for i in range(len(x)):
            x_i = x[i]
            if type(x_i) is Structure:
                atoms = structure_to_atoms(x_i)
            else:
                atoms = x_to_atoms(x[i])
            g, lg = Graph.atom_dgl_multigraph(
                atoms,
                cutoff=float(self.cutoff),
                max_neighbors=self.max_neighbors,
            )
            lat = torch.tensor(atoms.lattice_mat)
            out_data = (
                self.model([g.to(self.device), lg.to(self.device), lat.to(self.device)])
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
            outs.append(out_data)
        return np.array(outs).flatten()

class DielectricProperty(AlignnProperty):
    def __init__(self, device):
        model_name = 'jv_dfpt_piezo_max_dielectric_alignn'
        super().__init__(model_name, device)

class BulkModulusProperty(AlignnProperty):
    def __init__(self, device):
        model_name = 'jv_bulk_modulus_kv_alignn'
        super().__init__(model_name, device)

class BandgapProperty(AlignnProperty):
    def __init__(self, device):
        model_name = 'jv_optb88vdw_bandgap_alignn'
        super().__init__(model_name, device)



class EnergyAboveHull(Property):
    """
    Compute energy above hull per atom for a batch of structures.
    """

    def __init__(self):
        print("WARNING: this is the UNCORRECTED energy above hull. \n\
                The MP2020 compatibility is never applied, therefore the uncorrected energy \n \
                is compared against a corrected reference hull entries, which aren't comparable. \n\
                The resulting E_hull systematically overestimates energy for some \n\
                crystal systems. Use CorrectedEnergyAboveHull instead.")
        self.reference = ReferenceMP2020Correction()
        self.compatibility = MaterialsProject2020Compatibility()
        self.max_e_above_hull = 1000
        # Cache the MatterSim potential so it's loaded once, not every compute() call
        self._potential = Potential.from_checkpoint(
            device=str(get_device()), load_path=None, load_training_state=False
        )

        # Cache phase diagram construction by chemical system
        @lru_cache(maxsize=128)
        def _get_phase_diagram_cached(chemsys: str):
            from mattergen.evaluation.utils.utils import expand_into_subsystems
            from pymatgen.analysis.phase_diagram import PhaseDiagram

            subsys = expand_into_subsystems(chemsys)
            ref_entries = [
                entry
                for s in subsys
                for key in ["-".join(sorted(s))]
                for entry in self.reference.entries_by_chemsys.get(key, [])
                if not np.isnan(getattr(entry, 'energy', np.nan))
            ]

            if not ref_entries:
                # Let callers handle missing reference data by raising
                raise ValueError(f"No reference entries found for chemical system: {chemsys}")

            return PhaseDiagram(ref_entries)

        self._get_phase_diagram_cached = _get_phase_diagram_cached

    def compute(self, x: list[Structure] | ChemGraphBatch):
        # Convert model output to pymatgen Structure objects
        if isinstance(x, ChemGraphBatch):
            strucs = x_to_struc(x)
        else:
            strucs = x

        # Compute total energies using cached MatterSim potential
        atoms = [AseAtomsAdaptor.get_atoms(s) for s in strucs]
        dataloader = build_dataloader(atoms, batch_size=len(atoms), only_inference=True)
        energies, _, _ = self._potential.predict_properties(
            dataloader, include_forces=False, include_stresses=False
        )

        # Build ComputedEntry objects and process compatibility

        entries = []
        for i, (s, e) in enumerate(zip(strucs, energies)):
            ce = ComputedEntry(composition=s.composition, energy=float(e), entry_id=i)
            try:
                # Apply MaterialsProject2020Compatibility processing
                self.compatibility.process_entry(ce)
            except Exception:
                # If compatibility processing fails, continue with raw entry
                pass
            entries.append(ce)

        # Group entries by chemical system
        from mattergen.evaluation.utils.utils import generate_chemsys_dict

        entries_by_chemsys = generate_chemsys_dict(entries)

        # Prepare output array
        e_above_hull = np.full(len(entries), np.inf, dtype=float)

        # For each chemical system, build/get phase diagram and compute e_above_hull
        for chemsys, ents in entries_by_chemsys.items():
            try:
                pd = self._get_phase_diagram_cached(chemsys)
            except Exception:
                # Missing reference data or other error -> leave as nan
                continue

            for ent in ents:
                try:
                    e_ah = pd.get_e_above_hull(entry=ent, allow_negative=True)
                except Exception:
                    e_ah = np.inf
                # store value at the original entry id (we set entry_id=i earlier)
                if getattr(ent, 'entry_id', None) is not None:
                    e_above_hull[int(ent.entry_id)] = e_ah

        return e_above_hull

class CorrectedEnergyAboveHull(Property):
    """
    Compute energy above hull per atom for a batch of structures.
    """

    def __init__(self):
        self.reference = ReferenceMP2020Correction()
        self.compatibility = MaterialsProject2020Compatibility()
        self.max_e_above_hull = 1000
        # Cache the MatterSim potential so it's loaded once, not every compute() call
        self._potential = Potential.from_checkpoint(
            device=str(get_device()), load_path=None, load_training_state=False
        )

        # Cache phase diagram construction by chemical system
        @lru_cache(maxsize=128)
        def _get_phase_diagram_cached(chemsys: str):
            from mattergen.evaluation.utils.utils import expand_into_subsystems
            from pymatgen.analysis.phase_diagram import PhaseDiagram

            subsys = expand_into_subsystems(chemsys)
            ref_entries = [
                entry
                for s in subsys
                for key in ["-".join(sorted(s))]
                for entry in self.reference.entries_by_chemsys.get(key, [])
                if not np.isnan(getattr(entry, 'energy', np.nan))
            ]

            if not ref_entries:
                # Let callers handle missing reference data by raising
                raise ValueError(f"No reference entries found for chemical system: {chemsys}")

            return PhaseDiagram(ref_entries)

        self._get_phase_diagram_cached = _get_phase_diagram_cached

    def compute(self, x: list[Structure] | ChemGraphBatch):
        # Convert model output to pymatgen Structure objects
        if isinstance(x, ChemGraphBatch):
            strucs = x_to_struc(x)
        else:
            strucs = x

        # Compute total energies using cached MatterSim potential
        atoms = [AseAtomsAdaptor.get_atoms(s) for s in strucs]
        dataloader = build_dataloader(atoms, batch_size=len(atoms), only_inference=True)
        energies, _, _ = self._potential.predict_properties(
            dataloader, include_forces=False, include_stresses=False
        )

        # Build ComputedStructureEntry objects with MP2020 corrections applied.
        #
        # The correction MUST be applied here: MatterSim predicts *uncorrected* PBE energies
        # (measured 0.0155 eV/atom MAE against MP's `uncorrected_energy` for oxides/fluorides,
        # vs 0.7887 against the corrected `energy`), while the reference hull below is built
        # from MP2020-*corrected* entries. Scoring one against the other is a unit mismatch.
        #
        # Going through VasprunLike is what makes the correction land. A bare
        # `ComputedEntry(composition=..., energy=...)` carries no VASP provenance, so
        # MaterialsProject2020Compatibility cannot determine the run type; `process_entry`
        # then returns None (meaning "discard as incompatible") and leaves the entry
        # *uncorrected*. VasprunLike synthesises run_type/hubbards/potcar_symbols/oxide_type
        # from MPRelaxSet, matching MetricsEvaluator exactly (see
        # mattergen/evaluation/utils/metrics_structure_summary.py).
        entries = []
        for i, (s, e) in enumerate(zip(strucs, energies)):
            entry = VasprunLike(structure=s, energy=float(e)).get_computed_entry(
                inc_structure=True, energy_correction_scheme=self.compatibility
            )
            entry.entry_id = i  # scatter-back index for e_above_hull, set after construction
            entries.append(entry)

        # Group entries by chemical system
        from mattergen.evaluation.utils.utils import generate_chemsys_dict

        entries_by_chemsys = generate_chemsys_dict(entries)

        # Prepare output array
        e_above_hull = np.full(len(entries), np.inf, dtype=float)

        # For each chemical system, build/get phase diagram and compute e_above_hull
        for chemsys, ents in entries_by_chemsys.items():
            try:
                pd = self._get_phase_diagram_cached(chemsys)
            except Exception:
                # Missing reference data or other error -> leave as nan
                continue

            for ent in ents:
                try:
                    e_ah = pd.get_e_above_hull(entry=ent, allow_negative=True)
                except Exception:
                    e_ah = np.inf
                # store value at the original entry id (we set entry_id=i earlier)
                if getattr(ent, 'entry_id', None) is not None:
                    e_above_hull[int(ent.entry_id)] = e_ah

        return e_above_hull


class MaxReward(Reward):
    def __init__(self, property: Property, weight=1.0):
        self.property = property
        self.weight = weight

    def compute(self, x: ChemGraphBatch):
        values = self.property.compute(x)
        reward = values * self.weight
        return reward


class TargetReward(Reward):
    def __init__(self, property: Property, target_value, weight=1.0):
        self.property = property
        self.target_value = target_value
        self.weight = weight

    def compute(self, x: ChemGraphBatch):
        values = self.property.compute(x)
        reward = -np.abs(values - self.target_value) * self.weight
        return reward

class MinReward(Reward):
    def __init__(self, property: Property, weight=1.0):
        self.property = property
        self.weight = weight

    def compute(self, x: ChemGraphBatch):
        values = self.property.compute(x)
        reward = -values * self.weight
        return reward

class TargetDensity15Reward(TargetReward):
    def __init__(self, weight=1.0):
        super().__init__(Density(), target_value=15.0, weight=weight)

class TargetDensity10Reward(TargetReward):
    def __init__(self, weight=1.0):
        super().__init__(Density(), target_value=10.0, weight=weight)


class MinDensityReward(MinReward):
    def __init__(self, weight=1.0):
        super().__init__(Density(), weight=weight)

class MaxDielectricReward(MaxReward):
    def __init__(self, device, weight=1.0):
        super().__init__(DielectricProperty(device), weight=weight)

class MaxBulkModulusReward(MaxReward):
    def __init__(self, device, weight=1.0):
        super().__init__(BulkModulusProperty(device), weight=weight)

class TargetBulkModulusReward(TargetReward):
    def __init__(self, device, target_value, weight=1.0):
        super().__init__(
            BulkModulusProperty(device),
            target_value=target_value,
            weight=weight,
        )

class MaxBandgapReward(MaxReward):
    def __init__(self, device, weight=1.0):
        super().__init__(BandgapProperty(device), weight=weight)

# Floor applied before taking the log of a property value. Roughly half of the
# ALIGNN bandgap predictions (and a few hundred dielectric ones) are <= 0
# (metals), for which log is undefined; clamping to this small positive floor
# maps them to a finite, low reward instead of -inf.
LOG_REWARD_FLOOR = 0.1

class LogProperty(Property):
    """Log-transform of another property, clamped to a positive floor.

    Wraps an inner ``Property`` and returns ``log(clip(inner(x), floor, None))``.
    The clamp keeps non-positive predictions (e.g. metallic bandgaps) finite.
    """
    def __init__(self, inner: Property, floor: float = LOG_REWARD_FLOOR):
        self.inner = inner
        self.floor = floor

    def compute(self, x):
        values = np.asarray(self.inner.compute(x), dtype=np.float64)
        return np.log(np.clip(values, self.floor, None))

class MaxLogDielectricReward(MaxReward):
    def __init__(self, device, weight=1.0, floor: float = LOG_REWARD_FLOOR):
        super().__init__(LogProperty(DielectricProperty(device), floor=floor), weight=weight)

class MaxLogBandgapReward(MaxReward):
    def __init__(self, device, weight=1.0, floor: float = LOG_REWARD_FLOOR):
        super().__init__(LogProperty(BandgapProperty(device), floor=floor), weight=weight)


class MinEnergyAboveHullReward(MinReward):
    def __init__(self, weight=1.0):
        super().__init__(EnergyAboveHull(), weight=weight)

class MinCorrectedEnergyAboveHullReward(MinReward):
    def __init__(self, weight=1.0):
        super().__init__(CorrectedEnergyAboveHull(), weight=weight)

class NullReward(Reward):
    def compute(self, x):
        return np.zeros(len(x), dtype=float)


class SpaceGroupReward(Reward):
    """Binary reward: 1.0 if structure belongs to the target space group, else 0.0."""

    def __init__(self, target_space_group: int, weight: float = 1.0):
        self.target_space_group = target_space_group
        self.weight = weight

    def compute(self, x) -> np.ndarray:
        strucs = x if isinstance(x, list) else x_to_struc(x)
        rewards = []
        for struc in strucs:
            try:
                analyzer = SpacegroupAnalyzer(struc, symprec=0.1)
                sg_num = analyzer.get_space_group_number()
                rewards.append(1.0 if sg_num == self.target_space_group else 0.0)
            except Exception as e:
                print(f"Warning: space group analysis failed: {e}")
                rewards.append(0.0)
        return np.array(rewards) * self.weight


class Fm3BarmReward(SpaceGroupReward):
    """Reward for Fm3-m (space group 225)."""

    def __init__(self, weight: float = 1.0):
        super().__init__(target_space_group=225, weight=weight)

class Im3BarReward(SpaceGroupReward):
    """Reward for Im3- (space group 204)."""

    def __init__(self, weight: float = 1.0):
        super().__init__(target_space_group=204, weight=weight)

    







