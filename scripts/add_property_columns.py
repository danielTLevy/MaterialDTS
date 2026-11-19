"""Add the property columns used by MaterialDTS to MatterGen's MP-20 CSVs.

Adds `bandgap_alignn` and `dielectric_alignn` (ALIGNN predictions, using the same pretrained
models as the rewards in `src/core/materials_rewards.py`), `density` (g/cm^3, from pymatgen) and
`ml_bulk_modulus` (a copy of MP-20's `azure_bulk_modulus`) to `train.csv`, `val.csv` and
`test.csv`, in place.

Usage:
    python -m scripts.add_property_columns --csv-folder remote/mattergen/datasets/mp_20
"""
import argparse
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm

from src.core.materials_rewards import BandgapProperty, DielectricProperty

SPLITS = ("train", "val", "test")


def add_columns(csv_path: Path, properties: dict, limit: int | None = None) -> None:
    """Compute the property columns for one CSV and write it back in place."""
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)
    structures = [Structure.from_str(cif, fmt="cif") for cif in tqdm(df["cif"], desc=f"{csv_path.name}: parse")]
    df["density"] = [s.density for s in structures]
    df["ml_bulk_modulus"] = df["azure_bulk_modulus"]
    for column, prop in properties.items():
        df[column] = prop.compute(structures)
    df.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-folder", type=Path, required=True, help="Folder containing train.csv, val.csv and test.csv.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Only keep the first N rows of each CSV.")
    args = parser.parse_args()

    properties = {
        "bandgap_alignn": BandgapProperty(args.device),
        "dielectric_alignn": DielectricProperty(args.device),
    }
    for model in properties.values():
        model.model.to(args.device).eval()
    for split in SPLITS:
        add_columns(args.csv_folder / f"{split}.csv", properties, args.limit)


if __name__ == "__main__":
    main()
