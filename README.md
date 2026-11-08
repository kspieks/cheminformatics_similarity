# cheminformatics_similarity

Utility functions to compute chemical similarity between sets of molecules:

- **Fingerprints**: RDKit path fingerprints (minPath=1, maxPath=7, 2048 bits) as bit or count vectors, generated in parallel with joblib.
- **Similarity matrices**: Tanimoto or Dice similarity between two lists of SMILES, returned as a numpy array of shape `(len(list1), len(list2))`.
- **Nearest neighbors**: the n most or least similar molecules in one list for each molecule in another.
- **Activity cliffs**: Structure-Activity Landscape Index (SALI) for every pair in a dataset, with filtering and plotting helpers.

## Main commands

```bash
# once, after cloning
brew install uv
uv sync

# compute a similarity matrix between two csv files
uv run python calculate_chemical_similarity.py --data_path1 a.csv --data_path2 b.csv --out_file sim.npz

# open the example notebooks
uv run jupyter lab
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`). Python 3.12; minimum
versions are declared in `pyproject.toml` and the exact versions used are locked in `uv.lock`.

```bash
uv sync    # creates .venv and installs the package in editable mode
```

`uv sync` also installs the packages needed for the notebooks (JupyterLab, ipykernel, scipy).
To skip them, use `uv sync --no-default-groups`.

To run `python` without the `uv run` prefix, activate the environment once per terminal:

```bash
source .venv/bin/activate
```

## Compute a similarity matrix

`calculate_chemical_similarity.py` reads two csv files that each have a `smiles` column,
computes fingerprints, and saves the similarity matrix to a compressed `.npz` file.

```bash
uv run python calculate_chemical_similarity.py \
    --data_path1 examples/data/ESOL/delaney.csv \
    --data_path2 examples/data/ESOL/delaney.csv \
    --fp_type count \
    --metric tanimoto \
    --out_file similarity_matrix.npz
```

| Option | Values | Default |
|---|---|---|
| `--fp_type` | `count`, `bit` | `count` |
| `--metric` | `tanimoto`, `dice` | `tanimoto` |
| `--n_cpus` | CPUs for the similarity matrix | 2 |
| `--n_cpus_featurize` | CPUs for fingerprint generation | 2 |

Load the result with:

```python
import numpy as np
similarity_matrix = np.load("similarity_matrix.npz")["similarity"]
```

`examples/example.sh` runs the same command on the ESOL dataset:

```bash
cd examples
uv run bash example.sh
```

## Notebooks

```bash
uv run jupyter lab
```

In VS Code, select `.venv/bin/python` as the notebook kernel instead.

| Notebook | Contents |
|---|---|
| `examples/calculate_similarity_matrix.ipynb` | Similarity matrix on ESOL; draws pairs with similarity > 0.97 |
| `examples/compare_similarity_metrics.ipynb` | RDKit vs SciPy Tanimoto and Dice, bit vs count vectors |
| `examples/MolsToGridImage.ipynb` | Drawing molecule grids with substructure highlights |
| `examples/data/VEGFR2/VEGFR2_SALI_maxPath7_bit.ipynb` | SALI activity cliffs on VEGFR2 IC50 data |
| `examples/data/covid_moonshot/covid_SALI_reproduce_pat_walters.ipynb` | SALI on COVID Moonshot data |
| `examples/visualize_chemical_space.ipynb` | UMAP of chemical space; also requires [BMLC](https://github.com/kspieks/BMLC), `descriptastorus`, `scikit-learn`, and `umap-learn` |

## Updating package versions

```bash
uv lock --upgrade   # newest versions allowed by pyproject.toml
uv sync
```

Rerun the notebooks to check the outputs, then commit the updated `uv.lock`.

## Project structure

```
├── cheminformatics_similarity/      # Package
│   ├── fingerprints.py              # bit and count RDKit fingerprints
│   ├── similarity.py                # similarity matrix, n most/least similar
│   ├── sali.py                      # SALI activity cliffs
│   └── parsing.py                   # CLI arguments
├── calculate_chemical_similarity.py # CLI: similarity matrix from two csv files
├── examples/                        # Notebooks, example.sh, and datasets
├── pyproject.toml
└── uv.lock
```
