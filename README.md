# PC-Algorithm

Python implementations and simulation experiments accompanying my draft paper, **Root-Induced Network-Packet Decomposition of the PC Algorithm: Structure, Correctness, Estimation, and Limits**.

The code studies causal discovery with PC-stable and a decomposition based on the graph obtained from marginal-independence tests. It includes level-synchronous and packet-first schedules, separator selection, synthetic data generators, and Gaussian and discrete conditional-independence tests.

## Installation

Tested with Python 3.10.6, NumPy 2.2.6, and SciPy 1.15.3.

```bash
git clone https://github.com/lyndaSm/PC-Algorithm.git
cd PC-Algorithm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Quick start

Run the small worked example:

```bash
python code/example.py
```

Compare the level-synchronous and packet-first schedules with PC-stable:

```bash
python code/variants.py
```

For an experiment, run one command at a time. For reproducible process settings on macOS/Linux, set these before starting Python:

```bash
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Source files

| File | Purpose |
|---|---|
| `code/simulate.py` | PC-stable, order-zero decomposition, graph utilities, separator selection, data generators, CI tests, and evaluation metrics. |
| `code/variants.py` | Level-synchronous schedule A, packet-first schedule B, and schedule-agreement experiments. |
| `code/experiments.py` | Gaussian and discrete studies, order-zero sensitivity, and small-separator existence experiments. |
| `code/v2_experiments.py` | Hub/chain generators, multiplicity control, cut objectives, and hub test-work partitions. |
| `code/review_experiments.py` | Shared marginal-p-value and multiple-testing helpers used by the experiment modules. |
| `code/robustness.py` | Bonferroni/Holm/BH comparisons, balance sensitivity, and fixed-worker test-work estimates. |
| `code/spurious.py` | Single-root negative-control experiment. |
| `code/bounded_sep.py` | Bridge-family generator and test-work experiments. |
| `code/truth_variants.py` | Skeleton and unshielded-v-structure accuracy, with paired confidence intervals. |
| `code/example.py` | Six-vertex worked example evaluated with an exact d-separation oracle. |

Modules import one another from `code/`; keep this directory intact. Experiment entry points print results to the terminal. For example, `python code/robustness.py fwer` runs the multiplicity comparison, and `python code/truth_variants.py` runs the truth-based accuracy study. `simulate.py` supplies the algorithm functions; use the experiment modules to run studies.

## Interpretation

With a common significance level and deterministic test ordering, the level-synchronous schedule is compared, test-for-test, with PC-stable. Packet-first scheduling can change finite-sample decisions. The balanced separator is a heuristic; failure to find a cut does not prove that no admissible cut exists.

Reported parallel gains are estimates based on conditional independence test results. This repository does not implement concurrent packet execution with measurements. Some inherited experiment outputs include legacy CPDAG diagnostics; the manuscript's finite-sample accuracy analysis uses the skeleton and unshielded-v-structure metrics in `truth_variants.py`.

The corresponding Version 6 study was checked across 14 experiment steps, including 13 manuscript tables and 838 numeric cells. Computation times and decisions near numerical thresholds can depend on the environment. The repository contains source code; generated output and the author's audit files are maintained separately.

## Citation

Author: Linda Smail, Zayed University, Dubai, United Arab Emirates.

Use [CITATION.cff](CITATION.cff) for software citation metadata. The accompanying article is a manuscript; no journal publication or archival DOI is asserted here.
