"""
Table 6: spurious decompositions on single-root DAGs.

Theorem 5 forbids any decomposition of a DAG with one root, so every cut
reported on the chain family is an artefact of estimation error. This measures
how often each objective reports one, and what it costs when it does: a packet
boundary drawn through a true edge deletes that edge irrecoverably
(Proposition 13), since cross-packet pairs are never revisited.

This script was written when the reproducibility package was assembled and the
table was found to have no producing script; it reproduces the published
results_spurious.txt exactly.

Run: python3 spurious.py
"""
from itertools import combinations

import numpy as np

from simulate import (balanced_cut, gaussian_sample, min_vertex_cut,
                      skeleton_of)
from v2_experiments import chain_dag, order_zero

ALPHA = 0.01
N = 1000
REPS = 25


def run():
    print("SPURIOUS DECOMPOSITIONS ON SINGLE-ROOT DAGs (chain family)")
    print(f"Theorem: no decomposition exists. {REPS} reps, "
          "Bonferroni order-zero.\n")
    hdr = (f"{'k':>3} {'n':>4} {'objective':>10} | {'reports a cut':>14} "
           f"{'true edges cut':>15} {'reps losing >=1':>16}")
    print(hdr)
    print("-" * len(hdr))
    for k in (4, 6, 8):
        m = 5
        for obj in ("minimum", "balanced"):
            found, severed, lost_any = 0, [], 0
            for rep in range(REPS):
                rng = np.random.default_rng(555 + 37 * rep + k)
                nodes, edges, _ = chain_dag(k, m, rng)
                n = len(nodes)
                X = gaussian_sample(nodes, edges, N, rng)
                G0 = order_zero(X, nodes, ALPHA / (n * (n - 1) / 2))
                if obj == "minimum":
                    R1, pk = min_vertex_cut(nodes, G0, max_size=4)
                else:
                    R1, pk = balanced_cut(nodes, G0, beta=0.3)
                if R1 is None:
                    severed.append(0)
                    continue
                found += 1
                grp = {v: 0 for v in R1}
                for i, c in enumerate(pk, 1):
                    for v in c:
                        grp[v] = i
                bad = [e for e in skeleton_of(edges)
                       if all(grp.get(v, 0) > 0 for v in e)
                       and len({grp[v] for v in e}) == 2]
                severed.append(len(bad))
                lost_any += bool(bad)
            print(f"{k:>3} {n:>4} {obj:>10} | {100*found/REPS:>13.0f}% "
                  f"{np.mean(severed):>15.2f} {100*lost_any/REPS:>15.0f}%")
    print("\nA reported cut on a single-root DAG is necessarily spurious: the")
    print("theorem forbids a real one, so any packets come from estimation "
          "error.")


if __name__ == "__main__":
    run()
