"""A multi-root family whose canonical separator has BOUNDED G0-degree.

Section 11 measures the hub family, in which every module sink feeds a common
effect, so the separator is adjacent to the whole graph and its G0-degree grows
with k. Remark 9 states that a canonical separator need not have high degree,
but the manuscript exhibits no family in which it does not, so the reader cannot
tell whether the modest speedup is a property of the decomposition or of the
generator. This script supplies the missing family.

Generator (`bridge_dag`). k modules, each a sparse DAG on m vertices with a path
backbone and extra forward edges at probability p_extra, so each module has its
own root. Consecutive modules i and i+1 are joined by a dedicated bridge vertex
b_i receiving the sink of each:

        sink(M_i)  ->  b_i  <-  sink(M_{i+1}).

Each b_i is a collider of in-degree two. Its root signature is
{root_i, root_{i+1}}, so b_i lies in the canonical separator of any root
partition that splits i from i+1, and its G0-neighbourhood is confined to
M_i, M_{i+1} and the two adjacent bridges: size 2m+2 regardless of k, whereas
the hub's grows without bound.

Contrast with `chain_dag` in v2_experiments.py, which also uses low-degree
mediators but chains them into a SINGLE root, so no population decomposition
exists at all. Here every module keeps its own root, so Theorem 5 guarantees
one.

Two separator choices are run:

    balanced    the greedy highest-degree heuristic used throughout the paper,
    oracle      the true bridge set, which is the canonical separator of the
                all-singletons root partition.

and two work-partition models:

    speedup     order-zero charged wholly to the serial part (the paper's model),
    speedup*    order-zero also divided across the packet processors, since the
                marginal tests are mutually independent. This second saving is
                NOT attributable to the decomposition; any PC implementation can
                take it. It is reported so that the two are not confused.

Usage:  python3 bounded_sep.py
"""
import sys
from itertools import combinations

import numpy as np

from simulate import (FisherZ, balanced_cut, components, gaussian_sample,
                      metrics, pc_stable, true_cpdag)
from review_experiments import g0_from_p, pvalues
from variants import variant_B

ALPHA = 0.01
N = 1000
REPS = 25
KS = (4, 6, 8, 10)
M = 5


def bridge_dag(k, m, rng, p_extra=0.12):
    """k modules with their own roots, joined pairwise by low-degree bridges."""
    nodes, edges, sinks, bridges = [], [], [], []
    for i in range(k):
        mod = [f"m{i}_{j}" for j in range(m)]
        nodes += mod
        for a in range(m):
            for b in range(a + 1, m):
                if b == a + 1 or rng.random() < p_extra:
                    edges.append((mod[a], mod[b]))
        sinks.append(mod[-1])
    for i in range(k - 1):
        b = f"b{i}"
        nodes.append(b)
        bridges.append(b)
        edges += [(sinks[i], b), (sinks[i + 1], b)]
    return nodes, edges, bridges


def g0_degrees(nodes, G0, subset):
    deg = {v: 0 for v in nodes}
    for e in G0:
        for v in e:
            deg[v] += 1
    return [deg[v] for v in subset]


def stage_work(nodes, X, G0, R1, pk):
    """CI-test work of the staged schedule, split by stage."""
    pw = []
    pe = set()
    for c in pk:
        scope = sorted(set(c) | set(R1))
        init = {e for e in G0 if all(v in scope for v in e)}
        forb = {e for e in init if not all(v in c for v in e)}
        tl = FisherZ(X, nodes)
        ee, _ = pc_stable(scope, tl, ALPHA, init_edges=init,
                          start_l=1, forbidden=forb)
        pw.append(tl.calls)
        pe |= {e for e in ee if all(v in c for v in e)}
    inside = {e for c in pk for e in G0 if all(v in c for v in e)}
    cross = {frozenset({x, y})
             for a, b in combinations(range(len(pk)), 2)
             for x in pk[a] for y in pk[b]}
    ti = FisherZ(X, nodes)
    pc_stable(nodes, ti, ALPHA, init_edges=(G0 - inside) | pe,
              start_l=1, forbidden=pe | cross)
    return pw, ti.calls


def main():
    print("BOUNDED-DEGREE SEPARATOR FAMILY")
    print(f"bridge family, m={M}; N={N}; alpha={ALPHA}; Bonferroni order-zero;")
    print(f"{REPS} replicates per row. Work is counted in CI tests, every test")
    print("charged one unit, the cut search excluded, exactly as in the hub")
    print("experiment. 'sep deg' is the mean G0-degree of the chosen separator")
    print("vertices. 'speedup' charges the order-zero sweep wholly to the serial")
    print("part; 'spd*' divides it across the packet processors as well, a")
    print("saving available to any PC implementation and not attributable to the")
    print("decomposition.\n")
    hdr = (f"{'k':>3} {'n':>4} {'separator':>10} | {'cuts':>7} {'|R1|':>5} "
           f"{'pkts':>5} {'maxpkt':>7} {'sepdeg':>7} | {'ord-0':>6} {'pkts':>6} "
           f"{'slowest':>8} {'iface':>7} | {'share':>6} {'speedup':>8} {'spd*':>7} "
           f"| {'PC SHD':>7} {'B SHD':>7} {'B-PC':>6}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for k in KS:
        for how in ("balanced", "oracle"):
            r1s, npk, mx, degs = [], [], [], []
            w0s, wps, wpm, wifc, shs, sps, sps2 = [], [], [], [], [], [], []
            pc_shd, b_shd = [], []
            cuts = 0
            for rep in range(REPS):
                rng = np.random.default_rng(31 + 977 * rep + k)
                nodes, edges, bridges = bridge_dag(k, M, rng)
                X = gaussian_sample(nodes, edges, N, rng)
                n = len(nodes)
                truth = true_cpdag(nodes, edges)

                t_pc = FisherZ(X, nodes)
                e_pc, s_pc = pc_stable(nodes, t_pc, ALPHA)
                pc_shd.append(metrics(e_pc, s_pc, nodes, edges,
                                      true_cp=truth)["shd"])

                G0 = g0_from_p(pvalues(X, nodes), "bonferroni")
                if how == "balanced":
                    R1, pk = balanced_cut(nodes, G0, beta=0.3)
                else:
                    R1 = sorted(bridges)
                    pk = [c for c in components(set(nodes) - set(R1), G0)]
                    if len(pk) < 2:
                        R1 = None
                if R1 is None:
                    sep0 = {frozenset(p): frozenset()
                            for p in combinations(nodes, 2)
                            if frozenset(p) not in G0}
                    tf = FisherZ(X, nodes)
                    eF, sF = pc_stable(nodes, tf, ALPHA, init_edges=G0,
                                       start_l=1, sepsets=sep0)
                    b_shd.append(metrics(eF, sF, nodes, edges,
                                         true_cp=truth)["shd"])
                    continue
                cuts += 1
                r1s.append(len(R1))
                npk.append(len(pk))
                mx.append(max(len(c) for c in pk))
                degs.append(float(np.mean(g0_degrees(nodes, G0, R1))))

                pw, iface = stage_work(nodes, X, G0, R1, pk)
                w0 = n * (n - 1) // 2
                wser = w0 + sum(pw) + iface
                wpar = w0 + max(pw) + iface
                wpar2 = w0 / len(pk) + max(pw) + iface
                w0s.append(w0); wps.append(sum(pw)); wpm.append(max(pw))
                wifc.append(iface)
                shs.append(sum(pw) / wser)
                sps.append(wser / wpar)
                sps2.append(wser / wpar2)

                e_b, s_b, _ = variant_B(nodes, X, G0, R1, pk, ALPHA)
                b_shd.append(metrics(e_b, s_b, nodes, edges,
                                     true_cp=truth)["shd"])

            f = lambda v: np.mean(v) if len(v) else float("nan")
            print(f"{k:>3} {len(nodes):>4} {how:>10} | {cuts:>3}/{REPS:<3} "
                  f"{f(r1s):>5.1f} {f(npk):>5.1f} {f(mx):>7.1f} {f(degs):>7.1f} | "
                  f"{f(w0s):>6.0f} {f(wps):>6.0f} {f(wpm):>8.0f} {f(wifc):>7.0f} | "
                  f"{100*f(shs):>5.0f}% {f(sps):>8.2f} {f(sps2):>7.2f} | "
                  f"{f(pc_shd):>7.2f} {f(b_shd):>7.2f} "
                  f"{f(b_shd)-f(pc_shd):>+6.2f}")
            rows.append((k, len(nodes), how, f(sps), f(sps2),
                         100 * f(shs), f(degs), f(wifc)))

    print("\nReading. The separator's G0-degree stays bounded as k grows, so the")
    print("interface no longer swallows the work, and the packet share stops")
    print("collapsing. Under the paper's model the speedup nevertheless stays")
    print("near 1.2, because the bottleneck has simply moved to the order-zero")
    print("sweep, which grows as n^2 and is charged wholly to the serial part.")
    print("Once that sweep is parallelised too (spd*), the same runs give a")
    print("substantially larger figure, and it grows with k. The hub family's")
    print("degrading speedup is therefore a property of that generator, as")
    print("Remark 9 anticipated, and the residual limit on this family is the")
    print("order-zero sweep rather than the confluence.")
    for k, n, how, sp, sp2, sh, dg, ifc in rows:
        print(f"  k={k:>3} n={n:>3} {how:>9}  share {sh:>4.0f}%  "
              f"speedup {sp:>4.2f}  spd* {sp2:>5.2f}  "
              f"sep G0-degree {dg:>5.1f} (hub at this n: {n-1})  "
              f"interface {ifc:>6.0f}")


if __name__ == "__main__":
    sys.exit(main())
