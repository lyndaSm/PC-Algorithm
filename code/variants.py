"""
The two variants of the packet algorithm, stated explicitly and compared.

  A. LEVEL-SYNCHRONOUS packet PC. One order-independent sweep over G0 with
     cross-packet pairs forbidden. Packet and interface pairs are processed at
     the same conditioning order against the same frozen adjacency graph.

  B. STAGED (packet-first) PC. Each packet is searched to completion over
     G0[R0^i u R1], then a single interface pass. This is the variant that
     parallelises, and the one Section 11 times.

They are NOT the same algorithm: B can test conditioning sets that A never
reaches, because stale interface edges are still present while a packet is being
completed. This script measures the difference.
"""
from itertools import combinations

import numpy as np

from simulate import (FisherZ, balanced_cut, cpdag, gaussian_sample, pc_stable,
                      true_cpdag)
from review_experiments import g0_from_p, pvalues
from v2_experiments import hub_dag

ALPHA = 0.01


def variant_A(nodes, X, G0, R1, packets, alpha=ALPHA):
    """Level-synchronous: one sweep, cross-packet pairs forbidden."""
    cross = {frozenset({x, y})
             for a, b in combinations(range(len(packets)), 2)
             for x in packets[a] for y in packets[b]}
    sep0 = {frozenset(p): frozenset() for p in combinations(nodes, 2)
            if frozenset(p) not in G0}
    t = FisherZ(X, nodes)
    e, s = pc_stable(nodes, t, alpha, init_edges=G0, start_l=1,
                     forbidden=cross, sepsets=sep0)
    return e, s, t.calls


def variant_B(nodes, X, G0, R1, packets, alpha=ALPHA):
    """Staged: packets to completion, then the interface."""
    calls = 0
    sep = {frozenset(p): frozenset() for p in combinations(nodes, 2)
           if frozenset(p) not in G0}
    pe = set()
    for c in packets:
        scope = sorted(set(c) | set(R1))
        init = {e for e in G0 if all(v in scope for v in e)}
        forb = {e for e in init if not all(v in c for v in e)}
        t = FisherZ(X, nodes)
        ee, ss = pc_stable(scope, t, alpha, init_edges=init, start_l=1,
                           forbidden=forb)
        calls += t.calls
        sep.update(ss)
        pe |= {e for e in ee if all(v in c for v in e)}
    inside = {e for c in packets for e in G0 if all(v in c for v in e)}
    cross = {frozenset({x, y})
             for a, b in combinations(range(len(packets)), 2)
             for x in packets[a] for y in packets[b]}
    t = FisherZ(X, nodes)
    e, s = pc_stable(nodes, t, alpha, init_edges=(G0 - inside) | pe,
                     start_l=1, forbidden=pe | cross)
    calls += t.calls
    sep.update(s)
    return e, sep, calls


def wilson(k, n, z=1.96):
    """Wilson score interval, valid at k=0 and k=n unlike the normal interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


if __name__ == "__main__":
    import sys
    REPS = 25
    ks = [int(a) for a in sys.argv[1:] if a.isdigit()] or [4, 6, 8]
    quiet = len(ks) < 3
    if not quiet:
        print("VARIANT A (level-synchronous) vs VARIANT B (staged) vs PC-stable")
        print(f"hub family, balanced cut, N=1000, {REPS} reps, alpha={ALPHA}")
        print("Every proportion is over all "
              f"{REPS} replicates: when no cut is found, Algorithm 1's "
              "ordinary-PC fallback is run and scored.\n")
    hdr = (f"{'k':>3} {'n':>4} {'order-0 level':>16} | {'A tests':>8} "
           f"{'A skel':>7} {'A CPDAG':>8} | {'B tests':>8} {'B skel':>7} "
           f"{'B CPDAG':>8}")
    if not quiet:
        print(hdr)
        print("-" * len(hdr))
    for k in ks:
        m = 5
        for label, meth in (("alpha_0 = alpha", "uncorrected"),
                            ("Bonferroni", "bonferroni")):
            aT = aS = aC = bT = bS = bC = used = 0
            fbT = fbS = fbC = 0
            for rep in range(REPS):
                rng = np.random.default_rng(31 + 977 * rep + k)
                nodes, edges, _ = hub_dag(k, m, rng)
                X = gaussian_sample(nodes, edges, 1000, rng)
                n = len(nodes)
                tb = FisherZ(X, nodes)
                e_pc, s_pc = pc_stable(nodes, tb, ALPHA)
                pcT = tb.calls
                ref = cpdag(e_pc, s_pc, nodes)
                npairs = n * (n - 1) // 2
                G0 = g0_from_p(pvalues(X, nodes), meth)
                R1, pk = balanced_cut(nodes, G0, beta=0.3)
                if R1 is None:
                    # No cut: Algorithm 1 falls back to ordinary PC, resuming
                    # from the order-zero graph already computed. We RUN that
                    # fallback rather than assume its outcome, so the row is
                    # over all REPS replicates and matches the protocol.
                    sep0 = {frozenset(p): frozenset()
                            for p in combinations(nodes, 2)
                            if frozenset(p) not in G0}
                    tf = FisherZ(X, nodes)
                    eF, sF = pc_stable(nodes, tf, ALPHA, init_edges=G0,
                                       start_l=1, sepsets=sep0)
                    hitT = (tf.calls + npairs == pcT)
                    hitS = (eF == e_pc)
                    hitC = (cpdag(eF, sF, nodes) == ref)
                    aT += hitT; aS += hitS; aC += hitC
                    bT += hitT; bS += hitS; bC += hitC
                    fbT += hitT; fbS += hitS; fbC += hitC
                    continue
                used += 1
                eA, sA, cA = variant_A(nodes, X, G0, R1, pk)
                eB, sB, cB = variant_B(nodes, X, G0, R1, pk)
                aT += (cA + npairs == pcT); aS += (eA == e_pc)
                aC += (cpdag(eA, sA, nodes) == ref)
                bT += (cB + npairs == pcT); bS += (eB == e_pc)
                bC += (cpdag(eB, sB, nodes) == ref)
            f = lambda x: f"{100*x/REPS:>7.0f}%"
            print(f"{k:>3} {n:>4} {label:>16} | {f(aT)} {f(aS)} {f(aC)} "
                  f"| {f(bT)} {f(bS)} {f(bC)}   [cut found {used}/{REPS}]")
            # Exact numerators, so the paper's caption never has to be inferred
            # from a rounded percentage. Both conventions are printed: over all
            # replicates (the protocol, with fallback) and over successful cuts.
            print(f"      counts /{REPS}: A {aT},{aS},{aC}  B {bT},{bS},{bC}"
                  f"   | among the {used} cut replicates:"
                  f" A {aT-fbT},{aS-fbS},{aC-fbC}"
                  f"  B {bT-fbT},{bS-fbS},{bC-fbC}"
                  f"   | fallback {REPS-used} replicates hit"
                  f" {fbT},{fbS},{fbC}")
    if quiet:
        raise SystemExit
    print()
    print("Wilson 95% intervals for a 25-replicate proportion:")
    for kk in (0, 21, 24, 25):
        lo, hi = wilson(kk, 25)
        print(f"   {kk:>2}/25 = {100*kk/25:>3.0f}%  ->  [{100*lo:.0f}%, {100*hi:.0f}%]")
