"""
Two experiments prompted by review, plus one the reviews missed.

  bh     Benjamini-Hochberg vs Bonferroni vs uncorrected at order zero.
         Does BH keep multiplicity control without the power loss that makes
         Bonferroni over-correct at small n?
  agree  Decompose the decomposed-vs-serial disagreement into adjacency
         differences and orientation-only differences.
  heur   The balanced-cut heuristic's known failure mode: it is tuned to
         separators that are hubs. Measured on a family whose separator
         vertices are low-degree mediators.

Run: python3 review_experiments.py {bh|agree|heur}
"""
import sys
import time
from itertools import combinations

import numpy as np
from scipy import stats

from simulate import (FisherZ, balanced_cut, components, gaussian_sample,
                      inferred_vstructures, min_vertex_cut, pc_stable, skeleton_of)
from v2_experiments import chain_dag, hub_dag, module_of

ALPHA = 0.01
REPS = 25


# --------------------------------------------------------------- p-values
def pvalues(X, nodes):
    """All marginal Fisher-z p-values, as a dict on unordered pairs."""
    C = np.corrcoef(X, rowvar=False)
    n = X.shape[0]
    idx = {v: i for i, v in enumerate(nodes)}
    out = {}
    for x, y in combinations(nodes, 2):
        r = float(np.clip(C[idx[x], idx[y]], -0.999999, 0.999999))
        z = 0.5 * np.sqrt(n - 3) * np.log((1 + r) / (1 - r))
        out[frozenset({x, y})] = 2 * (1 - stats.norm.cdf(abs(z)))
    return out


def g0_from_p(p, method, alpha=ALPHA):
    """Dependent pairs = rejected hypotheses of marginal independence."""
    pairs = sorted(p, key=lambda e: p[e])
    mtests = len(pairs)
    if method == "uncorrected":
        thr = {e: alpha for e in pairs}
        return {e for e in pairs if p[e] <= thr[e]}
    if method == "bonferroni":
        return {e for e in pairs if p[e] <= alpha / mtests}
    if method == "holm":
        # Holm step-down FWER control.  Reject H_(1),...,H_(k) until the
        # first p_(i) > alpha/(m-i+1); all later hypotheses are retained.
        rejected = set()
        for i, e in enumerate(pairs, 1):
            if p[e] <= alpha / (mtests - i + 1):
                rejected.add(e)
            else:
                break
        return rejected
    if method == "bh":
        # Benjamini-Hochberg: largest i with p_(i) <= i/m * alpha
        kmax = 0
        for i, e in enumerate(pairs, 1):
            if p[e] <= i / mtests * alpha:
                kmax = i
        return set(pairs[:kmax])
    raise ValueError(method)


# ------------------------------------------------------------------- BH
def exp_bh():
    print("BH vs BONFERRONI vs UNCORRECTED AT ORDER ZERO")
    print(f"alpha={ALPHA}, {REPS} reps\n")
    print("Part A: multiplicity regime (hub family, packets from the TRUE "
          "separator)")
    hdr = (f"{'k':>3} {'n':>4} {'method':>12} | {'|G0|':>6} {'false xmod':>11} "
           f"{'packets':>8} {'max pkt':>8} {'exact':>6}")
    print(hdr)
    print("-" * len(hdr))
    for k in (6, 8, 10, 12):
        m = 5
        for meth in ("uncorrected", "bonferroni", "bh"):
            sz, fe, npk, mx, ex = [], [], [], [], 0
            for rep in range(REPS):
                rng = np.random.default_rng(31 + 977 * rep + k)
                nodes, edges, truesep = hub_dag(k, m, rng)
                X = gaussian_sample(nodes, edges, 1000, rng)
                G0 = g0_from_p(pvalues(X, nodes), meth)
                sz.append(len(G0))
                fe.append(sum(1 for e in G0
                              if len({module_of(v) for v in e}) == 2
                              and all(module_of(v) != "SEP" for v in e)))
                comps = components(set(nodes) - set(truesep), G0)
                npk.append(len(comps))
                mx.append(max((len(c) for c in comps), default=0))
                if len(comps) == k and max((len(c) for c in comps),
                                           default=0) == m:
                    ex += 1
            print(f"{k:>3} {len(nodes):>4} {meth:>12} | {np.mean(sz):>6.0f} "
                  f"{np.mean(fe):>11.1f} {np.mean(npk):>8.1f} "
                  f"{np.mean(mx):>8.1f} {100*ex/REPS:>5.0f}%")

    print("\nPart B: power regime (small n, where Bonferroni over-corrects)")
    hdr = (f"{'n':>3} {'N':>6} {'method':>12} | {'|G0|':>6} {'found':>6} "
           f"{'|R1|':>5} {'split err':>10} {'lost edges':>11}")
    print(hdr)
    print("-" * len(hdr))
    for N in (200, 1000):
        for meth in ("uncorrected", "bonferroni", "bh"):
            sz, found, r1s, splitbad, lost = [], 0, [], 0, []
            for rep in range(REPS):
                rng = np.random.default_rng(90000 + 211 * rep + N)
                nodes, edges, _ = hub_dag(3, 4, rng)
                truth = skeleton_of(edges)
                X = gaussian_sample(nodes, edges, N, rng)
                G0 = g0_from_p(pvalues(X, nodes), meth)
                sz.append(len(G0))
                R1, pk = min_vertex_cut(nodes, G0, max_size=3)
                if R1 is None:
                    lost.append(0)
                    continue
                found += 1
                r1s.append(len(R1))
                grp = {v: 0 for v in R1}
                for i, c in enumerate(pk, 1):
                    for v in c:
                        grp[v] = i
                bad = [e for e in truth
                       if all(grp.get(v, 0) > 0 for v in e)
                       and len({grp[v] for v in e}) == 2]
                lost.append(len(bad))
                splitbad += bool(bad)
            print(f"{len(nodes):>3} {N:>6} {meth:>12} | {np.mean(sz):>6.0f} "
                  f"{100*found/REPS:>5.0f}% "
                  f"{(np.mean(r1s) if r1s else float('nan')):>5.1f} "
                  f"{100*splitbad/REPS:>9.0f}% {np.mean(lost):>11.2f}")


# ---------------------------------------------------------------- agree
def exp_agree():
    print("WHERE DOES THE DECOMPOSED-VS-SERIAL DISAGREEMENT COME FROM?")
    print(f"hub family, Bonferroni order-zero, balanced cut, {REPS} reps\n")
    hdr = (f"{'k':>3} {'n':>4} | {'skeleton same':>14} {'colliders same':>14} "
           f"| {'orientation-only':>17} {'adjacency':>10}")
    print(hdr)
    print("-" * len(hdr))
    for k in (4, 6, 8, 10, 12):
        m = 5
        skel_same = cp_same = used = 0
        for rep in range(REPS):
            rng = np.random.default_rng(31 + 977 * rep + k)
            nodes, edges, _ = hub_dag(k, m, rng)
            X = gaussian_sample(nodes, edges, 1000, rng)
            n = len(nodes)
            tb = FisherZ(X, nodes)
            e_pc, s_pc = pc_stable(nodes, tb, ALPHA)
            G0 = g0_from_p(pvalues(X, nodes), "bonferroni")
            R1, pk = balanced_cut(nodes, G0, beta=0.3)
            if R1 is None:
                continue
            used += 1
            sep = {e: frozenset() for e in
                   ({frozenset(p) for p in combinations(nodes, 2)} - G0)}
            pe = set()
            for c in pk:
                scope = sorted(set(c) | set(R1))
                init = {e for e in G0 if all(v in scope for v in e)}
                forb = {e for e in init if not all(v in c for v in e)}
                tl = FisherZ(X, nodes)
                ee, ss = pc_stable(scope, tl, ALPHA, init_edges=init,
                                   start_l=1, forbidden=forb)
                sep.update(ss)
                pe |= {e for e in ee if all(v in c for v in e)}
            inside = {e for c in pk for e in G0 if all(v in c for v in e)}
            cross = {frozenset({x, y})
                     for a, b in combinations(range(len(pk)), 2)
                     for x in pk[a] for y in pk[b]}
            ti = FisherZ(X, nodes)
            e_en, s_en = pc_stable(nodes, ti, ALPHA,
                                   init_edges=(G0 - inside) | pe,
                                   start_l=1, forbidden=pe | cross)
            sep.update(s_en)
            same_skel = (e_en == e_pc)
            skel_same += same_skel
            c1 = inferred_vstructures(e_pc, s_pc, nodes)
            c2 = inferred_vstructures(e_en, sep, nodes)
            cp_same += (c1 == c2)
        so, co = 100 * skel_same / used, 100 * cp_same / used
        print(f"{k:>3} {len(nodes):>4} | {so:>13.0f}% {co:>10.0f}% "
              f"| {so-co:>16.0f}% {100-so:>9.0f}%"
              f"   [cut found {used}/{REPS}; numerators "
              f"skeleton {skel_same}/{used}, colliders {cp_same}/{used}]")
    print("\n'orientation-only' = same skeleton but different inferred-collider set;")
    print("'adjacency'        = skeletons differ.")


# ----------------------------------------------------------------- heur
def exp_heur():
    print("FAILURE MODE OF THE BALANCED-CUT HEURISTIC")
    print("The heuristic deletes highest-degree vertices, so it finds hub")
    print("separators. On a family whose separators are low-degree mediators")
    print("it removes the wrong vertices. Chain family, Bonferroni order-zero,")
    print(f"{REPS} reps.\n")
    hdr = (f"{'family':>22} {'k':>3} {'n':>4} {'objective':>14} | "
           f"{'found':>6} {'|R1|':>6} {'packets':>8} {'max pkt':>8} "
           f"{'ideal max':>10}")
    print(hdr)
    print("-" * len(hdr))
    for gen, name, ideal in ((hub_dag, "hub (separator = hub)", 5),
                             (chain_dag, "chain (mediators)", 5)):
        for k in (4, 6):
            m = 5
            for obj in ("minimum", "balanced"):
                r1, npk, mx, found = [], [], [], 0
                for rep in range(REPS):
                    rng = np.random.default_rng(555 + 37 * rep + k)
                    nodes, edges, _ = gen(k, m, rng)
                    X = gaussian_sample(nodes, edges, 1000, rng)
                    G0 = g0_from_p(pvalues(X, nodes), "bonferroni")
                    if obj == "minimum":
                        R1, pk = min_vertex_cut(nodes, G0, max_size=4)
                    else:
                        R1, pk = balanced_cut(nodes, G0, beta=0.3)
                    if R1 is None:
                        continue
                    found += 1
                    r1.append(len(R1))
                    npk.append(len(pk))
                    mx.append(max(len(c) for c in pk))
                if not found:
                    print(f"{name:>22} {k:>3} {len(nodes):>4} {obj:>14} | "
                          f"{0:>5}%  (no decomposition exists)")
                    continue
                print(f"{name:>22} {k:>3} {len(nodes):>4} {obj:>14} | "
                      f"{100*found/REPS:>5.0f}% {np.mean(r1):>6.1f} "
                      f"{np.mean(npk):>8.1f} {np.mean(mx):>8.1f} "
                      f"{ideal:>10}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "bh"
    t0 = time.time()
    {"bh": exp_bh, "agree": exp_agree, "heur": exp_heur}[which]()
    print(f"\nelapsed {time.time()-t0:.0f}s")
