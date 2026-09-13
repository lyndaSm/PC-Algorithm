"""Truth-based evaluation of PC-stable, schedule (A) and schedule (B).

The finite-sample skeleton and the set of inferred unshielded colliders are
well-defined even when erroneous CI decisions make a fully oriented PC output
inconsistent.  Therefore this script evaluates skeleton F1/SHD and v-structure
F1/SHD against the generating DAG, rather than assuming every finite-sample
orientation is a valid CPDAG.

    PC      ordinary PC-stable skeleton search at alpha,
    A       Algorithm 1, schedule (A), FWER-corrected order zero, balanced cut,
    B       Algorithm 1, schedule (B), same order-zero graph and cut.

PC vs A measures the net finite-sample effect of changing the order-zero graph;
A vs B isolates staging conditional on that same graph and separator.
"""
from itertools import combinations
import sys

import numpy as np
from scipy import stats

from simulate import (FisherZ, balanced_cut, gaussian_sample, metrics, pc_stable)
from review_experiments import g0_from_p, pvalues
from v2_experiments import hub_dag
from variants import variant_A, variant_B

ALPHA = 0.01
N = 1000
REPS = 25
KS = (4, 6, 8, 10, 12)
METHOD = "bonferroni"


def ordinary_from_g0(nodes, X, G0):
    """Ordinary PC continuation after an externally supplied order-zero graph."""
    sep0 = {frozenset(p): frozenset() for p in combinations(nodes, 2)
            if frozenset(p) not in G0}
    test = FisherZ(X, nodes)
    edges, sep = pc_stable(nodes, test, ALPHA, init_edges=G0,
                           start_l=1, sepsets=sep0)
    return edges, sep, test.calls


def mean_ci(d):
    """Paired mean difference and two-sided 95% t interval."""
    a = np.asarray(d, dtype=float)
    m = float(np.mean(a))
    if len(a) < 2:
        return m, float("nan"), float("nan")
    se = float(stats.sem(a))
    if se == 0:
        return m, m, m
    h = float(stats.t.ppf(0.975, len(a) - 1) * se)
    return m, m - h, m + h


def run_k(k):
    pc_f1 = []; a_f1 = []; b_f1 = []
    pc_vf1 = []; a_vf1 = []; b_vf1 = []
    pc_s = []; a_s = []; b_s = []
    pc_v = []; a_v = []; b_v = []
    cuts = 0
    for rep in range(REPS):
        rng = np.random.default_rng(31 + 977 * rep + k)
        nodes, true_edges, _ = hub_dag(k, 5, rng)
        X = gaussian_sample(nodes, true_edges, N, rng)

        t_pc = FisherZ(X, nodes)
        e_pc, s_pc = pc_stable(nodes, t_pc, ALPHA)
        m_pc = metrics(e_pc, s_pc, nodes, true_edges)

        G0 = g0_from_p(pvalues(X, nodes), METHOD)
        R1, packets = balanced_cut(nodes, G0, beta=0.3)
        if R1 is None:
            e_a, s_a, _ = ordinary_from_g0(nodes, X, G0)
            e_b, s_b = e_a, s_a
        else:
            cuts += 1
            e_a, s_a, _ = variant_A(nodes, X, G0, R1, packets, ALPHA)
            e_b, s_b, _ = variant_B(nodes, X, G0, R1, packets, ALPHA)
        m_a = metrics(e_a, s_a, nodes, true_edges)
        m_b = metrics(e_b, s_b, nodes, true_edges)

        for arr, val in ((pc_f1,m_pc['f1']), (a_f1,m_a['f1']), (b_f1,m_b['f1']),
                         (pc_vf1,m_pc['vf1']), (a_vf1,m_a['vf1']), (b_vf1,m_b['vf1']),
                         (pc_s,m_pc['skel_shd']), (a_s,m_a['skel_shd']), (b_s,m_b['skel_shd']),
                         (pc_v,m_pc['vshd']), (a_v,m_a['vshd']), (b_v,m_b['vshd'])):
            arr.append(val)

    da = np.asarray(a_s)-np.asarray(pc_s)
    db = np.asarray(b_s)-np.asarray(a_s)
    dva = np.asarray(a_v)-np.asarray(pc_v)
    dvb = np.asarray(b_v)-np.asarray(a_v)
    return dict(k=k,n=len(nodes),cuts=cuts,
                pc_f1=np.mean(pc_f1),a_f1=np.mean(a_f1),b_f1=np.mean(b_f1),
                pc_vf1=np.mean(pc_vf1),a_vf1=np.mean(a_vf1),b_vf1=np.mean(b_vf1),
                pc_s=np.mean(pc_s),a_s=np.mean(a_s),b_s=np.mean(b_s),
                pc_v=np.mean(pc_v),a_v=np.mean(a_v),b_v=np.mean(b_v),
                dA=mean_ci(da),dB=mean_ci(db),dvA=mean_ci(dva),dvB=mean_ci(dvb))


def main():
    ks = tuple(int(x) for x in sys.argv[1:] if x.isdigit()) or KS
    print("TRUTH-BASED COMPARISON: PC-stable vs schedule (A) vs schedule (B)")
    print(f"Hub family; N={N}; alpha={ALPHA}; {METHOD} order-zero; balanced cut; "
          f"{REPS} paired replicates per row\n")
    print("Main accuracy table")
    print(f"{'k':>3} {'n':>4} {'cuts':>7} | {'skel F1 PC':>10} {'A':>6} {'B':>6} | "
          f"{'vF1 PC':>7} {'A':>6} {'B':>6} | {'skSHD PC':>8} {'A':>6} {'B':>6}")
    print('-'*92)
    rows=[]
    for k in ks:
        r=run_k(k); rows.append(r)
        print(f"{k:>3} {r['n']:>4} {r['cuts']:>3}/{REPS:<3} | "
              f"{r['pc_f1']:>10.3f} {r['a_f1']:>6.3f} {r['b_f1']:>6.3f} | "
              f"{r['pc_vf1']:>7.3f} {r['a_vf1']:>6.3f} {r['b_vf1']:>6.3f} | "
              f"{r['pc_s']:>8.2f} {r['a_s']:>6.2f} {r['b_s']:>6.2f}")
    print("\nPaired mean differences [95% t CI]")
    print("             skeleton SHD                         v-structure SHD")
    print(" k      A-PC                B-A                 A-PC                B-A")
    for r in rows:
        f=lambda z:f"{z[0]:+.2f} [{z[1]:+.2f},{z[2]:+.2f}]"
        print(f"{r['k']:>2}  {f(r['dA']):>21} {f(r['dB']):>21} {f(r['dvA']):>21} {f(r['dvB']):>21}")


if __name__ == '__main__':
    main()
