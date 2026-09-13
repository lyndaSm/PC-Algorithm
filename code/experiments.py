"""
Experiment drivers for Section 9. Run:  python3 experiments.py all
"""
import sys
import time
from itertools import combinations

import numpy as np

from simulate import (ASIA_CPT, ASIA_EDGES, ASIA_NODES, FisherZ, GSquare,
                      asia_sample, discrete_sample,
                      enhanced_pc, gaussian_sample, metrics, min_vertex_cut, balanced_cut,
                      modular_dag, pc_stable, skeleton_of)

REPS = 50
ALPHA = 0.01


def summarise(rows):
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


# ---------------------------------------------------------------- Gaussian
def exp_gaussian():
    print("\n" + "=" * 88)
    print("9.1  LINEAR-GAUSSIAN MODULAR NETWORKS  (Fisher-z, alpha=%.2f, %d reps)"
          % (ALPHA, REPS))
    print("=" * 88)
    header = (f"{'k':>2} {'m':>2} {'n':>4} {'N':>6} | "
              f"{'PC F1':>6} {'Enh F1':>6} | {'PC tst':>8} {'Enh tst':>8} | "
              f"{'dec%':>5} {'|R1|':>5}")
    print(header)
    print("-" * len(header))
    results = []
    for k, m in [(2, 4), (3, 4), (4, 4), (4, 6)]:
        for N in (200, 1000, 5000):
            rows_pc, rows_en, tests_pc, tests_en = [], [], [], []
            dec, r1s = 0, []
            for rep in range(REPS):
                rng = np.random.default_rng(1000 + 97 * rep + 7 * k + m + N)
                nodes, edges = modular_dag(k, m, rng)
                truth = skeleton_of(edges)
                X = gaussian_sample(nodes, edges, N, rng)

                t = FisherZ(X, nodes)
                e1, s1 = pc_stable(nodes, t, ALPHA)
                tests_pc.append(t.calls)
                rows_pc.append(metrics(e1, s1, nodes, edges))

                t2 = FisherZ(X, nodes)
                e2, s2, info = enhanced_pc(nodes, t2, ALPHA)
                tests_en.append(t2.calls)
                rows_en.append(metrics(e2, s2, nodes, edges))
                if info["decomposed"]:
                    dec += 1
                    r1s.append(len(info["R1"]))
            a, b = summarise(rows_pc), summarise(rows_en)
            n = len(nodes)
            print(f"{k:>2} {m:>2} {n:>4} {N:>6} | "
                  f"{a['f1'][0]:>6.3f} {b['f1'][0]:>6.3f} | "
                  f"{np.mean(tests_pc):>8.0f} {np.mean(tests_en):>8.0f} | "
                  f"{100*dec/REPS:>5.0f} "
                  f"{(np.mean(r1s) if r1s else float('nan')):>5.1f}")
            results.append((k, m, n, N, a, b, np.mean(tests_pc),
                            np.mean(tests_en), 100 * dec / REPS,
                            np.mean(r1s) if r1s else float('nan')))
    return results


# ---------------------------------------------------------------- discrete
def exp_discrete(param="random"):
    print("\n" + "=" * 88)
    print("9.2  CHEST CLINIC, BINARY DATA, %s CPTs  (G^2, alpha=%.2f, %d reps)"
          % (param.upper(), ALPHA, REPS))
    print("=" * 88)
    header = (f"{'N':>7} | {'PC skSHD':>8} {'Enh skSHD':>9} | {'PC F1':>7} {'Enh F1':>7} "
              f"| {'PC tst':>7} {'Enh tst':>8} | {'dec%':>5} {'exact%':>7} {'|R1|':>5}")
    print(header)
    print("-" * len(header))
    out = []
    for N in (200, 1000, 5000, 20000, 100000):
        rp, re_, tp, te = [], [], [], []
        dec, exact, r1s = 0, 0, []
        for rep in range(REPS):
            rng = np.random.default_rng(50000 + 131 * rep + N)
            if param == 'published':
                X = asia_sample(N, rng)
            else:
                X = discrete_sample(ASIA_NODES, ASIA_EDGES, N, rng)
            t = GSquare(X, ASIA_NODES)
            e1, s1 = pc_stable(ASIA_NODES, t, ALPHA)
            tp.append(t.calls)
            rp.append(metrics(e1, s1, ASIA_NODES, ASIA_EDGES))
            t2 = GSquare(X, ASIA_NODES)
            e2, s2, info = enhanced_pc(ASIA_NODES, t2, ALPHA)
            te.append(t2.calls)
            mm = metrics(e2, s2, ASIA_NODES, ASIA_EDGES)
            re_.append(mm)
            if info["decomposed"]:
                dec += 1
                r1s.append(len(info["R1"]))
                if sorted(info["R1"]) == ["D", "O", "X"]:
                    exact += 1
        a, b = summarise(rp), summarise(re_)
        print(f"{N:>7} | {a['shd'][0]:>8.2f} {b['shd'][0]:>8.2f} | "
              f"{a['f1'][0]:>7.3f} {b['f1'][0]:>7.3f} | "
              f"{np.mean(tp):>7.0f} {np.mean(te):>8.0f} | "
              f"{100*dec/REPS:>5.0f} {100*exact/REPS:>7.0f} "
              f"{(np.mean(r1s) if r1s else float('nan')):>5.1f}")
        out.append((N, a, b, np.mean(tp), np.mean(te), 100 * dec / REPS,
                    100 * exact / REPS))
    return out


# ---------------------------------------------------------------- alpha0
def exp_alpha0():
    """Sensitivity of the *recommended balanced pipeline* to alpha_0.

    Earlier versions routed this experiment through ``enhanced_pc``, whose
    transparent reference separator is the exhaustive size<=3 minimum cut.
    That did not match the manuscript's stated balanced-cut pipeline.  Here
    the order-zero graph is built explicitly, the beta=0.3 balanced heuristic
    is applied, and schedule (A) continues from exactly that graph.  If the
    balanced cut is not found, Step 2's ordinary-PC continuation is used.
    """
    print("\n" + "=" * 96)
    print("SENSITIVITY TO THE ORDER-ZERO LEVEL alpha_0  "
          "(balanced cut beta=0.3; Gaussian k=3 m=4; alpha=%.2f; %d reps)"
          % (ALPHA, REPS))
    print("=" * 96)
    header = (f"{'N':>6} {'alpha0':>8} | {'found':>5} {'|R1|':>5} | "
              f"{'split err%':>10} {'lost edges':>11} | {'F1':>6} | {'tests':>7}")
    print(header)
    print("-" * len(header))
    rows = []
    for N in (200, 1000):
        for a0 in (0.05, 0.01, 0.001, 1e-4, 1e-6):
            found, splitbad, lost, f1s, tsts, r1s = 0, 0, [], [], [], []
            for rep in range(REPS):
                rng = np.random.default_rng(90000 + 211 * rep + N)
                nodes, edges = modular_dag(3, 4, rng)
                truth = skeleton_of(edges)
                X = gaussian_sample(nodes, edges, N, rng)

                # Step 1, retaining the same test object so the count includes
                # the marginal sweep and every higher-order CI test.
                test = FisherZ(X, nodes)
                G0, sep0 = set(), {}
                for x, y in combinations(nodes, 2):
                    pair = frozenset({x, y})
                    if test(x, y, [], a0):
                        sep0[pair] = frozenset()
                    else:
                        G0.add(pair)

                R1, packets = balanced_cut(nodes, G0, beta=0.3)
                if R1 is None:
                    est, sep = pc_stable(nodes, test, ALPHA, init_edges=G0,
                                         start_l=1, sepsets=sep0)
                    lost.append(0)
                else:
                    found += 1
                    r1s.append(len(R1))
                    group = {v: 0 for v in R1}
                    for i, comp in enumerate(packets, 1):
                        for v in comp:
                            group[v] = i
                    bad = [e for e in truth
                           if all(group.get(v, 0) > 0 for v in e)
                           and len({group[v] for v in e}) == 2]
                    lost.append(len(bad))
                    splitbad += bool(bad)

                    cross = {frozenset({x, y})
                             for i, j in combinations(range(len(packets)), 2)
                             for x in packets[i] for y in packets[j]}
                    est, sep = pc_stable(nodes, test, ALPHA, init_edges=G0,
                                         start_l=1, forbidden=cross,
                                         sepsets=sep0)

                m = metrics(est, sep, nodes, edges)
                f1s.append(m['f1'])
                tsts.append(test.calls)

            print(f"{N:>6} {a0:>8.0e} | {100*found/REPS:>5.0f} "
                  f"{(np.mean(r1s) if r1s else float('nan')):>5.1f} | "
                  f"{100*splitbad/REPS:>10.0f} {np.mean(lost):>11.2f} "
                  f"| {np.mean(f1s):>6.3f} | {np.mean(tsts):>7.0f}")
            rows.append((N, a0, 100 * found / REPS,
                         np.mean(r1s) if r1s else float('nan'),
                         100 * splitbad / REPS, np.mean(lost),
                         np.mean(f1s), np.mean(tsts)))
    return rows


# ------------------------------------------------------- decomposability
def exp_decomposability():
    print("\n" + "=" * 96)
    print("9.4  HOW OFTEN DOES A SMALL CUT EXIST?")
    print("This experiment uses exhaustive separators of size <= 3; it measures")
    print("small-separator existence, not the balanced-cut pipeline. Both order-zero")
    print("levels are shown to isolate the effect of multiplicity control.")
    print("=" * 96)
    header = (f"{'graph':>16} {'n':>4} {'N':>6} {'order-0':>12} | {'G0 dens':>8} "
              f"{'dec%':>6} {'|R1|':>6} {'max pkt':>8}")
    print(header)
    print("-" * len(header))
    rows = []
    specs = [("modular k=3 m=4", lambda r: modular_dag(3, 4, r)),
             ("modular k=4 m=6", lambda r: modular_dag(4, 6, r)),
             ("modular k=2 m=8", lambda r: modular_dag(2, 8, r))]
    for name, gen in specs:
        for N in (200, 1000, 5000):
            for level in ("uncorrected", "Bonferroni"):
                dens, dec, r1s, mx = [], 0, [], []
                for rep in range(REPS):
                    rng = np.random.default_rng(70000 + 173 * rep + N)
                    nodes, edges = gen(rng)
                    X = gaussian_sample(nodes, edges, N, rng)
                    n = len(nodes)
                    a0 = (ALPHA if level == "uncorrected"
                          else ALPHA / (n * (n - 1) / 2))
                    t = FisherZ(X, nodes)
                    G0 = set()
                    for x, y in combinations(nodes, 2):
                        if not t(x, y, [], a0):
                            G0.add(frozenset({x, y}))
                    dens.append(len(G0) / (n * (n - 1) / 2))
                    R1, packets = min_vertex_cut(nodes, G0, max_size=3)
                    if R1 is not None:
                        dec += 1
                        r1s.append(len(R1))
                        mx.append(max(len(c) for c in packets))
                print(f"{name:>16} {len(nodes):>4} {N:>6} {level:>12} | "
                      f"{np.mean(dens):>8.2f} "
                      f"{100*dec/REPS:>6.0f} "
                      f"{(np.mean(r1s) if r1s else float('nan')):>6.1f} "
                      f"{(np.mean(mx) if mx else float('nan')):>8.1f}")
                rows.append((name, len(nodes), N, level, np.mean(dens),
                             100 * dec / REPS,
                             np.mean(r1s) if r1s else float('nan'),
                             np.mean(mx) if mx else float('nan')))
    return rows





# ------------------------------------------------------- parallel wall clock
def exp_parallel():
    """Work partitioning implied by Corollary 6.

    Executes Algorithm 1 in decomposed form: each packet-local search is run
    separately over G0[R0^i u R1] restricted to pairs inside R0^i, then a single
    interface pass handles every pair touching R1. Corollary 6 guarantees the
    packet-local searches are mutually independent, so on k processors their
    cost is the slowest of them rather than their sum.

    Reported speedup is (order-0 + sum of packets + interface) divided by
    (order-0 + slowest packet + interface). This measures the partition of work
    only: there is no process spawning, scheduling or communication cost in
    these figures, so it is an upper bound on any real implementation.
    """
    import time as _t
    from simulate import min_vertex_cut

    print("\n" + "=" * 88)
    print("9.5  WORK PARTITIONING AND CRITICAL PATH  (Gaussian, alpha=%.2f, %d reps)"
          % (ALPHA, REPS))
    print("=" * 88)
    header = (f"{'k':>2} {'m':>2} {'n':>4} {'N':>6} | {'order-0':>8} "
              f"{'packets':>8} {'slowest':>8} {'interface':>10} | "
              f"{'pkt share':>10} {'speedup':>8} {'agree':>6}")
    print(header)
    print("-" * len(header))
    rows = []
    for k, m in [(3, 4), (4, 4), (4, 6), (6, 4)]:
        for N in (1000, 5000):
            z, ptot, crit, ifc, sp, share = [], [], [], [], [], []
            agree = 0
            used = 0
            for rep in range(REPS):
                rng = np.random.default_rng(1000 + 97 * rep + 7 * k + m + N)
                nodes, edges = modular_dag(k, m, rng)
                X = gaussian_sample(nodes, edges, N, rng)

                # reference: Algorithm 1 as a single pass
                tref = FisherZ(X, nodes)
                ref_edges, _, info = enhanced_pc(nodes, tref, ALPHA)
                if not info["decomposed"]:
                    continue
                R1, packets = info["R1"], info["packets"]
                used += 1

                # --- stage 1: order-zero sweep
                t0 = _t.perf_counter()
                t2 = FisherZ(X, nodes)
                G0, sep = set(), {}
                for x, y in combinations(nodes, 2):
                    if t2(x, y, [], ALPHA):
                        sep[frozenset({x, y})] = frozenset()
                    else:
                        G0.add(frozenset({x, y}))
                min_vertex_cut(nodes, G0, max_size=3)
                t_zero = (_t.perf_counter() - t0) * 1000

                # --- stage 2: packet-local searches, independently timed
                pkt_times, pkt_edges = [], set()
                for c in packets:
                    scope = sorted(set(c) | set(R1))
                    init = {e for e in G0 if all(v in scope for v in e)}
                    forb = {e for e in init if not all(v in c for v in e)}
                    t0 = _t.perf_counter()
                    tl = FisherZ(X, nodes)
                    ee, _ = pc_stable(scope, tl, ALPHA, init_edges=init,
                                      start_l=1, forbidden=forb)
                    pkt_times.append((_t.perf_counter() - t0) * 1000)
                    pkt_edges |= {e for e in ee if all(v in c for v in e)}

                # --- stage 3: interface pass over pairs touching R1
                inside = {e for c in packets for e in G0
                          if all(v in c for v in e)}
                cross = {frozenset({x, y})
                         for a, b in combinations(range(len(packets)), 2)
                         for x in packets[a] for y in packets[b]}
                start = (G0 - inside) | pkt_edges
                t0 = _t.perf_counter()
                ti = FisherZ(X, nodes)
                fin, _ = pc_stable(nodes, ti, ALPHA, init_edges=start,
                                   start_l=1, forbidden=pkt_edges | cross)
                t_iface = (_t.perf_counter() - t0) * 1000

                if fin == ref_edges:
                    agree += 1
                total_pkt = sum(pkt_times)
                critical = max(pkt_times)
                serial = t_zero + total_pkt + t_iface
                par = t_zero + critical + t_iface
                z.append(t_zero); ptot.append(total_pkt)
                crit.append(critical); ifc.append(t_iface)
                sp.append(serial / par)
                share.append(total_pkt / serial)
            if not used:
                continue
            print(f"{k:>2} {m:>2} {len(nodes):>4} {N:>6} | {np.mean(z):>8.1f} "
                  f"{np.mean(ptot):>8.1f} {np.mean(crit):>8.1f} "
                  f"{np.mean(ifc):>10.1f} | {100*np.mean(share):>9.0f}% "
                  f"{np.mean(sp):>8.2f} {100*agree/used:>5.0f}%")
            rows.append((k, m, len(nodes), N, np.mean(z), np.mean(ptot),
                         np.mean(crit), np.mean(ifc), 100 * np.mean(share),
                         np.mean(sp), 100 * agree / used))
    return rows


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if what in ("gaussian", "all"):
        exp_gaussian()
    if what in ("discrete", "all"):
        exp_discrete("random")
        exp_discrete("published")
    if what in ("alpha0", "all"):
        exp_alpha0()
    if what in ("decomposability", "all"):
        exp_decomposability()
    if what in ("parallel", "all"):
        exp_parallel()
    print("\nelapsed %.1f s" % (time.time() - t0))