"""
Experiments for the v2 paper. Run one at a time:

    python3 v2_experiments.py e1      multiplicity control at order 0
    python3 v2_experiments.py e2      cut objective: minimum vs balanced
    python3 v2_experiments.py e3      stage timings and parallel speedup
    python3 v2_experiments.py e4      separator selection: the open problem
"""
import sys
import time
from itertools import combinations

import numpy as np

from simulate import (FisherZ, balanced_cut, components, gaussian_sample,
                      metrics, min_vertex_cut, pc_stable, true_cpdag)

ALPHA = 0.01
N = 1000
REPS = 25


# --------------------------------------------------------------- generators
def hub_dag(k, m, rng, p_extra=0.12):
    """k modules; each module sink feeds a shared hub; hub feeds two readouts.
    The separator is a HIGH-degree vertex set."""
    nodes, edges, sinks = [], [], []
    for i in range(k):
        mod = [f"m{i}_{j}" for j in range(m)]
        nodes += mod
        for a in range(m):
            for b in range(a + 1, m):
                if b == a + 1 or rng.random() < p_extra:
                    edges.append((mod[a], mod[b]))
        sinks.append(mod[-1])
    nodes += ["h", "r1", "r2"]
    for s in sinks:
        edges.append((s, "h"))
    edges += [("h", "r1"), ("h", "r2")]
    return nodes, edges, ["h", "r1", "r2"]


def chain_dag(k, m, rng, p_extra=0.12):
    """Communities in a chain, joined by LOW-degree mediators s_i."""
    nodes, edges, seps = [], [], []
    prev = None
    for i in range(k):
        mod = [f"c{i}_{j}" for j in range(m)]
        nodes += mod
        for a in range(m):
            for b in range(a + 1, m):
                if b == a + 1 or rng.random() < p_extra:
                    edges.append((mod[a], mod[b]))
        if prev is not None:
            s = f"s{i}"
            nodes.append(s)
            seps.append(s)
            edges += [(prev, s), (s, mod[0])]
        prev = mod[-1]
    return nodes, edges, seps


def module_of(v):
    return v.split("_")[0] if ("_" in v) else "SEP"


def order_zero(X, nodes, alpha):
    t = FisherZ(X, nodes)
    G0 = set()
    for x, y in combinations(nodes, 2):
        if not t(x, y, [], alpha):
            G0.add(frozenset({x, y}))
    return G0


# ------------------------------------------------------------------- E1
def e1():
    print("E1  MULTIPLICITY CONTROL AT ORDER ZERO   "
          f"(hub family, N={N}, alpha={ALPHA}, {REPS} reps)")
    print("Packets computed by removing the TRUE separator, to isolate the")
    print("multiplicity effect from the separator-selection effect.\n")
    hdr = (f"{'k':>3} {'n':>4} {'order-0 level':>16} | {'|G0|':>6} "
           f"{'false xmod edges':>17} {'packets':>8} {'max pkt':>8} "
           f"{'exact':>6}")
    print(hdr)
    print("-" * len(hdr))
    for k in (6, 8, 10, 12):
        m = 5
        for label in ("uncorrected", "Bonferroni"):
            fe, np_, mx, ex, sz = [], [], [], 0, []
            for rep in range(REPS):
                rng = np.random.default_rng(31 + 977 * rep + k)
                nodes, edges, truesep = hub_dag(k, m, rng)
                X = gaussian_sample(nodes, edges, N, rng)
                n = len(nodes)
                a0 = ALPHA if label == "uncorrected" else ALPHA / (n * (n - 1) / 2)
                G0 = order_zero(X, nodes, a0)
                sz.append(len(G0))
                fe.append(sum(1 for e in G0
                              if len({module_of(v) for v in e}) == 2
                              and all(module_of(v) != "SEP" for v in e)))
                comps = components(set(nodes) - set(truesep), G0)
                np_.append(len(comps))
                mx.append(max(len(c) for c in comps) if comps else 0)
                if len(comps) == k and max(len(c) for c in comps) == m:
                    ex += 1
            print(f"{k:>3} {len(nodes):>4} {label:>16} | {np.mean(sz):>6.0f} "
                  f"{np.mean(fe):>17.1f} {np.mean(np_):>8.1f} "
                  f"{np.mean(mx):>8.1f} {100*ex/REPS:>5.0f}%")


# ------------------------------------------------------------------- E2
def e2():
    print("E2  CUT OBJECTIVE: MINIMUM VERSUS BALANCED   "
          f"(hub family, Bonferroni order-0, N={N}, {REPS} reps)\n")
    hdr = (f"{'k':>3} {'n':>4} {'objective':>12} | {'found':>6} {'|R1|':>6} "
           f"{'packets':>8} {'max pkt':>8} {'ideal max':>10}")
    print(hdr)
    print("-" * len(hdr))
    for k in (6, 8, 10, 12):
        m = 5
        for obj in ("minimum", "balanced"):
            r1, npk, mx, found = [], [], [], 0
            for rep in range(REPS):
                rng = np.random.default_rng(31 + 977 * rep + k)
                nodes, edges, _ = hub_dag(k, m, rng)
                X = gaussian_sample(nodes, edges, N, rng)
                n = len(nodes)
                G0 = order_zero(X, nodes, ALPHA / (n * (n - 1) / 2))
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
            print(f"{k:>3} {len(nodes):>4} {obj:>12} | {100*found/REPS:>5.0f}% "
                  f"{np.mean(r1):>6.1f} {np.mean(npk):>8.1f} "
                  f"{np.mean(mx):>8.1f} {m:>10}")


# ------------------------------------------------------------------- E3
def e3():
    print("E3  STAGE TIMINGS AND PARALLEL SPEEDUP   "
          f"(hub family, Bonferroni order-0, balanced cut, N={N}, {REPS} reps)\n")
    hdr = (f"{'k':>3} {'n':>4} | {'order-0':>8} {'packets':>8} {'slowest':>8} "
           f"{'interface':>10} | {'share':>6} {'speedup':>8} {'agree':>6}"
           f" || CI-TEST WORK: {'ord-0':>6} {'pkts':>6} {'slowest':>7} "
           f"{'iface':>6} {'share':>6} {'speedup':>8} {'spd*':>8}")
    print(hdr)
    print("-" * len(hdr))
    for k in (4, 6, 8, 10, 12):
        m = 5
        z, ps, pm, ifc, sh, sp, ag, used = [], [], [], [], [], [], 0, 0
        wz, wps, wpm, wifc, wsh, wsp = [], [], [], [], [], []
        wsp2 = []
        for rep in range(REPS):
            rng = np.random.default_rng(31 + 977 * rep + k)
            nodes, edges, _ = hub_dag(k, m, rng)
            X = gaussian_sample(nodes, edges, N, rng)
            n = len(nodes)
            tb = FisherZ(X, nodes)
            e_pc, _ = pc_stable(nodes, tb, ALPHA)

            t0 = time.perf_counter()
            G0 = order_zero(X, nodes, ALPHA / (n * (n - 1) / 2))
            R1, pk = balanced_cut(nodes, G0, beta=0.3)
            tz = time.perf_counter() - t0
            if R1 is None:
                continue
            used += 1
            pt, pe, pw = [], set(), []
            for c in pk:
                scope = sorted(set(c) | set(R1))
                init = {e for e in G0 if all(v in scope for v in e)}
                forb = {e for e in init if not all(v in c for v in e)}
                t0 = time.perf_counter()
                tl = FisherZ(X, nodes)
                ee, _ = pc_stable(scope, tl, ALPHA, init_edges=init,
                                  start_l=1, forbidden=forb)
                pt.append(time.perf_counter() - t0)
                pw.append(tl.calls)
                pe |= {e for e in ee if all(v in c for v in e)}
            inside = {e for c in pk for e in G0 if all(v in c for v in e)}
            cross = {frozenset({x, y})
                     for a, b in combinations(range(len(pk)), 2)
                     for x in pk[a] for y in pk[b]}
            t0 = time.perf_counter()
            ti = FisherZ(X, nodes)
            e_en, _ = pc_stable(nodes, ti, ALPHA,
                                init_edges=(G0 - inside) | pe,
                                start_l=1, forbidden=pe | cross)
            tif = time.perf_counter() - t0
            ser, par = tz + sum(pt) + tif, tz + max(pt) + tif
            # Deterministic work measure: CI tests, not wall-clock seconds.
            w0 = n * (n - 1) // 2
            wser = w0 + sum(pw) + ti.calls
            wpar = w0 + max(pw) + ti.calls
            # Second model: the order-zero sweep is embarrassingly parallel
            # (all pairs are independent), so charge it to the parallel part
            # too, across the same len(pk) processors. This saving is NOT
            # attributable to the decomposition: any PC implementation can
            # take it. It is reported so the two effects are not confused.
            wpar2 = w0 / len(pk) + max(pw) + ti.calls
            wsp2.append(wser / wpar2)
            wz.append(w0); wps.append(sum(pw)); wpm.append(max(pw))
            wifc.append(ti.calls)
            wsh.append(sum(pw) / wser); wsp.append(wser / wpar)
            z.append(tz * 1000)
            ps.append(sum(pt) * 1000)
            pm.append(max(pt) * 1000)
            ifc.append(tif * 1000)
            sh.append(sum(pt) / ser)
            sp.append(ser / par)
            ag += (e_en == e_pc)
        print(f"{k:>3} {len(nodes):>4} | {np.mean(z):>8.0f} {np.mean(ps):>8.0f} "
              f"{np.mean(pm):>8.0f} {np.mean(ifc):>10.0f} | "
              f"{100*np.mean(sh):>5.0f}% {np.mean(sp):>8.2f} "
              f"{100*ag/max(used,1):>5.0f}%"
              f" || {np.mean(wz):>6.0f} {np.mean(wps):>6.0f} {np.mean(wpm):>7.0f} "
              f"{np.mean(wifc):>6.0f} {100*np.mean(wsh):>5.0f}% "
              f"{np.mean(wsp):>8.2f} {np.mean(wsp2):>8.2f}"
              f"   [cut found {used}/{REPS}; timings and agreement are over "
              f"those {used}, being undefined without a cut]")


# ------------------------------------------------------------------- E4
def e4():
    print("E4  SEPARATOR SELECTION: THE OPEN PROBLEM   "
          f"(chain family, low-degree mediators, Bonferroni order-0, "
          f"N={N}, {REPS} reps)")
    print("The oracle row uses the TRUE mediators as the separator; the two")
    print("heuristics must find them from G0 alone.\n")
    hdr = (f"{'k':>3} {'n':>4} {'separator':>12} | {'found':>6} {'|R1|':>6} "
           f"{'packets':>8} {'max pkt':>8} | {'share':>6} {'speedup':>8}")
    print(hdr)
    print("-" * len(hdr))
    for k in (4, 6, 8):
        m = 5
        for how in ("minimum", "greedy-degree", "oracle"):
            r1, npk, mx, sh, sp, found = [], [], [], [], [], 0
            for rep in range(REPS):
                rng = np.random.default_rng(555 + 37 * rep + k)
                nodes, edges, truesep = chain_dag(k, m, rng)
                X = gaussian_sample(nodes, edges, N, rng)
                n = len(nodes)
                t0 = time.perf_counter()
                G0 = order_zero(X, nodes, ALPHA / (n * (n - 1) / 2))
                if how == "minimum":
                    R1, pk = min_vertex_cut(nodes, G0, max_size=4)
                elif how == "greedy-degree":
                    R1, pk = balanced_cut(nodes, G0, beta=0.3)
                else:
                    R1 = sorted(truesep)
                    pk = components(set(nodes) - set(R1), G0)
                    if len([c for c in pk if len(c) >= 2]) < 2:
                        R1, pk = None, None
                tz = time.perf_counter() - t0
                if R1 is None:
                    continue
                found += 1
                r1.append(len(R1))
                npk.append(len(pk))
                mx.append(max(len(c) for c in pk))
                pt, pe = [], set()
                for c in pk:
                    scope = sorted(set(c) | set(R1))
                    init = {e for e in G0 if all(v in scope for v in e)}
                    forb = {e for e in init if not all(v in c for v in e)}
                    t0 = time.perf_counter()
                    tl = FisherZ(X, nodes)
                    ee, _ = pc_stable(scope, tl, ALPHA, init_edges=init,
                                      start_l=1, forbidden=forb)
                    pt.append(time.perf_counter() - t0)
                    pe |= {e for e in ee if all(v in c for v in e)}
                inside = {e for c in pk for e in G0 if all(v in c for v in e)}
                cross = {frozenset({x, y})
                         for a, b in combinations(range(len(pk)), 2)
                         for x in pk[a] for y in pk[b]}
                t0 = time.perf_counter()
                ti = FisherZ(X, nodes)
                pc_stable(nodes, ti, ALPHA, init_edges=(G0 - inside) | pe,
                          start_l=1, forbidden=pe | cross)
                tif = time.perf_counter() - t0
                ser, par = tz + sum(pt) + tif, tz + max(pt) + tif
                sh.append(sum(pt) / ser)
                sp.append(ser / par)
            if not found:
                print(f"{k:>3} {n:>4} {how:>12} |    0%")
                continue
            print(f"{k:>3} {n:>4} {how:>12} | {100*found/REPS:>5.0f}% "
                  f"{np.mean(r1):>6.1f} {np.mean(npk):>8.1f} "
                  f"{np.mean(mx):>8.1f} | {100*np.mean(sh):>5.0f}% "
                  f"{np.mean(sp):>8.2f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "e1"
    t0 = time.time()
    {"e1": e1, "e2": e2, "e3": e3, "e4": e4}[which]()
    print(f"\nelapsed {time.time()-t0:.0f}s")
