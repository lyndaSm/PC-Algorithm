"""
Finite-sample simulation study for the Enhanced PC paper (Section 9).

Self-contained: implements the linear-Gaussian and discrete data generators,
the Fisher-z and G^2 conditional independence tests, PC-stable, and Enhanced PC
with the order-zero packet decomposition. No external causal-discovery library.

Usage:  python3 simulate.py gaussian | discrete | alpha0 | decomposability | all
"""
import sys
import time
from itertools import combinations

import numpy as np
from scipy import stats

RNG_MASTER = 20260810

# =====================================================================
#  Graph utilities
# =====================================================================


def components(vertices, edgeset):
    vertices = set(vertices)
    comps, seen = [], set()
    for v in sorted(vertices):
        if v in seen:
            continue
        comp, stack = set(), [v]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            for w in vertices:
                if w not in comp and frozenset({u, w}) in edgeset:
                    stack.append(w)
        seen |= comp
        comps.append(sorted(comp))
    return comps


def min_vertex_cut(vertices, edgeset, max_size=4):
    """Smallest vertex set whose deletion disconnects the graph.

    Exhaustive up to max_size; adequate for the sizes used here and keeps the
    implementation transparent. Returns (cut, packets) or (None, None).
    """
    vertices = sorted(vertices)
    if len(components(vertices, edgeset)) >= 2:
        return [], components(vertices, edgeset)
    for size in range(1, max_size + 1):
        for cand in combinations(vertices, size):
            rest = set(vertices) - set(cand)
            if not rest:
                continue
            comps = components(rest, edgeset)
            if len(comps) >= 2:
                return list(cand), comps
    return None, None


def skeleton_of(edges):
    return {frozenset(e) for e in edges}


def _meek_r1_r3_closure(skel, directed, nodes):
    """Close a partially oriented skeleton under the three Meek rules used by PC.

    For observational PC without background knowledge, repeated R1--R3 is the
    standard completion step (Kalisch & Buehlmann, 2007, Algorithm 2).  The
    implementation evaluates each rule against the current graph and applies
    newly implied orientations in deterministic lexicographic sweeps.

    If finite-sample collider decisions imply both directions for the same
    undirected edge in one sweep, the edge is left unoriented in that sweep.
    Under an oracle this conflict cannot occur.
    """
    skel = set(skel)
    directed = set(directed)

    def adjacent(a, b):
        return frozenset({a, b}) in skel

    def undirected(a, b):
        e = frozenset({a, b})
        return e in skel and (a, b) not in directed and (b, a) not in directed

    while True:
        proposed = set()
        for e in sorted(skel, key=lambda q: tuple(sorted(q))):
            a, b = sorted(e)
            if not undirected(a, b):
                continue
            for x, y in ((a, b), (b, a)):
                # R1: z -> x - y with z and y non-adjacent => x -> y.
                if any((z, x) in directed and z != y and not adjacent(z, y)
                       for z in nodes):
                    proposed.add((x, y))
                    continue

                # R2: x -> z -> y and x - y => x -> y.
                if any((x, z) in directed and (z, y) in directed
                       for z in nodes):
                    proposed.add((x, y))
                    continue

                # R3: x-c, x-d, c->y, d->y, c and d non-adjacent
                #     => x->y.
                cand = [z for z in nodes if z not in (x, y)
                        and undirected(x, z) and (z, y) in directed]
                if any(not adjacent(c, d) for c, d in combinations(cand, 2)):
                    proposed.add((x, y))

        # A sound oracle input never proposes both directions.  With fallible
        # tests, do not let a contradiction propagate further orientations.
        conflict_pairs = {frozenset({a, b}) for a, b in proposed
                          if (b, a) in proposed}
        proposed = {e for e in proposed if frozenset(e) not in conflict_pairs}
        new = {e for e in proposed if e not in directed
               and (e[1], e[0]) not in directed}
        if not new:
            break
        directed |= new

    undirected_edges = {e for e in skel
                        if tuple(sorted(e)) not in directed
                        and tuple(sorted(e))[::-1] not in directed}
    return directed, undirected_edges


def cpdag(edges, sepsets, nodes):
    """Orient a learned skeleton using PC collider decisions and Meek R1--R3.

    With oracle CI information this is the CPDAG.  With finite-sample CI errors,
    incompatible collider decisions can occur; conflicting arrow proposals on
    one edge are conservatively left undirected rather than propagated.  This
    convention is deterministic and is used identically for every method in the
    simulation comparisons.
    """
    skel = {frozenset(e) for e in edges}

    # Collect all collider-implied arrow proposals first.  This makes the
    # orientation convention independent of triple iteration order.
    proposed = set()
    for b in nodes:
        nb = sorted({v for v in nodes if frozenset({v, b}) in skel})
        for a, c in combinations(nb, 2):
            if frozenset({a, c}) in skel:
                continue
            S = sepsets.get(frozenset({a, c}))
            if S is not None and b not in S:
                proposed.add((a, b))
                proposed.add((c, b))

    conflict_pairs = {frozenset({a, b}) for a, b in proposed
                      if (b, a) in proposed}
    directed = {e for e in proposed if frozenset(e) not in conflict_pairs}
    return _meek_r1_r3_closure(skel, directed, nodes)


def true_cpdag(nodes, edges):
    """Exact CPDAG of a DAG from its skeleton and unshielded colliders.

    V-structures are oriented directly from the generating DAG and then the
    standard PC Meek rules R1--R3 are applied to closure.
    """
    skel = {frozenset(e) for e in edges}
    parents = {v: set() for v in nodes}
    for u, w in edges:
        parents[w].add(u)
    directed = set()
    for c in nodes:
        for a, b in combinations(sorted(parents[c]), 2):
            if frozenset({a, b}) not in skel:
                directed.add((a, c))
                directed.add((b, c))
    return _meek_r1_r3_closure(skel, directed, nodes)

def shd_cpdag(da, ua, db, ub, nodes):
    """Structural Hamming distance between two CPDAGs given as (directed,
    undirected) edge sets."""
    d = 0
    all_pairs = {frozenset(p) for p in combinations(nodes, 2)}
    for p in all_pairs:
        x, y = sorted(p)
        in_a = p in ua or (x, y) in da or (y, x) in da
        in_b = p in ub or (x, y) in db or (y, x) in db
        if in_a != in_b:
            d += 1
        elif in_a:
            sa = ("u" if p in ua else ("f" if (x, y) in da else "b"))
            sb = ("u" if p in ub else ("f" if (x, y) in db else "b"))
            if sa != sb:
                d += 1
    return d


# =====================================================================
#  Data generators
# =====================================================================


def modular_dag(k, m, rng, p_extra=0.12):
    """k modules of m vertices; within a module a random sparse DAG; each
    module sink feeds a shared hub h; h feeds two readout vertices."""
    nodes, edges = [], []
    sinks = []
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
    return nodes, edges


ASIA_NODES = ["A", "T", "S", "L", "B", "O", "X", "D"]
ASIA_EDGES = [("A", "T"), ("S", "L"), ("S", "B"), ("T", "O"),
              ("L", "O"), ("O", "X"), ("O", "D"), ("B", "D")]

# Published Chest Clinic parameters (Lauritzen and Spiegelhalter 1988), as
# distributed with the bnlearn package. P(v = yes | parents).
#   Note: "Either" (O) is a DETERMINISTIC logical OR of T and L. That
#   determinism violates faithfulness: conditioning on {T,L} fixes O, which
#   makes O independent of X and of D given {T,L} even though O->X and O->D are
#   edges of the graph. No constraint-based algorithm can recover this network.
ASIA_CPT = {
    "A": 0.01,
    "S": 0.50,
    "T": {1: 0.05, 0: 0.01},                 # | A
    "L": {1: 0.10, 0: 0.01},                 # | S
    "B": {1: 0.60, 0: 0.30},                 # | S
    "X": {1: 0.98, 0: 0.05},                 # | O
    "D": {(1, 1): 0.90, (1, 0): 0.70,        # | (O, B)
          (0, 1): 0.80, (0, 0): 0.10},
}


def asia_sample(n, rng):
    """Sample the Chest Clinic network from the published parameters.

    Column order matches ASIA_NODES.
    """
    i = {v: k for k, v in enumerate(ASIA_NODES)}
    X = np.zeros((n, len(ASIA_NODES)), dtype=int)
    a = (rng.random(n) < ASIA_CPT["A"]).astype(int)
    s = (rng.random(n) < ASIA_CPT["S"]).astype(int)
    t = (rng.random(n) < np.where(a == 1, ASIA_CPT["T"][1],
                                  ASIA_CPT["T"][0])).astype(int)
    ll = (rng.random(n) < np.where(s == 1, ASIA_CPT["L"][1],
                                   ASIA_CPT["L"][0])).astype(int)
    b = (rng.random(n) < np.where(s == 1, ASIA_CPT["B"][1],
                                  ASIA_CPT["B"][0])).astype(int)
    o = ((t == 1) | (ll == 1)).astype(int)          # deterministic OR
    x = (rng.random(n) < np.where(o == 1, ASIA_CPT["X"][1],
                                  ASIA_CPT["X"][0])).astype(int)
    pd = np.where(o == 1,
                  np.where(b == 1, ASIA_CPT["D"][(1, 1)], ASIA_CPT["D"][(1, 0)]),
                  np.where(b == 1, ASIA_CPT["D"][(0, 1)], ASIA_CPT["D"][(0, 0)]))
    d = (rng.random(n) < pd).astype(int)
    for name, col in zip("ATSLBOXD", [a, t, s, ll, b, o, x, d]):
        X[:, i[name]] = col
    return X


def gaussian_sample(nodes, edges, n, rng, lo=0.4, hi=0.9):
    """Linear SEM  X_v = sum_p b_pv X_p + e_v,  e_v ~ N(0,1)."""
    idx = {v: i for i, v in enumerate(nodes)}
    p = len(nodes)
    B = np.zeros((p, p))
    for u, v in edges:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        B[idx[u], idx[v]] = sign * rng.uniform(lo, hi)
    # topological order
    order, remaining = [], list(nodes)
    par = {v: {u for u, w in edges if w == v} for v in nodes}
    while remaining:
        for v in list(remaining):
            if par[v] <= set(order):
                order.append(v)
                remaining.remove(v)
                break
    X = np.zeros((n, p))
    for v in order:
        j = idx[v]
        X[:, j] = rng.standard_normal(n)
        for u in par[v]:
            X[:, j] += B[idx[u], j] * X[:, idx[u]]
    return X


def discrete_sample(nodes, edges, n, rng, strength=0.85):
    """Binary variables; each vertex is a noisy-OR-like function of parents."""
    idx = {v: i for i, v in enumerate(nodes)}
    par = {v: sorted({u for u, w in edges if w == v}) for v in nodes}
    order, remaining = [], list(nodes)
    while remaining:
        for v in list(remaining):
            if set(par[v]) <= set(order):
                order.append(v)
                remaining.remove(v)
                break
    cpt = {}
    for v in nodes:
        k = len(par[v])
        # P(v=1 | parent configuration), well away from 0 and 1
        cpt[v] = rng.uniform(0.5 - strength / 2, 0.5 + strength / 2, size=2 ** k)
    X = np.zeros((n, len(nodes)), dtype=int)
    for v in order:
        k = len(par[v])
        if k == 0:
            probs = np.full(n, cpt[v][0])
        else:
            code = np.zeros(n, dtype=int)
            for b, u in enumerate(par[v]):
                code += X[:, idx[u]] << b
            probs = cpt[v][code]
        X[:, idx[v]] = (rng.random(n) < probs).astype(int)
    return X


# =====================================================================
#  Conditional independence tests
# =====================================================================


class FisherZ:
    name = "Fisher-z"

    def __init__(self, X, nodes):
        self.C = np.corrcoef(X, rowvar=False)
        self.n = X.shape[0]
        self.idx = {v: i for i, v in enumerate(nodes)}
        self.calls = 0

    def __call__(self, x, y, S, alpha):
        self.calls += 1
        ix, iy = self.idx[x], self.idx[y]
        iS = [self.idx[v] for v in S]
        sub = self.C[np.ix_([ix, iy] + iS, [ix, iy] + iS)]
        try:
            P = np.linalg.pinv(sub)
            den = P[0, 0] * P[1, 1]
            if not np.isfinite(den) or den <= 0:
                return False  # unavailable/ill-conditioned: retain the edge
            r = -P[0, 1] / np.sqrt(den)
        except Exception:
            return False  # absence of a usable test is not evidence of independence
        if not np.isfinite(r):
            return False
        r = float(np.clip(r, -0.999999, 0.999999))
        dof = self.n - len(S) - 3
        if dof < 1:
            return False
        z = 0.5 * np.sqrt(dof) * np.log((1 + r) / (1 - r))
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        return pval > alpha


class GSquare:
    name = "G^2"

    def __init__(self, X, nodes):
        self.X = X
        self.idx = {v: i for i, v in enumerate(nodes)}
        self.n = X.shape[0]
        self.calls = 0

    def __call__(self, x, y, S, alpha):
        self.calls += 1
        ix, iy = self.idx[x], self.idx[y]
        S = list(S)
        if S:
            code = np.zeros(self.n, dtype=int)
            for b, v in enumerate(S):
                code += self.X[:, self.idx[v]] << b
        else:
            code = np.zeros(self.n, dtype=int)
        g, dof = 0.0, 0
        for c in np.unique(code):
            sel = code == c
            if sel.sum() < 5:
                continue
            a = self.X[sel, ix]
            b = self.X[sel, iy]
            tab = np.zeros((2, 2))
            for i in (0, 1):
                for j in (0, 1):
                    tab[i, j] = np.sum((a == i) & (b == j))
            tot = tab.sum()
            if tot == 0:
                continue
            exp = np.outer(tab.sum(1), tab.sum(0)) / tot
            # Stratum size alone is not enough: the chi-square approximation to
            # the G^2 null is driven by the EXPECTED cell counts. Discard a
            # stratum in which any cell that can occur has expected count below
            # one. (Verified: identical to the size-only rule on all 150
            # replicates at N = 200, 1000, 5000 and on N = 20000, so no
            # published number depends on it. The stricter all-cells->=5 rule
            # does change results -- conservatively, retaining more edges.)
            occ = np.outer(tab.sum(1) > 0, tab.sum(0) > 0)
            if occ.any() and exp[occ].min() < 1.0:
                continue
            mask = (tab > 0) & (exp > 0)
            if not mask.any():
                continue
            g += 2 * np.sum(tab[mask] * np.log(tab[mask] / exp[mask]))
            r = int((tab.sum(1) > 0).sum())
            cc = int((tab.sum(0) > 0).sum())
            dof += max(0, (r - 1) * (cc - 1))
        if dof == 0:
            # No usable stratum: the test is unavailable. Absence of data is not
            # evidence of independence, so report dependence and retain the edge.
            return False
        return stats.chi2.sf(g, dof) > alpha


# =====================================================================
#  Algorithms
# =====================================================================


def pc_stable(nodes, test, alpha, init_edges=None, start_l=0,
              forbidden=None, sepsets=None, max_l=None):
    """Order-independent skeleton search. forbidden: pairs never tested."""
    if init_edges is None:
        edges = {frozenset(p) for p in combinations(nodes, 2)}
    else:
        edges = set(init_edges)
    sep = dict(sepsets or {})
    l = start_l
    while True:
        adj = {v: {w for w in nodes if frozenset({v, w}) in edges} for v in nodes}
        cont = False
        for e in sorted(edges, key=lambda s: sorted(s)):
            if e not in edges:
                continue
            x, y = sorted(e)
            if forbidden and e in forbidden:
                continue
            done = False
            tried = set()
            for a, b in ((x, y), (y, x)):
                pool = sorted(adj[a] - {b})
                if len(pool) >= l:
                    cont = True
                for S in combinations(pool, l):
                    if frozenset(S) in tried:
                        continue  # same conditioning set from the other ordering
                    tried.add(frozenset(S))
                    if test(a, b, list(S), alpha):
                        edges.discard(e)
                        sep[e] = frozenset(S)
                        done = True
                        break
                if done:
                    break
        if not cont or (max_l is not None and l >= max_l):
            break
        l += 1
    return edges, sep


def enhanced_pc(nodes, test, alpha, alpha0=None, max_cut=3):
    """Order-zero decomposition, then packet-local and interface search."""
    alpha0 = alpha if alpha0 is None else alpha0
    # Step 1: order-zero sweep
    G0, sep = set(), {}
    for x, y in combinations(nodes, 2):
        if test(x, y, [], alpha0):
            sep[frozenset({x, y})] = frozenset()
        else:
            G0.add(frozenset({x, y}))
    # Step 2: decomposition
    R1, packets = min_vertex_cut(nodes, G0, max_size=max_cut)
    info = {"decomposed": R1 is not None,
            "R1": R1, "packets": packets, "G0_edges": len(G0)}
    if R1 is None:
        edges, sep = pc_stable(nodes, test, alpha, init_edges=G0,
                               start_l=1, sepsets=sep)
        return edges, sep, info
    # Steps 3-5: cross-packet pairs are never revisited
    group = {v: 0 for v in R1}
    for i, c in enumerate(packets, 1):
        for v in c:
            group[v] = i
    forbidden = {frozenset({x, y})
                 for i, j in combinations(range(len(packets)), 2)
                 for x in packets[i] for y in packets[j]}
    edges, sep = pc_stable(nodes, test, alpha, init_edges=G0, start_l=1,
                           forbidden=forbidden, sepsets=sep)
    return edges, sep, info


# =====================================================================
#  Metrics
# =====================================================================



def inferred_vstructures(edges, sepsets, nodes):
    """Unshielded colliders implied by the learned skeleton and separating sets.

    Returned triples are (a,b,c) with a<c lexicographically and a->b<-c the
    inferred collider.  This object remains well-defined even when finite-sample
    CI errors make the collection of collider decisions jointly inconsistent,
    which is why the paper uses it for finite-sample orientation evaluation
    instead of assuming the oriented output is necessarily a valid CPDAG.
    """
    skel = {frozenset(e) for e in edges}
    out = set()
    for b in nodes:
        nb = sorted(v for v in nodes if frozenset({v, b}) in skel)
        for a, c in combinations(nb, 2):
            if frozenset({a, c}) in skel:
                continue
            S = sepsets.get(frozenset({a, c}))
            if S is not None and b not in S:
                out.add((a, b, c))
    return out


def true_vstructures(nodes, edges):
    """Unshielded colliders in the generating DAG."""
    skel = skeleton_of(edges)
    parents = {v: set() for v in nodes}
    for u, v in edges:
        parents[v].add(u)
    out = set()
    for b in nodes:
        for a, c in combinations(sorted(parents[b]), 2):
            if frozenset({a, c}) not in skel:
                out.add((a, b, c))
    return out


def vstructure_scores(est, truth):
    tp = len(est & truth)
    fp = len(est - truth)
    fn = len(truth - est)
    prec = tp / (tp + fp) if tp + fp else (1.0 if not truth else 0.0)
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(vtp=tp, vfp=fp, vfn=fn, vprec=prec, vrec=rec, vf1=f1,
                vshd=fp + fn)

def metrics(est_edges, est_sep, nodes, true_edges, true_cp=None):
    """Finite-sample accuracy of a learned skeleton and its collider decisions.

    ``shd`` is retained as a backward-compatible alias for *skeleton* SHD
    (false positives + false negatives).  The revised paper does not call the
    orientation obtained from fallible CI decisions a CPDAG: inconsistent
    collider decisions can make a finite-sample PC orientation non-extendable.
    Orientation accuracy is therefore reported through unshielded-v-structure
    precision/recall/F1 and symmetric-difference count (``vshd``), all of which
    remain well-defined without assuming a valid equivalence-class graph.
    """
    truth = skeleton_of(true_edges)
    tp = len(est_edges & truth)
    fp = len(est_edges - truth)
    fn = len(truth - est_edges)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    skel_shd = fp + fn

    vest = inferred_vstructures(est_edges, est_sep, nodes)
    vtrue = true_vstructures(nodes, true_edges)
    vs = vstructure_scores(vest, vtrue)

    out = dict(tp=tp, fp=fp, fn=fn, prec=prec, rec=rec, f1=f1,
               skel_shd=skel_shd, shd=skel_shd)
    out.update(vs)
    return out

def oracle_sepsets(nodes, edges):
    """Exact sepsets from d-separation, for SHD against the true CPDAG."""
    parents = {v: set() for v in nodes}
    for u, w in edges:
        parents[w].add(u)

    def anc(ns):
        seen, stack = set(ns), list(ns)
        while stack:
            n = stack.pop()
            for p in parents[n]:
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def dsep(x, y, Z):
        Z = set(Z)
        an = anc({x, y} | Z)
        adjm = {v: set() for v in an}
        for v in an:
            ps = [p for p in parents[v] if p in an]
            for p in ps:
                adjm[v].add(p)
                adjm[p].add(v)
            for a, b in combinations(ps, 2):
                adjm[a].add(b)
                adjm[b].add(a)
        seen, stack = {x}, [x]
        while stack:
            n = stack.pop()
            if n == y:
                return False
            for w in adjm[n]:
                if w not in seen and w not in Z:
                    seen.add(w)
                    stack.append(w)
        return True

    skel = skeleton_of(edges)
    sep = {}
    for x, y in combinations(nodes, 2):
        if frozenset({x, y}) in skel:
            continue
        found = None
        pool = sorted((parents[x] | parents[y]) - {x, y})
        for k in range(len(pool) + 1):
            for S in combinations(pool, k):
                if dsep(x, y, set(S)):
                    found = frozenset(S)
                    break
            if found is not None:
                break
        sep[frozenset({x, y})] = found if found is not None else frozenset()
    return sep


def balanced_cut(vertices, edgeset, budget=None, beta=0.5, min_packet=2):
    """Choose a separator that yields *balanced* packets, not merely a small one.

    A minimum vertex cut answers the wrong question: it returns the cheapest way
    to disconnect the graph, which is typically to peel off a single low-degree
    vertex, leaving one giant packet and no useful parallelism. What the
    decomposition needs is a separator whose removal leaves packets of
    comparable size.

    Greedy heuristic: repeatedly delete the highest-degree remaining vertex
    (hubs are what hold the packets together) until the largest component holds
    at most beta * |V| vertices. Then prune the separator, discarding any vertex
    whose reinstatement does not merge two packets.

    Returns (R1, packets) or (None, None) if no admissible cut is found.
    """
    V = set(vertices)
    n = len(V)
    if budget is None:
        budget = max(3, int(0.25 * n))
    removed = []
    while len(removed) <= budget:
        rest = V - set(removed)
        all_comps = components(rest, edgeset)
        eligible = [c for c in all_comps if len(c) >= min_packet]
        # Acceptance requires at least two non-trivial packets, but Definition 2
        # makes *every* connected component a packet, singletons included.
        # Hence balance is checked on all components and all components are
        # returned.  The previous implementation filtered singletons out of the
        # return value, which could leave vertices assigned to neither R1 nor a
        # packet on graphs where singleton components occur.
        if (len(eligible) >= 2
                and max((len(c) for c in all_comps), default=0) <= beta * n):
            # prune: drop separator vertices that are not needed while
            # preserving the same admissibility criterion.
            pruned = list(removed)
            for v in list(pruned):
                trial = [w for w in pruned if w != v]
                cs_all = components(V - set(trial), edgeset)
                cs_eligible = [c for c in cs_all if len(c) >= min_packet]
                if (len(cs_eligible) >= 2
                        and max((len(c) for c in cs_all), default=0) <= beta * n):
                    pruned = trial
            return sorted(pruned), components(V - set(pruned), edgeset)
        # Iterate in sorted order: `rest` is a set, and iterating it directly
        # made the degree tie-break depend on Python's per-process string hash
        # seed, so the chosen separator -- and every number downstream of it --
        # varied between runs of the same script. Sorting fixes the tie-break to
        # the lexicographically first vertex of maximum degree.
        deg = {v: sum(1 for e in edgeset if v in e and not (e - {v}) <= set(removed))
               for v in sorted(rest)}
        if not deg or max(deg.values()) == 0:
            break
        removed.append(max(sorted(deg), key=lambda v: deg[v]))
    return None, None
