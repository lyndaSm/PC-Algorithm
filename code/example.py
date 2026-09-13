"""
Worked example for the v2 paper: a six-vertex modular DAG.

    A1 -> A2 -> S <- B2 <- B1,      S -> D

Two packets {A1,A2} and {B1,B2} joined only through the collider S and its
child D. Everything below is computed with an exact d-separation oracle; no
statement in the paper's example section is asserted by hand.
"""
from itertools import combinations

NODES = ["A1", "A2", "B1", "B2", "S", "D"]
EDGES = [("A1", "A2"), ("A2", "S"), ("B1", "B2"), ("B2", "S"), ("S", "D")]

parents = {v: set() for v in NODES}
for u, w in EDGES:
    parents[w].add(u)


def ancestors(ns):
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
    an = ancestors({x, y} | Z)
    adj = {v: set() for v in an}
    for v in an:
        ps = [p for p in parents[v] if p in an]
        for p in ps:
            adj[v].add(p)
            adj[p].add(v)
        for a, b in combinations(ps, 2):
            adj[a].add(b)
            adj[b].add(a)
    seen, stack = {x}, [x]
    while stack:
        n = stack.pop()
        if n == y:
            return False
        for w in adj[n]:
            if w not in seen and w not in Z:
                seen.add(w)
                stack.append(w)
    return True


def subsets(s, k):
    return [set(c) for c in combinations(sorted(s), k)]


def components(vs, es):
    vs = set(vs)
    out, seen = [], set()
    for v in sorted(vs):
        if v in seen:
            continue
        comp, st = set(), [v]
        while st:
            u = st.pop()
            if u in comp:
                continue
            comp.add(u)
            for w in vs:
                if w not in comp and frozenset({u, w}) in es:
                    st.append(w)
        seen |= comp
        out.append(sorted(comp))
    return out


truth = {frozenset(e) for e in EDGES}

print("=" * 70)
print("STEP 1  order-zero sweep")
print("=" * 70)
G0, indep0 = set(), []
for x, y in combinations(NODES, 2):
    if dsep(x, y, set()):
        indep0.append((x, y))
    else:
        G0.add(frozenset({x, y}))
print("marginal independences:",
      ", ".join(f"{a}_||_{b}" for a, b in indep0))
print(f"|E(G0)| = {len(G0)} of {len(NODES)*(len(NODES)-1)//2} pairs; "
      f"true skeleton has {len(truth)} edges")
print("G0 contains the true skeleton:", truth <= G0)

print()
print("=" * 70)
print("STEP 2  decomposition")
print("=" * 70)
best = None
for size in range(1, len(NODES)):
    for cand in combinations(sorted(NODES), size):
        comps = [c for c in components(set(NODES) - set(cand), G0) if len(c) >= 2]
        if len(comps) >= 2:
            best = (sorted(cand), comps)
            break
    if best:
        break
R1, packets = best
print(f"minimum separator: R1 = {{{','.join(R1)}}}")
for i, c in enumerate(packets, 1):
    print(f"  packet R0^{i} = {{{','.join(c)}}}")
cross = [(x, y) for i, j in combinations(range(len(packets)), 2)
         for x in packets[i] for y in packets[j]]
print(f"cross-packet pairs settled at order 0: {len(cross)} "
      f"({', '.join(f'{a}-{b}' for a, b in cross)})")
print("  none of them is a true edge:",
      not any(frozenset({a, b}) in truth for a, b in cross))

print()
print("Corollary (adjacency confinement) check:")
for i, c in enumerate(packets, 1):
    for v in c:
        adj_v = sorted(w for w in NODES if frozenset({v, w}) in truth)
        ok = set(adj_v) <= set(c) | set(R1)
        print(f"  adj({v}) = {{{','.join(adj_v)}}}  confined: {ok}")

print()
print("=" * 70)
print("STEP 3-4  skeleton search  (packet-local, then interface)")
print("=" * 70)


def search(scope, init, pairs, label):
    """PC-stable restricted to the given pairs; returns surviving edges."""
    edges = set(init)
    sep = {}
    l = 1
    while True:
        adj = {v: {w for w in scope if frozenset({v, w}) in edges} for v in scope}
        cont = False
        for e in sorted(edges, key=lambda s: sorted(s)):
            if e not in edges or e not in pairs:
                continue
            x, y = sorted(e)
            done = False
            for a, b in ((x, y), (y, x)):
                pool = adj[a] - {b}
                if len(pool) >= l:
                    cont = True
                for S in subsets(pool, l):
                    if dsep(a, b, S):
                        edges.discard(e)
                        sep[e] = frozenset(S)
                        print(f"  [{label}] l={l}: {x} _||_ {y} | "
                              f"{{{','.join(sorted(S))}}}  -> remove")
                        done = True
                        break
                if done:
                    break
        if not cont:
            break
        l += 1
    return edges, sep


allsep = {frozenset({a, b}): frozenset() for a, b in indep0}
kept = set()
for i, c in enumerate(packets, 1):
    scope = sorted(set(c) | set(R1))
    init = {e for e in G0 if all(v in scope for v in e)}
    pairs = {frozenset(p) for p in combinations(sorted(c), 2)}
    ee, ss = search(scope, init, pairs, f"packet {i}")
    allsep.update(ss)
    kept |= {e for e in ee if e in pairs}

iface_pairs = {e for e in G0 if any(v in R1 for v in e)}
init = {e for e in G0 if not (all(v not in R1 for v in e)
                              and e not in kept)}
ee, ss = search(NODES, init, iface_pairs, "interface")
allsep.update(ss)
final = {e for e in ee if e in iface_pairs} | kept

print()
print("final skeleton:", sorted("-".join(sorted(e)) for e in final))
print("true skeleton :", sorted("-".join(sorted(e)) for e in truth))
print("CORRECT:", final == truth)

print()
print("=" * 70)
print("STEP 5  orientation")
print("=" * 70)
for b in NODES:
    nb = sorted(v for v in NODES if frozenset({v, b}) in final)
    for a, c in combinations(nb, 2):
        if frozenset({a, c}) in final:
            continue
        S = allsep.get(frozenset({a, c}))
        if S is not None and b not in S:
            print(f"  v-structure {a} -> {b} <- {c}   "
                  f"sepset({a},{c}) = {{{','.join(sorted(S))}}}")
true_v = [(a, c, b) for c in NODES for a, b in combinations(sorted(parents[c]), 2)
          if frozenset({a, b}) not in truth]
print("true v-structures:", [f"{a}->{c}<-{b}" for a, c, b in true_v])
