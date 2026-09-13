"""Targeted robustness checks added after the full technical audit.

Experiments:
  fwer      Bonferroni vs Holm vs BH vs uncorrected at order zero.
  beta      Sensitivity of the balanced-cut heuristic to beta.
  workers   Fixed processor budgets for ideal packet scheduling (CI-test work).

All experiments are paired on deterministic replicate seeds.
"""
from itertools import combinations
import sys
import numpy as np

from simulate import FisherZ, balanced_cut, components, gaussian_sample, skeleton_of
from review_experiments import g0_from_p, pvalues
from v2_experiments import hub_dag, module_of
from bounded_sep import bridge_dag, stage_work

ALPHA=0.01
N=1000
REPS=25


def fwer():
    print("ORDER-ZERO MULTIPLICITY: UNCORRECTED / BH / BONFERRONI / HOLM")
    print(f"Hub family, N={N}, alpha={ALPHA}, {REPS} paired replicates; true separator used.\n")
    print(f"{'k':>3} {'n':>4} {'method':>12} | {'|G0|':>6} {'false xmod':>10} {'packets':>7} {'maxpkt':>7} {'exact':>6}")
    print('-'*78)
    for k in (6,8,10,12):
        vals={m:dict(sz=[],fe=[],npk=[],mx=[],ex=0) for m in ('uncorrected','bh','bonferroni','holm')}
        for rep in range(REPS):
            rng=np.random.default_rng(31+977*rep+k)
            nodes,edges,truesep=hub_dag(k,5,rng)
            X=gaussian_sample(nodes,edges,N,rng)
            p=pvalues(X,nodes)
            for meth,d in vals.items():
                G0=g0_from_p(p,meth)
                d['sz'].append(len(G0))
                d['fe'].append(sum(1 for e in G0
                                   if len({module_of(v) for v in e})==2
                                   and all(module_of(v)!='SEP' for v in e)))
                cs=components(set(nodes)-set(truesep),G0)
                d['npk'].append(len(cs)); d['mx'].append(max(map(len,cs),default=0))
                d['ex'] += (len(cs)==k and all(len(c)==5 for c in cs))
        for meth in ('uncorrected','bh','bonferroni','holm'):
            d=vals[meth]
            print(f"{k:>3} {len(nodes):>4} {meth:>12} | {np.mean(d['sz']):>6.0f} {np.mean(d['fe']):>10.1f} "
                  f"{np.mean(d['npk']):>7.1f} {np.mean(d['mx']):>7.1f} {100*d['ex']/REPS:>5.0f}%")

    print("\nSMALL-n POWER / SPURIOUS-SPLIT CHECK")
    print(f"{'N':>5} {'method':>12} | {'found':>6} {'spliterr':>8} {'lost':>6} {'|R1|':>5}")
    print('-'*52)
    for Ns in (200,1000):
        vals={m:dict(found=0,bad=0,lost=[],r1=[]) for m in ('uncorrected','bh','bonferroni','holm')}
        for rep in range(REPS):
            rng=np.random.default_rng(90000+211*rep+Ns)
            nodes,edges,_=hub_dag(3,4,rng)
            truth=skeleton_of(edges)
            X=gaussian_sample(nodes,edges,Ns,rng)
            p=pvalues(X,nodes)
            for meth,d in vals.items():
                G0=g0_from_p(p,meth)
                # This experiment studies a small-cut search, not the balanced heuristic.
                from simulate import min_vertex_cut
                R1,pk=min_vertex_cut(nodes,G0,max_size=3)
                if R1 is None:
                    d['lost'].append(0); continue
                d['found']+=1; d['r1'].append(len(R1))
                grp={v:0 for v in R1}
                for i,c in enumerate(pk,1):
                    for v in c: grp[v]=i
                bad=[e for e in truth if all(grp.get(v,0)>0 for v in e)
                     and len({grp[v] for v in e})==2]
                d['lost'].append(len(bad)); d['bad']+=bool(bad)
        for meth in ('uncorrected','bh','bonferroni','holm'):
            d=vals[meth]
            print(f"{Ns:>5} {meth:>12} | {100*d['found']/REPS:>5.0f}% {100*d['bad']/REPS:>7.0f}% "
                  f"{np.mean(d['lost']):>6.2f} {(np.mean(d['r1']) if d['r1'] else float('nan')):>5.1f}")


def beta():
    print("BALANCED-CUT SENSITIVITY TO beta")
    print(f"Bonferroni order zero; N={N}; {REPS} paired replicates.\n")
    print(f"{'family':>8} {'k':>3} {'n':>4} {'beta':>5} | {'found':>6} {'|R1|':>5} {'pkts':>5} {'maxpkt':>7} {'max/n':>6}")
    print('-'*72)
    specs=[('hub',8,lambda r:hub_dag(8,5,r)[:2]),
           ('hub',12,lambda r:hub_dag(12,5,r)[:2]),
           ('bridge',8,lambda r:bridge_dag(8,5,r)[:2]),
           ('bridge',10,lambda r:bridge_dag(10,5,r)[:2])]
    for name,k,gen in specs:
        cache=[]
        for rep in range(REPS):
            rng=np.random.default_rng(31+977*rep+k)
            nodes,edges=gen(rng)
            X=gaussian_sample(nodes,edges,N,rng)
            cache.append((nodes,g0_from_p(pvalues(X,nodes),'bonferroni')))
        for b in (0.2,0.3,0.4,0.5):
            r1=[]; npk=[]; mx=[]; found=0
            for nodes,G0 in cache:
                R,pk=balanced_cut(nodes,G0,beta=b)
                if R is None: continue
                found+=1;r1.append(len(R));npk.append(len(pk));mx.append(max(map(len,pk)))
            n=len(cache[0][0])
            print(f"{name:>8} {k:>3} {n:>4} {b:>5.1f} | {100*found/REPS:>5.0f}% "
                  f"{(np.mean(r1) if r1 else float('nan')):>5.1f} {(np.mean(npk) if npk else float('nan')):>5.1f} "
                  f"{(np.mean(mx) if mx else float('nan')):>7.1f} {(np.mean(mx)/n if mx else float('nan')):>6.2f}")


def lpt_makespan(jobs,P):
    """Deterministic longest-processing-time list scheduling."""
    loads=[0.0]*P
    for job in sorted(jobs,reverse=True):
        j=min(range(P),key=lambda q:(loads[q],q))
        loads[j]+=job
    return max(loads,default=0.0)


def workers():
    print("FIXED PROCESSOR BUDGETS (CI-TEST WORK; LPT PACKET SCHEDULING)")
    print(f"Bonferroni order zero; N={N}; {REPS} paired replicates; cut-search cost excluded.\n")
    print(f"{'fam':>6} {'k':>3} {'n':>4} | {'P=2':>6} {'P=4':>6} {'P=8':>6} {'P=inf':>6} | {'cuts':>7}")
    print('-'*63)
    specs=[]
    for k in (4,6,8,10,12): specs.append(('hub',k))
    for k in (4,6,8,10): specs.append(('bridge',k))
    for fam,k in specs:
        sp={2:[],4:[],8:[],999:[]};cuts=0
        for rep in range(REPS):
            rng=np.random.default_rng(31+977*rep+k)
            if fam=='hub': nodes,edges,_=hub_dag(k,5,rng)
            else: nodes,edges,_=bridge_dag(k,5,rng)
            X=gaussian_sample(nodes,edges,N,rng); n=len(nodes)
            G0=g0_from_p(pvalues(X,nodes),'bonferroni')
            R,pk=balanced_cut(nodes,G0,beta=.3)
            if R is None: continue
            cuts+=1
            pw,iface=stage_work(nodes,X,G0,R,pk)
            w0=n*(n-1)//2; serial=w0+sum(pw)+iface
            for P in (2,4,8):
                sp[P].append(serial/(w0+lpt_makespan(pw,P)+iface))
            sp[999].append(serial/(w0+max(pw)+iface))
        f=lambda P:np.mean(sp[P]) if sp[P] else float('nan')
        print(f"{fam:>6} {k:>3} {n:>4} | {f(2):>6.2f} {f(4):>6.2f} {f(8):>6.2f} {f(999):>6.2f} | {cuts:>3}/{REPS:<3}")


if __name__=='__main__':
    which=sys.argv[1] if len(sys.argv)>1 else 'fwer'
    {'fwer':fwer,'beta':beta,'workers':workers}[which]()
