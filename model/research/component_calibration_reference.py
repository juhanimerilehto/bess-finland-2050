"""Reference calculation behind the component cost model (cost.py, method C).

Computes cumulative doublings of global lithium-ion and stationary production from
annual demand anchors, and the 2025 split of the European 4 h turnkey price into
cells, freight, DC balance of system, power conversion and the European premium.
Standalone; run with `python model/research/component_calibration_reference.py`.
"""
import math
def path(anchors):
    ys=sorted(anchors); out={}
    for a,b in zip(ys,ys[1:]):
        for y in range(a,b+1):
            out[y]=anchors[a]+(anchors[b]-anchors[a])*(y-a)/(b-a)
    return out
def cum(c0,an):
    p=path(an); c=c0; res={2025:c0}
    for y in range(2026,2051):
        c+=p[y]; res[y]=c
    return {y:math.log2(res[y]/c0) for y in res}
LI={'low':cum(4.7,{2025:1.59,2030:3.3,2035:4.6,2050:6.0}),
    'cen':cum(4.7,{2025:1.59,2030:4.0,2035:6.0,2050:9.0}),
    'high':cum(5.4,{2025:2.28,2030:6.0,2035:10.5,2050:15.0})}
ST={'low':cum(0.72,{2025:0.315,2030:0.7,2035:0.8,2050:1.0}),
    'cen':cum(0.72,{2025:0.315,2030:1.0,2035:1.3,2050:2.0}),
    'high':cum(0.72,{2025:0.315,2030:1.5,2035:2.0,2050:3.5})}
Y=[2025,2030,2035,2040,2050]
for k in LI: print('Li dbl',k,[round(LI[k][y],2) for y in Y])
for k in ST: print('St dbl',k,[round(ST[k][y],2) for y in Y])
# 2025 Europe 4h turnkey anchor (model): 128.5 + 75/4
T0=128.5+75/4
cell0=36.0; frt0=7.0; dcb0=16.0; pcs0_kw=75.0; prem0=T0-cell0-frt0-dcb0-pcs0_kw/4
print('T0',T0,'premium0',round(prem0,1))
# cases: name: (cell LR, Li-dbl scenario, cell floor, dcbos LR, pcs LR, st-dbl scen, freight path, premium rate to 2035, after, epc rate/yr, grid EUR/kW, 2030 cell normalisation)
cases={
 'Low-cost':dict(lrc=0.21,li='high',floor=12,lrb=0.15,lrp=0.15,st='high',frt=5,pr1=-0.10,pr2=-0.03,epc=-0.02,grid=100,norm=1.0),
 'Central': dict(lrc=0.18,li='cen', floor=18,lrb=0.12,lrp=0.12,st='cen', frt=7,pr1=-0.05,pr2=-0.02,epc=-0.01,grid=150,norm=1.0),
 'High-cost':dict(lrc=0.15,li='low', floor=28,lrb=0.08,lrp=0.08,st='low', frt=9,pr1=-0.02,pr2=-0.01,epc=0.0,grid=250,norm=1.2),
}
for name,c in cases.items():
    print('\n==',name)
    for y in Y:
        dl=LI[c['li']][y]; ds=ST[c['st']][y]
        cell=max(cell0*(1-c['lrc'])**dl, c['floor'] if y>2025 else 0)
        if y==2030: cell=max(cell, cell*c['norm'])
        if y==2030 and c['norm']>1: cell=cell0*(1-c['lrc'])**dl*c['norm']
        dcb=dcb0*(1-c['lrb'])**ds
        pcskw=pcs0_kw*(1-c['lrp'])**ds
        frt=frt0 if y==2025 else c['frt']
        t=y-2025
        prem=prem0*((1+c['pr1'])**min(t,10))*((1+c['pr2'])**max(t-10,0))
        turn=cell+frt+dcb+pcskw/4+prem
        epc=0.15*T0*(1+c['epc'])**t
        g=150 if y==2025 else c['grid']
        allin4=turn+epc+g/4
        # 2h all-in: energy items x1, pcs/2, prem scale same per kWh (approx), grid/2
        allin2=cell+frt+dcb+pcskw/2+prem+epc+g/2
        print(y, f"cell {cell:5.1f} frt {frt:4.1f} dcbos {dcb:5.1f} pcs {pcskw:5.1f}/kW premium {prem:5.1f} | turnkey4h {turn:6.1f} idx {turn/T0:.2f} | epc {epc:5.1f} grid {g}/kW | allin4h {allin4:6.1f} allin2h {allin2:6.1f}")
# user model
print('\nUser model 4h allin:')
for y,I in [(2025,1),(2035,0.607),(2050,0.46)]:
    print(y, round(((128.5*4+75)*I*1.15+150)/4,1), '2h', round(((128.5*2+75)*I*1.15+150)/2,1))
print('NREL chain', 0.607*178/234, 0.607*178/243)

print('\nEurope/China ratios')
for name,c in cases.items():
    out=[]
    for y in Y:
        dl=LI[c['li']][y]; ds=ST[c['st']][y]; t=y-2025
        cell=max(cell0*(1-c['lrc'])**dl, c['floor'] if y>2025 else 0)
        if y==2030 and c['norm']>1: cell=cell0*(1-c['lrc'])**dl*c['norm']
        dcb=dcb0*(1-c['lrb'])**ds; pcskw=pcs0_kw*(1-c['lrp'])**ds
        frt=frt0 if y==2025 else c['frt']
        prem=prem0*((1+c['pr1'])**min(t,10))*((1+c['pr2'])**max(t-10,0))
        eu=cell+frt+dcb+pcskw/4+prem
        cn=cell+0.9*dcb+0.4*pcskw/4+3
        out.append(f"{y}: EU {eu:.0f} CN {cn:.0f} ratio {eu/cn:.2f} gap {eu-cn:.0f}")
    print(name, ' | '.join(out))
