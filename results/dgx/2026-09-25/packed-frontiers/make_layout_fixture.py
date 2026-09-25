"""Encrypted regression inputs with live shared windows and nonzero neighbors."""
import hashlib, json
from pathlib import Path
import numpy as np
from fhemamba.packed_program import PackedProgram
from fhemamba.ops import ChebPoly

folder = Path(__file__).resolve().parent / 'layout-fixture'
folder.mkdir(exist_ok=True)
poly = ChebPoly((.1,.4,-.07,.01,.002,-.001,.0002,.0001),-1,1)
p = PackedProgram({(0,'gate'):poly}, slots=32768, bound=64, extended=True)
v = np.linspace(-8,8,256);v[32:48]=np.linspace(-.5,.5,16);v[80:96]=np.linspace(.4,-.4,16)
x=p.input(v)
a=x[32:48];b=x[80:96]
outputs=[a,b,a+b,a*b,a*a,-1*a,a+.1,a*np.linspace(.8,1,16)]
outputs += [a[3:11],a[np.array([1,4,7,10,13])],p.sum_last(a.reshape(4,4))]
outputs += [p.linear(a,np.eye(16)*.5),p._broadcast(a.reshape(16,1),(16,3))]
outputs += [p.concatenate([a,b]),p.nonlinear(a,'unused',(0,'gate')),p.nonlinear(b,'unused',(0,'gate'))]
# Deliberately reach the refresh frontier before creating several live windows.
deep=p.input(np.linspace(-.4,.4,64))
for _ in range(38):deep=deep*.999
for offset in [0,8,24,48]:
    view=deep[offset:offset+8]
    outputs += [view,p.nonlinear(view,'unused',(0,'gate'))]
for output in outputs:p.output(output,output.value)
p.write(folder/'program.txt')
(folder/'manifest.json').write_text(json.dumps({'schema':'packed-layout-regression-v1',**p.summary(),
 'files_sha256':{'program.txt':hashlib.sha256((folder/'program.txt').read_bytes()).hexdigest()},
 'coverage':['shared live alias','nested windows','offset mismatch','dirty multiplication','negation','public terms','linear','repeat','scatter','sum','noncontiguous gather','polynomial batch','refresh','pinned output']},indent=2)+'\n')
