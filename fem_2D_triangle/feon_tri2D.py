from feon.sa import *
if __name__ == "__main__":
    E = 210e6
    nu = 0.3
    t = 0.025

    n0 = Node(0, 0)
    n1 = Node(0.5,0)
    n2 = Node(0.5,0.25)
    n3 = Node(0,0.25)
    e0 = Tri2D11S((n0, n1, n2), E, nu, t)
    e1 = Tri2D11S((n0, n2, n3), E, nu, t)

    s = System()

    s.add_nodes(n0, n1, n2, n3)
    s.add_elements(e0, e1)
    s.add_node_force(1,Fx=9.375)
    s.add_node_force(2,Fx=9.375)
    s.add_fixed_sup(0,3)

    s.solve()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri    
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

nx = [nd.x for nd in s.get_nodes()]
ny = [nd.y for nd in s.get_nodes()]

nID = [[nd.ID for nd in el] for el in s.get_elements()]
tr = tri.Triangulation(nx, ny, nID)

ux = [nd.disp["Ux"] for nd in s.get_nodes()]
uy = [nd.disp["Ux"] for nd in s.get_nodes()]
ux = np.array(ux)
uy = np.array(uy)

sx = [el.stress["sx"][0][0] for el in s.get_elements()] 
sy = [el.stress["sy"][0][0] for el in s.get_elements()]
sxy = [el.stress["sxy"][0][0] for el in s.get_elements()]
sx = np.array(sx)
sy = np.array(sy)
sxy = np.array(sxy)

fig1,fig2 = plt.figure(),plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

ncb = ax1.tricontourf(tr, ux, color='k', cmap='jet')
fig1.colorbar(ncb)

patches = []
for el in s.get_elements():
    ex,ey = [],[]
    for nd in el.nodes:
        ex.append(nd.x)
        ey.append(nd.y)
        Polygon = Polygon(zip(ex,ey), closed=True)
        patches.append(Polygon)

pc = PatchCollection(patches, color='k',edgecolor='w', alpha=0.4)  
pc.set_array(np.array(sx))

ax2.add_collection(pc)
ax2.set_xlim([0, 0.5])
ax2.set_ylim([0, 0.25])
ax2.set_aspect('equal')
fig2.colorbar(pc)

plt.show()