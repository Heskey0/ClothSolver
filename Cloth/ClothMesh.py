from mesh_data import meshCloth
import taichi as ti
import numpy as np
ti.init(arch=ti.cpu)

numParticles = len(meshCloth['vertices']) // 3
numSurfs = len(meshCloth['faceTriIds']) // 3
print("vertex:", numParticles)
print("triangle:", numSurfs)

pos_np = np.array(meshCloth['vertices'], dtype=float)
surf_np = np.array(meshCloth['faceTriIds'], dtype=int)

pos_np = pos_np.reshape((-1,3))
surf_np = surf_np.reshape((-1,3))

pos = ti.Vector.field(3, float, numParticles)
surf = ti.Vector.field(3, int, numSurfs)

pos.from_numpy(pos_np)
surf.from_numpy(surf_np)


# ---------------------------------------------------------------------------- #
#                                      Init                                    #
# ---------------------------------------------------------------------------- #

'''
init Topology
'''

TripleList = []
neighbourList = []      # neighbour triangle list
# idx0, idx1: vertex idx of the public edge
bendingList = []        # neighbour triangle vertex list
stretchingList = []     # edge list

numEdge = 0
numNeighbourSurf = 0


def iniTopology():
    global numEdge
    global numNeighbourSurf

    for i in range(numSurfs):
        id0 = surf_np[i][0]
        id1 = surf_np[i][1]
        id2 = surf_np[i][2]
        TripleList.append({'idx0':id0, 'idx1':id1, 'idx_face':i})
        TripleList.append({'idx0':id1, 'idx1':id2, 'idx_face':i})
        TripleList.append({'idx0':id0, 'idx1':id2, 'idx_face':i})
    TripleList.sort(key=lambda k: (k.get('idx0'), k.get('idx1')))

    for i, t in enumerate(TripleList):
        if t['idx0'] == TripleList[i-1]['idx0'] and t['idx1'] == TripleList[i-1]['idx1']:
            neighbourList.append({TripleList[i-1]['idx_face'], t['idx_face']})
            idx0 = t['idx0']
            idx1 = t['idx1']
            idx2 = 0
            idx3 = 0
            for j in range(3):
                if surf_np[TripleList[i-1]['idx_face']][j] != idx0 and surf_np[TripleList[i-1]['idx_face']][j] != idx1:
                    idx2 = surf_np[TripleList[i-1]['idx_face']][j]
                if surf_np[t['idx_face']][j] != idx0 and surf_np[t['idx_face']][j] != idx1:
                    idx3 = surf_np[TripleList[i-1]['idx_face']][j]
            bendingList.append([idx0,idx1,idx2,idx3])
            numNeighbourSurf += 1
        else:
            stretchingList.append([t['idx0'], t['idx1']])
            numEdge += 1

iniTopology()

'''
Init Mass
'''

invMass = ti.field(ti.f32, shape=(numParticles))

stretchingIds = ti.Vector.field(2, ti.i32, numEdge)
bendingIds = ti.Vector.field(4, ti.i32, numNeighbourSurf)

stretchingIds.from_numpy(np.array(stretchingList))
bendingIds.from_numpy(np.array(bendingList))

@ti.kernel
def iniMass():
    for i in range(numSurfs):
        # init invMass
        id0 = surf[i][0]
        id1 = surf[i][1]
        id2 = surf[i][2]
        e0 = pos[id1] - pos[id0]
        e1 = pos[id2] - pos[id0]
        A = 0.5 * e0.cross(e1).norm()
        pInvMass = 1.0 / A / 3.0
        invMass[id0] += pInvMass
        invMass[id1] += pInvMass
        invMass[id2] += pInvMass
iniMass()

'''
Init restLength
'''

stretchingLengths = ti.field(ti.f32, shape=(numEdge))
bendingLengths = ti.field(ti.f32, shape=(numNeighbourSurf))

@ti.kernel
def iniRestLength():
    for i in range(numEdge):
        id0 = stretchingIds[i][0]
        id1 = stretchingIds[i][1]
        stretchingLengths[i] = (pos[id0] - pos[id1]).norm()
    for i in range(numNeighbourSurf):
        id0 = bendingIds[i][2]
        id1 = bendingIds[i][3]
        bendingLengths[i] = (pos[id0] - pos[id1]).norm()

iniRestLength()
