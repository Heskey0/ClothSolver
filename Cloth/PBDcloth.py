from ClothMesh import *
import taichi.math as tm

surf_show = ti.field(int, numSurfs * 3)
surf_show.from_numpy(surf_np.flatten())

prevPos = ti.Vector.field(3, float, numParticles)
restPos = ti.Vector.field(3, float, numParticles)
vel = ti.Vector.field(3, float, numParticles)

numSubsteps = 10
frameDt = 1.0 / 30.0
dt = frameDt / numSubsteps

fixPoint0 = True
pause = False

'''
physic quantity
'''
stretchingCompliance = 1.0
# 0~20
bendingCompliance = 2.0
thickness = 0.01
radius_search = 2*thickness


# ---------------------------------------------------------------------------- #
#                                     Hash                                     #
# ---------------------------------------------------------------------------- #
querySize = ti.field(dtype=ti.i32, shape=())

@ti.data_oriented
class Hash:
    # spacing: 格子的大小
    def __init__(self, spacing, maxNumObjects):
        self.spacing = spacing
        self.maxNumObjects = maxNumObjects
        self.tableSize = 2*maxNumObjects
        self.cellStart = ti.field(dtype=ti.i32, shape=(self.tableSize+1))
        self.cellEntries = ti.field(dtype=ti.i32, shape=(maxNumObjects))
        self.firstAdjId = ti.field(dtype=ti.i32, shape=(maxNumObjects+1))
        self.adjIds = ti.field(dtype=ti.i32, shape=(20*maxNumObjects))
        # 搜索到的物体集合
        self.queryIds = ti.field(dtype=ti.i32, shape=(maxNumObjects))

    @ti.func
    def hashCoords(self, xi, yi, zi):
        h = (int(xi) * 92837111) ^ (int(yi) * 689287499) ^ (int(zi) * 283923481)
        return ti.abs(h) % self.tableSize

    @ti.func
    def intCoord(self, coord):
        return ti.floor(coord / self.spacing)

    @ti.func
    def hashPos(self, pos, idx):
        return self.hashCoords(self.intCoord(pos[idx].x),
                               self.intCoord(pos[idx].y),
                               self.intCoord(pos[idx].z))

    @ti.kernel
    def create(self, pos: ti.template(), length: ti.template()):
        numObjects = length
        self.cellStart.fill(0)
        self.cellEntries.fill(0.0)

        for i in range(numObjects):
            h = self.hashPos(pos, i)
            self.cellStart[h] += 1

        # determine cells starts

        start = 0
        for i in range(self.tableSize):
            start += self.cellStart[i]
            self.cellStart[i] = start

        self.cellStart[self.tableSize] = start # guard

        # fill in objects ids

        for i in range(numObjects):
            h = self.hashPos(pos, i)
            self.cellStart[h] -= 1
            self.cellEntries[self.cellStart[h]] = i

    @ti.func
    def query(self, pos, idx, maxDist):
        querySize[None] = 0
        x0 = self.intCoord(pos[idx].x - maxDist)
        y0 = self.intCoord(pos[idx].y - maxDist)
        z0 = self.intCoord(pos[idx].z - maxDist)

        x1 = self.intCoord(pos[idx].x + maxDist)
        y1 = self.intCoord(pos[idx].y + maxDist)
        z1 = self.intCoord(pos[idx].z + maxDist)

        for xi in range(x0, x1+1):
            for yi in range(y0, y1 + 1):
                for zi in range(z0, z1 + 1):
                    h = self.hashCoords(xi, yi, zi)
                    start = self.cellStart[h]
                    end = self.cellStart[h + 1]

                    for i in range(start, end+1):
                        self.queryIds[querySize[None]] = self.cellEntries[i]
                        querySize[None] = querySize[None] + 1

    @ti.kernel
    def queryAll(self, pos: ti.template(), maxDist: ti.template()):
        # pre_num = 10*self.maxNumObjects
        num = 0
        maxDist2 = maxDist*maxDist
        for i in range(self.maxNumObjects):
            id0 = i
            self.firstAdjId[id0] = num
            self.query(pos, id0, maxDist)
            for j in range(querySize[None]):
                id1 = self.queryIds[j]
                if id1 >= id0:
                    continue
                dist2 = (pos[id0]-pos[id1]).norm()
                dist2 = dist2*dist2
                if dist2 > maxDist2:
                    continue
                # if num >= pre_num:
                #     pre_num = 2*num
                #     newIds = ti.field(dtype=ti.i32, shape=(2*num))
                #     self.adjIds = newIds
                self.adjIds[num] = id1
                num += 1
        # self.firstAdjId[self.maxNumObjects] = num

hash = Hash(radius_search, numParticles)

# ---------------------------------------------------------------------------- #
#                                    Solve                                     #
# ---------------------------------------------------------------------------- #
@ti.kernel
def preSolve():
    g = ti.Vector([0.0,-1.0,0.0])
    maxV = 0.2 * thickness / dt
    for i in pos:
        if i == 0 and fixPoint0:
            continue
        prevPos[i] = pos[i]
        vel[i] += g*dt
        vel[i] = ti.min(vel[i], maxV)
        pos[i] += vel[i]*dt
        if pos[i].y < 0.0:
            pos[i] = prevPos[i]
            pos[i].y = 0.0
            vel[i].y = -vel[i].y * 0.8

@ti.kernel
def solveStretching():
    alpha = stretchingCompliance / dt/dt
    for i in range(numEdge):
        if i == 0 and fixPoint0:
            continue
        id0 = stretchingIds[i][0]
        id1 = stretchingIds[i][1]
        w0 = invMass[id0]
        w1 = invMass[id1]
        w = w0 + w1
        Len = (pos[id0] - pos[id1]).norm()
        if Len == 0.0 or w ==0.0:
            continue
        grads = (pos[id0] - pos[id1]) / Len
        restLen = stretchingLengths[i]
        C = Len - restLen

        s = -C / (w+alpha)
        pos[id0] += grads * s * w0
        pos[id1] += -grads * s * w1

@ti.kernel
def solveBending():
    alpha = bendingCompliance / dt / dt
    for i in range(numEdge):
        if i == 0 and fixPoint0:
            continue
        id0 = bendingIds[i][2]
        id1 = bendingIds[i][3]
        w0 = invMass[id0]
        w1 = invMass[id1]
        w = w0 + w1
        Len = (pos[id0] - pos[id1]).norm()
        if Len == 0.0 or w == 0.0:
            continue
        grads = (pos[id0] - pos[id1]) / Len
        restLen = bendingLengths[i]
        C = Len - restLen

        s = -C / (w + alpha)
        pos[id0] += grads * s * w0
        pos[id1] += -grads * s * w1

def solve():
    solveStretching()
    solveBending()

@ti.kernel
def postSolve():
    for i in pos:
        if i == 0 and fixPoint0:
            pos[0] = restPos[0]
            continue
        vel[i] = (pos[i] - prevPos[i]) / dt

def iniHash():
    maxVelocity = 0.2 * thickness / dt
    hash.create(pos, numParticles)
    maxTravelDist = maxVelocity * frameDt
    hash.queryAll(pos, maxTravelDist)

@ti.kernel
def solveCollision():
    thickness2 = thickness * thickness
    for i in range(numParticles):
        if invMass[i] == 0.0:
            continue
        id0 = i
        first = hash.firstAdjId[i]
        last = hash.firstAdjId[i+1]
        for j in range(first, last):
            id1 = hash.adjIds[j]
            if(invMass[id1] == 0.0):
                continue
            vecs = pos[id1] - pos[id0]
            dist = vecs.norm()
            dist2 = dist*dist
            restDist = (restPos[id0] - restPos[id1]).norm()
            restDist2 = restDist*restDist
            minDist = thickness
            if dist2 > thickness2 or dist2 > restDist2 or dist2 == 0.0:
                continue
            if restDist2 < thickness2:
                minDist = ti.sqrt(restDist2)
            # position correction
            pos[id0] += -0.5 * vecs/dist * (minDist-dist)
            pos[id1] += 0.5 * vecs/dist * (minDist-dist)

            # velocities
            vecs = pos[id0]-prevPos[id0]
            vecs1 = pos[id1]-prevPos[id1]
            # average velocity
            vecs2 = 0.5*(vecs + vecs1)
            # velocity corrections
            vecs = vecs2 - vecs
            vecs1 = vecs2 - vecs1
            # add corrections
            friction = 0.0
            pos[id0] += vecs*friction
            pos[id1] += vecs1*friction


def substep():
    iniHash()
    preSolve()
    solve()
    solveCollision()
    postSolve()

# ---------------------------------------------------------------------------- #
#                                      gui                                     #
# ---------------------------------------------------------------------------- #
# init the window, canvas, scene and camerea
window = ti.ui.Window("pbd cloth", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0.067, 0.184, 0.255))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# initial camera position
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)


@ti.kernel
def init_pos():
    for i in range(numParticles):
        pos[i] += tm.vec3(0.5, 1, 0)
        restPos[i] = pos[i]
    if fixPoint0:
        pos[0] += ti.Vector([0.0,0.0,0.5])
    else:
        vel[0] = ti.Vector([0.5, 1.0, 0.5])


def main():
    init_pos()
    while window.running:
        global pause
        # do the simulation in each step
        for e in window.get_events("Press"):
            if e.key == 'p':
                pause = not pause
        if not pause:
            for _ in range(numSubsteps):
                substep()


        # set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # set the light
        # scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))

        # draw
        # scene.particles(pos, radius=0.02, color=(0, 1, 1))
        scene.mesh(pos, indices=surf_show, color=(1, 1, 0))

        # show the frame
        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    main()