import numpy as np
import gsd.hoomd


class GsdHandler():
    def __init__(self,filename):
        self.read_system(filename,-1)

    def read_system(self,input,target_frame):
        """
        Read in a snapshot from a gsd file or snapshot.
        """
        self.target_frame = target_frame
        self.frame = 0
        try:
            with gsd.hoomd.open(name=input, mode='r') as f:

                if(len(f)>1):
                    print("init_gsd can't have more than one frames")
                    # exit()

                if (target_frame==-1):
                    # frame = f.read_frame(len(f)-1)
                    frame = f[len(f)-1]
                    self.frame = len(f)-1
                else:
                    self.frame = target_frame
                    frame=f[int(target_frame)]
                    # frame = f.read_frame(target_frame)
                self.positions = (frame.particles.position).copy()
                self.velocities = (frame.particles.velocity).copy()
                self.bodi = (frame.particles.body).copy()
                self.moment_inertia = (frame.particles.moment_inertia).copy()
                self.orientations = (frame.particles.orientation).copy()
                self.mass = (frame.particles.mass).copy()
                self.angmom = (frame.particles.angmom).copy()

                self.types = (frame.particles.types).copy()
                self.typeid = (frame.particles.typeid).copy()

                self.charges = (frame.particles.charge).copy()

                self.Lx,self.Ly,self.Lz = frame.configuration.box[0:3]
                self.box = frame.configuration.box

        except:
            self.positions = (input.particles.position).copy()
            self.velocities = (input.particles.velocity).copy()
            self.types = (input.particles.types).copy()
            self.typeid = (input.particles.typeid).copy()
            self.bodi = (input.particles.body).copy()
            self.moment_inertia = (input.particles.moment_inertia).copy()
            self.orientations = (input.particles.orientation).copy()
            self.mass = (input.particles.mass).copy()
            self.angmom = (input.particles.angmom).copy()
            self.charges = (input.particles.charge).copy()

            self.Lx = input.box.Lx
            self.Ly = input.box.Lx
            self.Lz = input.box.Lx

        """
        snapshot.particles.body is broken, non body particles should have -1
        but the container I think only holds unsigned ints so -1 defaults to
        4294967295 which is very inconvenient for the rest oof the class
        so here I swithc it back to -1
        """
        self.body = np.array(self.bodi,dtype=int)
        self.body[self.body>9999999]= -1.0


    def get_central_pos(self):

        """
        This function returns only the central pos
        For cylinders you also need to return orientation pos

        below old:
        for the shape cylinder_v2 two positions are important
        0 for COM and 15 for the top bead (gives the direction of cylinder)

        this function returns those positions, if you use another shape you need
        another set of index to return
        """
        pos_COM = self.positions[self.typeid==0]
        # N_shape = len(pos_COM)
        # self.positions = self.positions.reshape(N_shape,-1,3)
        # important_pos1 = self.positions[:,0,:]
        # important_pos2 = self.positions[:,15,:]
        # important_pos = np.hstack((important_pos1,important_pos2))

        return pos_COM

    def get_COM_velocities(self):

        """
        If your input gsd has initialized/randomized velocities (for example if
        you get it from hoomd) carry them to this simulation
        """

        vel_COM = self.velocities[self.typeid==0]
        return vel_COM

    def getMass(self):
        masses = self.mass[self.typeid==0]
        return masses

    def getOrientations(self):
        or_COM = self.orientations[self.typeid==0]
        return or_COM

    def getAngmom(self):
        angmom_COM = self.angmom[self.typeid==0]
        return angmom_COM

    def getMoi(self):
        moi_COM = self.moment_inertia[self.typeid==0]
        return moi_COM

    def getCharges(self):
        charges = np.copy(self.charges)
        return charges

    def getAllbody(self):
        return self.body

    def getLx(self):
        return self.Lx

    def dummy(self):
        pass
