import numpy as np
import math
import random
import matplotlib.pyplot as plt



class Simul:
    """
    This is the prototype of the simulation code
    It moves the particles with at _velocity, using a vector notation: numpy should be used.
    """
    def __init__(self, simul_time, sigma, L):
        np.seterr(all='ignore')  # remove errors in where statements
        self.position = np.random.uniform(sigma, L-sigma,size=(150,2))
        self._velocity = np.random.normal(size=self.position.shape)  # random velocities
        self._i, self._j = np.triu_indices(self.position.shape[0], k=1)  # all pairs of indices between particles
        self.sigma = sigma  # particle radius
        self._simul_time = simul_time
        self.L = L 
        self.E_cin = len(self._velocity[:, 0]) * [0]
        self.Tf = 0.1
        self.Tc = 10
        self.graph_E = [0]


    def _wall_time(self):
        t=np.where(self._velocity>0, (self.L-self.sigma-self.position)/self._velocity, (self.sigma-self.position)/self._velocity)
        first_collision_time = np.min(t)
        particle, direction = np.unravel_index(t.argmin(),t.shape)
        print(particle,direction)
        return first_collision_time, particle, direction
        # calculate time of first collision, particle involved and direction


    def _pair_time(self):
        vij = self._velocity[self._i]-self._velocity[self._j]
        a = (vij**2).sum(1)
        rij = self.position[self._i]- self.position[self._j]
        b = (2*rij*vij).sum(1)
        c = (rij**2).sum(1)-4*self.sigma**2
        delta = b*b-4*a*c
        tco = np.where((delta>0)&(c>0)&(b<0), (-b-np.sqrt(delta))/(2*a), np.Inf)
        tmin = np.min(tco)
        ind = tco.argmin()
        return tmin,ind

   
    def md_step(self):
        print('Simul::md_step')
        ke_start = (self._velocity**2).sum()   # starting kinetic energy
        
        pressure = -1
        current_time = 0
        condition_on_time_variables = False
        
    
        w_time, particle, direction = self._wall_time()
        tmin, ind = self._pair_time()
        t = min(tmin,w_time)

        previous_position = np.copy(self.position)
        while current_time + t < self._simul_time:
            
            self.position += t*self._velocity
            current_time += t
            r = random.random()
            
            if w_time<tmin:
                if direction == 1:
                    self._velocity[particle,direction] = -self._velocity[particle,direction]
                elif self._velocity[particle,direction] < 0:
                    self._velocity[particle,direction] = np.sqrt(-2*np.log(r)*self.Tc)
                else:
                    self._velocity[particle,direction] = -np.sqrt(-2*np.log(r)*self.Tf)
            else :
                ind1=self._i[ind]
                ind2=self._j[ind]
                r2=self.position[ind1]-self.position[ind2]
                r=r2/np.linalg.norm(r2)
                dv=self._velocity[ind1]-self._velocity[ind2]
                v1=self._velocity[ind1]-r*np.dot(r,dv)
                v2=self._velocity[ind2]+r*np.dot(r,dv)
                self._velocity[ind1]=v1
                self._velocity[ind2]=v2
                tmin, ind = self._pair_time()

            

              
            w_time, particle, direction = self._wall_time()
            tmin, ind = self._pair_time()
            t = min(tmin,w_time)


            

        
        
        self.position += (self._simul_time-current_time) * self._velocity
        #assert  math.isclose (ke_start,  (self._velocity**2).sum() ) # check that we conserve energy after all the collisions

        
        hot_to_cold = np.where((previous_position[:, 0] < 30) & (self.position[:, 0] > 30), 1, 0)
        cold_to_hot = np.where((previous_position[:, 0] > 30) & (self.position[:, 0] < 30), 1, 0)
        self.E_cin += hot_to_cold*0.5*self._velocity[:, 0]**2 - cold_to_hot*0.5*self._velocity[:, 0]**2
        self.graph_E += [sum(self.E_cin)]
        
        print("E_cin_massique = ", sum(self.E_cin))
        
        #plt.plot(self.graph_E)
        return pressure



    def __str__(self):   # this is used to print the position and velocity of the particles
        p = np.array2string(self.position)
        v = np.array2string(self._velocity)
        return 'pos= '+p+'\n'+'vel= '+v+'\n'