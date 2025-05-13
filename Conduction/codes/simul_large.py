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
        self.Tf = 0.1
        self.Tc = 10
        self.sigma2 =10
        self.position2 = np.random.uniform(self.sigma2, L-self.sigma2,size=(1,2))
        self._velocity2 = np.random.normal(size=self.position2.shape)  # random velocities
        self.mass1=1
        self.mass2=50
        

    def _wall_time(self):
        t=np.where(self._velocity>0, (self.L-self.sigma-self.position)/self._velocity, (self.sigma-self.position)/self._velocity)
        first_collision_time = np.min(t)
        particle, direction = np.unravel_index(t.argmin(),t.shape)
        return first_collision_time, particle, direction
        # calculate time of first collision, particle involved and direction
        
    def _wall_time2(self):
        t=np.where(self._velocity2>0, (self.L-self.sigma2-self.position2)/self._velocity2, (self.sigma2-self.position2)/self._velocity2)
        first_collision_time = np.min(t)
        particle, direction = np.unravel_index(t.argmin(),t.shape)
        return first_collision_time, direction


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
    
    def _pair_time2(self): 
        rij=self.position-self.position2
        vij=self._velocity-self._velocity2
        a = (vij**2).sum(1)
        b = (2*rij*vij).sum(1)
        c = (rij**2).sum(1)-(self.sigma+self.sigma2)**2
        delta = b*b-4*a*c
        tco = np.where((delta>0)&(c>0)&(b<0), (-b-np.sqrt(delta))/(2*a), np.Inf)
        
        tmin = np.min(tco)
        p = tco.argmin()
        
        return tmin,p
        

   
    def md_step (self): 
        print('Simul: : md _step') 
        #ke_start = (self.mass1*self._velocity**2).sum() + (self.mass2*self._velocity2**2).sum() # Energie cinétqiue au début 
        current_time = 0 

        #on définit les temps de collisions entre particules et avec les murs 
        w_time, particle, direction = self._wall_time() 
        p_time, ind = self._pair_time() 
        
        w_time2, direction2 = self._wall_time2() 
        p_time2, p = self._pair_time2() 

        time = min(w_time, p_time, w_time2,p_time2) #le temps de collision pris en compte sera donc le plus petit des toutes 

        while current_time + time < self._simul_time: 
            
            self.position = self.position + time * self._velocity # mise à jour des positions au fur et à mesure des évènements physiques 
            self.position2 = self.position2 + time * self._velocity2 
            current_time = current_time + time # mise à jour du curseur de temps 
            r = random.random()

            if w_time == time: #Choc entre un mur et une petite particule
                if direction == 1:
                    self._velocity[particle,direction] = -self._velocity[particle,direction]
                elif self._velocity[particle,direction] < 0:
                    self._velocity[particle,direction] = np.sqrt(-2*np.log(r)*self.Tc)
                else:
                    self._velocity[particle,direction] = -np.sqrt(-2*np.log(r)*self.Tf)


            elif w_time2 == time : #Choc entre un mur et la grosse particule 
                if direction2 == 1:
                    self._velocity2 = -self._velocity2
                elif np.all(self._velocity2 < 0):
                    self._velocity2 = np.sqrt(-2*np.log(r)*self.Tc)
                else:
                    self._velocity2 = -np.sqrt(-2*np.log(r)*self.Tf)

            elif p_time2 == time : #Choc entre une petite particule et la grosse particule 
                dr = self.position[p] - self.position2 
                module = np.sqrt((dr**2).sum()) 
                dr /=module 
                tmp = (dr*(self._velocity[p] - self._velocity2)).sum() 
                self._velocity[p] = self._velocity[p] - dr * tmp*(2*self.mass2/ (self.mass1 + self.mass2)) #Mise à jour des vitesses après co 
                self._velocity2 = self._velocity2 + dr * tmp*(2*self.mass1/ (self.mass1 + self.mass2))

            else : #Choc entre deux petites particules 
                ind1=self._i[ind]
                ind2=self._j[ind]
                r2=self.position[ind1]-self.position[ind2]
                r=r2/np.linalg.norm(r2)
                dv=self._velocity[ind1]-self._velocity[ind2]
                v1=self._velocity[ind1]-r*np.dot(r,dv)
                v2=self._velocity[ind2]+r*np.dot(r,dv)
                self._velocity[ind1]=v1
                self._velocity[ind2]=v2
           
            w_time, particle, direction = self._wall_time() 
            p_time, ind = self._pair_time() 

            w_time2, direction2 = self._wall_time2() 
            p_time2, p = self._pair_time2() 

            time = min(w_time, p_time, w_time2,p_time2)
       
            
        self.position += (self._simul_time-current_time) * self._velocity
        self.position2 += (self._simul_time-current_time) * self._velocity2


    def __str__(self):   # this is used to print the position and velocity of the particles
        p = np.array2string(self.position)
        v = np.array2string(self._velocity)
        return 'pos= '+p+'\n'+'vel= '+v+'\n'