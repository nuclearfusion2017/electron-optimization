'''
True Workflow
 - Choose parameter to alter:
   - r
   - B_0
   - sep_dist

 - Choose start and stop ranges for each parameter
 - Initialize positions and velocities vectors from monte carlo style governor
 - Setup coil parameters for run set - No





structure
---------
inputs ->
  coil parameters
  electron gun parameters

outputs ->
  best coil distance for a given radius
  do coil radius and field strength affect each other?
  Idea is to input design restrictions and find best coil spacing
    --> see effects of increasing coil current (therefore B)
  --> total metric is minimizing average r^2 = x^2 + y^2 + (z-sep/2)^2
  -- bit of statistical analysis to determine how the curve is shifted (mean vs median)
  --> starting r vs avg r^2 distribution
  --> arange from 0 to r
  --> tweak

where better = lowest average r^2

Our goal is to produce a model and determine relationships.

for each coil arangement, test distance of l vs r, keeping D constant
--> hypothesis: larger L values will mean better runs
--> hypothesis: certain best range of coil sep values
--> hypothesis: field strength and coil radius will produce a ratio constant for all variations
  --> test by 2d array of coil radius x field strength
  --> heatmap graphic
--> insertion ratio

new_simulation


'''
# Imports
import sys
from math import sin, cos, tan, radians, sqrt, ceil
import pyopencl as cl
import numpy as np
import pyopencl.array as cl_array
from scipy.special import ellipk, ellipe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
import cPickle as pickle

# Constants
mu_0 = 1.25663706e-6
ellipe_table = ellipe(np.arange(0,1, 1.0/10000000.0))
ellipk_table = ellipk(np.arange(0,1,1.0/10000000.0))
e_charge = 1.6e-19  # Coulombs
e_mass = 9.1e-31    # Kilograms


'''
Takes array of coils and displays to screen. First and second coils are bounding
box coils.

Positions is list of positions
'''

class all:
    def __init__(self):
        print '-- New all object created --'

        # call GPU building
            # initialize GPU
            # load single particle simulation code
            # pass positions, velocities, coils
            # electron gun function returns positions and velocities


    class _GPU:
        def __init__(self, filename, device_id = 1):
            # Setup OpenCL platform
            platform = cl.get_platforms()
            computes = [platform[0].get_devices()[device_id]]
            print "New context created on", computes
            self.ctx = cl.Context(devices=computes)
            self.queue = cl.CommandQueue(self.ctx)
            self.mf = cl.mem_flags

            # Open and build cl code
            f = open(filename, 'r')
            fstr = "".join(f.readlines())
            self.program = cl.Program(self.ctx, fstr).build()

        def execute(self, sim, quiet=False):
            # 1 float is 4 bytes

            # Prepare input, output, and lookup val buffers
            self.p_buf = cl.Buffer(self.ctx,  self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=sim.positions )        # Positions
            self.v_buf = cl.Buffer(self.ctx,  self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=sim.velocities )       # Velocities
            self.coil_buf = cl.Buffer(self.ctx,  self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=sim.coils )         # Coils
            self.c_spheres_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = sim.c_spheres)# Charge spheres
            self.ee = cl.Buffer(self.ctx,  self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=sim.ee_table )            # Elliptical Integral 1
            self.ek = cl.Buffer(self.ctx,  self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=sim.ek_table )            # Elliptical Integral 2
            self.d_buf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, sim.bytesize * sim.num_particles * sim.num_steps)                 # Output r^2 buffer
            self.queue.finish()

            # Run Kernel
            kernelargs = (self.p_buf, self.v_buf, self.coil_buf, self.c_spheres_buf, self.ee, self.ek, self.d_buf, sim.sim_properties, sim.dt)

            if quiet!=True: print "Values successfully passed"

            self.program.compute_trajectory(self.queue, (int(sim.num_particles),), None, *(kernelargs))

            if quiet!=True: print "Kernels started"

            self.queue.finish()

            # Dump, clean, return -- must reshape data when using float4s
            self.ret_val = np.empty_like(np.ndarray((sim.num_particles, sim.num_steps, sim.bytesize/4)).astype(np.float32))
            read = cl.enqueue_copy(self.queue, self.ret_val, self.d_buf)
            self.queue.finish()
            read.wait()
#            print (read.profile.end-read.profile.start)
            self.d_buf.release()
            print "\a"
            if quiet!=True: print "Simulation finished"
            return self.ret_val



    class _SIMOBJECT:
        def __init__(self, positions, velocities, coils, num_particles, steps, bytesize=4, iter_nth = 1, dt = .0000000000002, num_coils = 2, avg_velo = 0, c_charge = 0.0):
            self.positions = positions.astype(np.float64)
            self.velocities = velocities.astype(np.float64)
            self.coils = np.array(coils).astype(np.float32)
            self.num_particles = np.int32(num_particles)
            self.num_steps = np.int32(steps)
            self.bytesize = bytesize
            self.ee_table = ellipe_table.astype(np.float32)
            self.ek_table = ellipk_table.astype(np.float32)

            self.dt = np.float64(dt)
            self.iter_nth = np.int32(iter_nth)
            self.num_coils = np.int32(num_coils)

            self.sim_properties = np.asarray([self.num_particles, self.num_steps, self.iter_nth, self.num_coils]).astype(np.int32)

            self.avg_velo = avg_velo
            self.c_spheres = np.asarray([c_charge]*num_particles, dtype = np.float64)

        def get_conf_times(self, store=True):
            conf_times = []
            #print radius, z_pos, dt, iter_nth
            radius = self.coils[0][1]
            z_pos = self.coils[1][0]
            dt = self.dt
            iter_nth = self.iter_nth
            r_vals = self.r_vals

            for p in range(len(r_vals)) :
                x_conf = len(np.where( abs(r_vals[p][:,0]) < radius)[0]) * dt * iter_nth * 1e9
                y_conf = len(np.where( abs(r_vals[p][:,1]) < radius)[0]) * dt * iter_nth * 1e9
                z_conf = len(np.where( abs((z_pos/2.0) - r_vals[p][:,2]) < (z_pos/2.0))[0]) * dt * iter_nth * 1e9

                conf_times.append(np.amin([x_conf,y_conf,z_conf]))

            if(store):
                self.conf_times = conf_times
            else:
                return conf_times

        def graph_conf_times(self, markersize = .5):
            def graph_clicked(event):
                print "clicked"
                self.graph_trajectory(int(event.x))

            fig = plt.figure()
            fig.canvas.mpl_connect('button_press_event', graph_clicked)
            plt.subplot(121)
            plt.scatter(range(len(self.conf_times)), self.conf_times, s = markersize)
            plt.show()
            plt.title("Mean time: " + str(np.mean(self.conf_times)) + " | First 20% Mean: " + str(np.mean(self.conf_times[0:int(0.2 * len(self.conf_times))])))

        def graph_trajectory(self, run_id):
            positions = self.r_vals[run_id]
            coil_1 = self.coils[run_id*self.num_coils]
            coil_2 = self.coils[run_id*self.num_coils+1]
            r = coil_1[1] # the radius of the circle

            steps = len(positions)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(positions[:,0], positions[:,1], zs= positions[:,2])
            ax.set_xlim([r, r * -1])
            ax.set_ylim([r, r * -1])
            ax.set_zlim([0, coil_2[0]])

            theta = np.linspace(0, 2*np.pi, 100)


            # compute x1 and x2

            loop_x = r*np.cos(theta)
            loop_y = r*np.sin(theta)
            loop_z=0
            ax.plot(loop_x,loop_y, loop_z)
            ax.plot(loop_x,loop_y, coil_2[0])

            ax.scatter(positions[0][0],positions[0][1],positions[0][2], color="green")

            ax.scatter(positions[steps-2][0],positions[steps-2][1],positions[steps-2][2], color="red")



    class _COIL:
        def __init__(self, radius = 0.05, current = 10000, z_pos = 0.0):
            self.radius = radius
            self.current = current
            self.z_pos = z_pos
            self.position = [0.0, 0.0, z_pos, 0.0]
            self.B_0 = self.current * mu_0 / (2.0 * self.radius)
            self.arr = np.array([z_pos, radius, self.B_0, 0.0]).astype(np.float32)

    def single_sim(self, device_id = 0):
        # Generate a single electron pos data
        # best of 1105.824 at -5000, 5000 [ 0,0.0004, -.03], [0,-5e3, 7e5]

        sp_charge = -1e-8
        #sp_charge = -15e-9
        ct = 23000
        #ct = 20000
        major_R = .014
        #major_R = .006
        zvelo = 1e6
        coil_1 = self._COIL( radius = .1, current = ct, z_pos = 0.0 )
        coil_2 = self._COIL( radius = .1, current = -ct, z_pos = 0.1)
        #coil_3 = self._COIL( radius = .03, current = 3000, z_pos = 0.06 )
        #coil_4 = self._COIL( radius = .03, current = -3000, z_pos = -.01 )

        coils = [coil_1.arr, coil_2.arr]#, coil_3.arr, coil_4.arr]

        # Constants
        e_charge = 1.6e-19  # Coulombs
        e_mass = 9.1e-31    # Kilograms
        e_gun_energy = 0 # measured in volts
        avg_velo = sqrt( (2.0 * e_gun_energy * e_charge) / e_mass) # m/s

        positions = np.array([[0.0000 , major_R, -0.03, 0.0,]])
        #velocities = np.array([[0.0, 0, avg_velo ,0.0,]]) #9.70017400e+05
        #velocities = np.array([[1e2, 0, avg_velo,0.0,]]) #9.70017400e+05
        velocities = np.array([[1e3, 0, zvelo]]) #9.70017400e+05

        print velocities

        #coils[0][2] = 0.06578967
        #coils[1][2] = -0.06578967


        num_particles = 1
        steps = 350000; #350000;
        bytesize = 16
        iter_nth = 36;
        dt = .0000000000002

        self.SINGLE_SIM = self._SIMOBJECT(positions, velocities, coils, num_particles, steps,num_coils = len(coils), bytesize = bytesize, iter_nth=iter_nth, dt = dt, c_charge = sp_charge)# -3e-11)#, c_charge = -1e-7)
        self.SINGLE_SIM.calculator = self._GPU(path_to_integrator, device_id)

        self.SINGLE_SIM.r_vals = self.SINGLE_SIM.calculator.execute( self.SINGLE_SIM)

        a = self.SINGLE_SIM.r_vals[0]

        self.SINGLE_SIM.graph_trajectory(0);

        self.SINGLE_SIM.get_conf_times()
        #self.SINGLE_SIM.conf_times = self.get_conf_times(self.SINGLE_SIM.r_vals, coil_1.radius, coil_2.z_pos, dt, iter_nth)
        #self, r_vals, radius, z_pos, dt, iter_nth

        print "Total confinement:", self.SINGLE_SIM.conf_times[0]
        plt.title(("Total confinement:", self.SINGLE_SIM.conf_times[0], " ns"))
        plt.show()


    def generic_simulation(self, num_particles = 10000, steps = 9000000, egun_energy = 1000, coil_current = 5000, e_gun_z = -.03, c_charge = 0.0, injection_radius= .0006,memory = 3000000000):
        coil_1 = self._COIL( radius = .05, current = coil_current, z_pos = 0.0 )
        coil_2 = self._COIL( radius = .05, current = coil_current*-1.0, z_pos = 0.05 )
        coils = [coil_1.arr, coil_2.arr]

        # Control parameters
        memory = memory
        bytesize = 16

        num_particles = num_particles
        total_steps = steps # ten million
        dt = .0000000000002

        mem_p_particle = memory/num_particles # can serve so many bytes to display
        steps = mem_p_particle/bytesize
        iter_nth = total_steps/steps
        print "Steps: ",steps," iter_nth: ", iter_nth

        e_gun_energy = egun_energy # measured in volts
        avg_velo = sqrt( (2.0 * e_gun_energy * e_charge) / e_mass) # m/s

        positions = np.tile( [0.0 ,injection_radius, e_gun_z, 0.0], (num_particles, 1))
        velocities = np.tile ([1e3, 0.0, avg_velo, 0.0],(num_particles, 1) )
        coils = np.tile(coils,(num_particles, 1) )

        c_spheres = np.asarray([c_charge]*num_particles, dtype=np.float64)

        return self._SIMOBJECT(positions, velocities, coils, num_particles, steps, bytesize = bytesize, iter_nth=iter_nth, dt = dt, avg_velo = avg_velo, c_charge = c_charge)


    def nd_paramspace(self, data, device_id = 2):
        '''
        Data is an array shaped into a set of paramters for the simulation
        Data is not a meshgrid, but rathter a list of arrays for each paramter.
         a[0] = injection_radius
         a[1] = Z_velocitiy
         a[2] = coil_current
         a[3] = coil_separation
         a[4] = space_charge
        '''
        paramspace = np.array(np.meshgrid(*data)).T.reshape(-1, len(data))
        num_particles = len(paramspace)

        positions = np.zeros((num_particles, 4))
        positions[:,1] = paramspace[:,0]
        positions[:,2] = -.03

        velocities = np.zeros((num_particles, 4))
        velocities[:,2] = paramspace[:,1]
        velocities[:,0] = 1e3

        # z, r, B_0
        coil_radius = 0.05
        coil_current = paramspace[:,2]
        coil_separation = paramspace[:,3]
        coils = np.zeros((num_particles*2, 4)).astype(np.float32)
        coils[:,0][1::2] = coil_separation
        coils[:,1] = coil_radius                                           # Coil radius
        coils[:,2] = coil_current.repeat(2) * mu_0 / (2.0 * coil_radius)
        coils[:,2][1::2] *= -1.0


        # we want 1000 location points per run
        # 3gb / 1000 = 750000 max_particles per run (memory limited)

#        particles_per_run
        ppr = 65536
        num_runs = int(ceil(num_particles / float(ppr) ))

        print "Number of runs required: " + str(num_runs)
        self.simulations = []
        for i in range(int(num_runs)):
            self.simulations.append( self._SIMOBJECT(positions[ppr*i:ppr*(i+1)], velocities[ppr*i:ppr*(i+1)], coils[ppr*i:ppr*(i+1)], num_particles =ppr, steps = 400, num_coils = 2, dt = .0000000000002, bytesize = 16, iter_nth = 10000, c_charge = -1e-12))

        print "All simulations created"

        sim_id = 0
        for sim in self.simulations:
            print "Running simulation - " + str(sim_id)
            if sim_id > -1: # change this id to skip over runs if gpu crashes
                sim.r_vals = self._GPU(path_to_integrator, device_id).execute(sim) # Returns r_vals
                np.save("simulations/Simulation - part "+str(sim_id), sim.get_conf_times(store=False))
            sim_id+=1

        print 'Simulations complete'

        #self.GUN_L.calculator = self._GPU(path_to_integrator, device_id)
        #self.GUN_L.r_vals = self.GUN_L.calculator.execute(self.GUN_L)





    def paramspace_per_sc(self, device_id):
        slices = 25

        injection_radius = np.linspace(0.0005, 0.005, slices)
        z_velocitiy = np.linspace(.5e6, 5e7, slices)
        coil_current = np.linspace(5000.0, 15000.0, slices)
        coil_separation = np.linspace(0.03, 0.1, slices)

        r_vals = self.nd_paramspace([injection_radius,z_velocitiy,coil_current,coil_separation])





    def gun_v_l(self, device_id=2):
        self.GUN_L = self.generic_simulation(egun_energy=1000, coil_current=40000)

        position_arr = np.linspace(0, -0.05, self.GUN_L.num_particles )
        self.GUN_L.positions[:,2] = position_arr

        self.GUN_L.calculator = self._GPU(path_to_integrator, device_id)
        self.GUN_L.r_vals = self.GUN_L.calculator.execute(self.GUN_L)

        self.GUN_L.conf_times = self.GUN_L.get_conf_times()

    def gun_v_l(self, device_id=2):
        self.GUN_L = self.generic_simulation(egun_energy=1000, coil_current=40000)

        position_arr = np.linspace(0, -0.05, self.GUN_L.num_particles )
        self.GUN_L.positions[:,2] = position_arr

        self.GUN_L.calculator = self._GPU(path_to_integrator, device_id)
        self.GUN_L.r_vals = self.GUN_L.calculator.execute(self.GUN_L)

        self.GUN_L.conf_times = self.GUN_L.get_conf_times()













    def r_v_E(self, device_id = 2):
        self.GUN_L = self.generic_simulation(num_particles = 32768, egun_energy=500, coil_current=1000, e_gun_z = -.1)

        r_lin = np.tile(np.linspace(-0.0001, -0.001, 32 ), (1, 32))[0]
        l_lin = np.linspace(-.02, -.06, 32).repeat(32)
        v_lin = (np.linspace(.01, 1, 32) * self.GUN_L.avg_velo).repeat(1024)

        self.GUN_L.positions[:,0] = r_lin.repeat(32)
        self.GUN_L.positions[:,2] = l_lin.repeat(32)
        self.GUN_L.velocities[:,2] = v_lin


        self.GUN_L.calculator = self._GPU(path_to_integrator, device_id)
        self.GUN_L.r_vals = self.GUN_L.calculator.execute(self.GUN_L)

        self.GUN_L.conf_times = self.GUN_L.get_conf_times()

        self.GUN_L.graph_conf_times()

    def egunE_v_CC(self,device_id = 2):
        cc_slices = 100
        ee_slices = 150
        cc = 10000
        ee = 3000
        row = cc_slices
        col = ee_slices
        self.GUN_L = self.generic_simulation(num_particles = (row*col), egun_energy=ee, coil_current=cc, e_gun_z = -.03, c_charge = -1e-9)
        v_lin = (np.linspace(.01, 1, col) * self.GUN_L.avg_velo).repeat(row)
        v_lin = (np.linspace(.01, 1, col) * z.GUN_L.avg_velo).repeat(row)
        CC_lin = np.linspace(1, cc, col).repeat(2)
        flip = np.ones(2 * col)
        flip[1::2] = flip[1::2]*-1
        CC_lin = CC_lin * flip * mu_0 / (2.0 * .05)
        self.GUN_L.positions[:,0] = np.asarray([0.0008]*row*col)
        self.GUN_L.velocities[:,2] = v_lin
        self.GUN_L.coils[:,2] = np.tile(CC_lin, (1,row))
        self.GUN_L.coils[:,0][1::2] = 0.05

        self.GUN_L.calculator = self._GPU(path_to_integrator, device_id)
        self.GUN_L.r_vals = self.GUN_L.calculator.execute(self.GUN_L)
        self.GUN_L.get_conf_times()

        self.GUN_L.graph_conf_times()
        plt.subplot(122)
        hm = sb.heatmap(np.asarray(self.GUN_L.conf_times).reshape(row,col), xticklabels=5, yticklabels=5, robust=False)
        hm.invert_yaxis()
        plt.title("EGUN Energy max: "+str(ee) + " | Coil Current max: " + str(cc))
        plt.show()

    def crit_val_show(self,device_id = 2):
        num_slices = 1500

        crit = 6.4e6
        velo = 592999.453328881
        v_lin = np.linspace(velo, 10000*velo, num_slices)
        CC_lin = v_lin / crit



        cc = 10000
        #row = cc_slices
        #col = ee_slices
        self.GUN_L = self.generic_simulation(num_particles = (num_slices), e_gun_z = -.03)

        #r_lin = np.tile(np.linspace(0, -0.005, 32 ), (1, 32))[0]
        #l_lin = np.linspace(-.01, -.07, 32).repeat(32)
        #v_lin = (np.linspace(.01, 1, col) * self.GUN_L.).repeat(row)
        #v_lin = (np.linspace(.01, 1, col) * z.GUN_L.).repeat(row)

        flip = np.ones(2 * num_slices)
        flip[1::2] = flip[1::2]*-1
        CC_lin = CC_lin.repeat(2) * flip

        #v_lin = CC_lin[0::2] * 10000000.0

        #        self.GUN_L.positions[:,0] = r_lin.repeat(32)
        #        self.GUN_L.positions[:,2] = l_lin.repeat(32)
        self.GUN_L.velocities[:,2] = v_lin
        self.GUN_L.coils[:,2] = CC_lin

        self.GUN_L.calculator = self._GPU(path_to_integrator, device_id)
        self.GUN_L.r_vals = self.GUN_L.calculator.execute(self.GUN_L)
        self.GUN_L.get_conf_times()

        self.GUN_L.graph_conf_times()
        #plt.subplot(122)
        #hm = sb.heatmap(np.asarray(self.GUN_L.conf_times).reshape(num_slices,1), xticklabels=5, yticklabels=5, robust=False)
        #hm.invert_yaxis()
        #plt.title("EGUN Energy max: "+str(ee) + " | Coil Current max: " + str(cc))
        #plt.show()


    def active_optimizer(self, device_id = 0, optimizer = 0):
        # Spins up an instance for every parameter changed and looks at which parameter positively impacted the simulation.
        # Sets new simulation to that paramter and retries over and over until it getss stuck


        num_particles = 4
        leap_factor = 1.02
        parameters = {"sp_charge":-11e-12 , "coil_current": 6990.0 , 'injection_radius': 0.00050, 'velocity': 12e5}
        coil_1 = self._COIL( radius = .05, current = parameters['coil_current'], z_pos = 0.0 )
        coil_2 = self._COIL( radius = .05, current = -parameters['coil_current'], z_pos = 0.05)
        coils = [coil_1.arr, coil_2.arr]

        if (optimizer == 0):
            self.OPTIMIZER = self.generic_simulation(num_particles = num_particles, e_gun_z = -.03, coil_current = parameters['coil_current'], c_charge = parameters['sp_charge'], injection_radius = parameters['injection_radius'], memory = 12000000)
            self.OPTIMIZER.velocities[:,2] = parameters['velocity']
            #sel   f.OPTIMIZER.coils = [coils
            self.OPTIMIZER.calculator = self._GPU(path_to_integrator, device_id)

        self.conf_times_over_time = []
        for i in range(100):
            self.OPTIMIZER.c_spheres *= np.asarray([leap_factor, 1.0, 1.0, 1.0])
            self.OPTIMIZER.coils[:,2] *= np.asarray([1.0, leap_factor, 1.0, 1.0]).repeat(2)
            self.OPTIMIZER.positions[:,1] *= np.asarray([1.0, 1.0, leap_factor, 1.0])
            self.OPTIMIZER.velocities[:,2] *= np.asarray([1.0, 1.0, 1.0, leap_factor])
            self.OPTIMIZER.r_vals = self.OPTIMIZER.calculator.execute(self.OPTIMIZER, quiet=True)
            self.OPTIMIZER.get_conf_times()
            #self.OPTIMIZER.graph_conf_times(markersize = 10)

            best_run = np.argmax(self.OPTIMIZER.conf_times)

            if best_run == 0:
                #print "Raised sp_charge: " + str(self.OPTIMIZER.)
                self.OPTIMIZER.coils[:,2] *= np.asarray([1.0, 1.0/leap_factor, 1.0, 1.0]).repeat(2);self.OPTIMIZER.positions[:,1] *= np.asarray([1.0, 1.0, 1.0/leap_factor, 1.0]);self.OPTIMIZER.velocities[:,2] *= np.asarray([1.0, 1.0, 1.0, 1.0/leap_factor])
                self.OPTIMIZER.c_spheres =self.OPTIMIZER.c_spheres[0].repeat(4)
            if best_run == 1:
                self.OPTIMIZER.c_spheres *= np.asarray([1.0/leap_factor, 1.0, 1.0, 1.0]);self.OPTIMIZER.positions[:,1] *= np.asarray([1.0, 1.0, 1.0/leap_factor, 1.0]);self.OPTIMIZER.velocities[:,2] *= np.asarray([1.0, 1.0, 1.0, 1.0/leap_factor])
                self.OPTIMIZER.coils[:,2] = np.tile(self.OPTIMIZER.coils[:,2][2:4].reshape(2,1), (4,1)).reshape(8)
            if best_run == 2:
                self.OPTIMIZER.c_spheres *= np.asarray([1.0/leap_factor, 1.0, 1.0, 1.0]); self.OPTIMIZER.coils[:,2] *= np.asarray([1.0, 1.0/leap_factor, 1.0, 1.0]).repeat(2);self.OPTIMIZER.velocities[:,2] *= np.asarray([1.0, 1.0, 1.0, 1.0/leap_factor])
                self.OPTIMIZER.positions[:,1] = self.OPTIMIZER.positions[:,1][2].repeat(4)
            if best_run == 3:
                self.OPTIMIZER.c_spheres *= np.asarray([1.0/leap_factor, 1.0, 1.0, 1.0]); self.OPTIMIZER.coils[:,2] *= np.asarray([1.0, 1.0/leap_factor, 1.0, 1.0]).repeat(2);self.OPTIMIZER.positions[:,1] *= np.asarray([1.0, 1.0, 1.0/leap_factor, 1.0]);
                self.OPTIMIZER.velocities[:,2] = self.OPTIMIZER.velocities[:,2][3].repeat(4)
            self.conf_times_over_time.append(np.max(self.OPTIMIZER.conf_times))
            print "Stepped: " + str(i) + " | Max Time: " + str(np.max(self.OPTIMIZER.conf_times)) + " Best_run = "+str(best_run)

        self.OPTIMIZER.graph_conf_times(markersize = 10)
        self.OPTIMIZER.graph_trajectory(best_run)
        # now have a simulation with 4 particles, initial charge, current, velocity

        #def generic_simulation(self, num_particles = 10000, steps = 9000000, egun_energy = 1000, coil_current = 5000, e_gun_z = -.03, c_charge = 0.0):


#path_to_integrator = '/Users/jonkelley/Desktop/temp_potentia/potential_optimizer/part1.cl'
#z.dim_by_dim()
#z.single_sim()
#z.EGUNvsDIST()
#z.single_sim()

import os
script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
script_dir = os.path.split(script_path)[0] #i.e. /path/to/dir/
rel_path = "part1.cl"
#rel_path = "trajectory_conf.cl"
path_to_integrator = os.path.join(script_dir, rel_path)

z = 0;

if __name__ == "__main__":
    z = all()
    simulations = {
        'single':z.single_sim,
        'gun_v_l':z.gun_v_l,
        'r_v_E':z.r_v_E,
        'egunE_v_CC':z.egunE_v_CC,
        'crit_val_show':z.crit_val_show,
        'active_optimizer':z.active_optimizer,
        'paramspace_per_sc':z.paramspace_per_sc

    }
    if len(sys.argv) == 1:
    #    rel_path = 'part1.cl'

        print "single sim"
        z.single_sim(0)
    else:
        if sys.argv[1] == "active_optimizer":
            if len(sys.argv) == 3:
                simulations[sys.argv[1]](int(sys.argv[2]),optimizer = 0)
            else:
                simulations[sys.argv[1]](int(sys.argv[2]),sys.argv[3])
        else:
            simulations[sys.argv[1]](int(sys.argv[2]))

        # hi
        # %run potential_optimizer.py{'single'} {0}

sim = z
