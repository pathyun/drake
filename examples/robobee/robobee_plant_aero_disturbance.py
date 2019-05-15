import math
import numpy as np

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    SignalLogger,
    VectorSystem,
    RotationMatrix,
    Quaternion,
    DrakeLcm,
    DrakeVisualizer, FloatingBaseType,
    RigidBodyPlant, RigidBodyTree,
    )
from pydrake.common import FindResourceOrThrow
# Based on RobobeePlant
# Backstepping input
# \dot{\Xi}_1 = \Xi_2
# \dot{\Xi}_2 = u
# This class takes as input the physical description
# of the system, in terms of the center of mass intertia matrix, and gravity

class RobobeePlantAero(VectorSystem):
    def __init__(self, m = 1., Ixx = 1., 
                       Iyy = 2., Izz = 3., g = 10., rw_l = 1, bw=1, bw_bound =0.3,
                        input_max = 10.):
        VectorSystem.__init__(self,
            4,                           # One input (torque at reaction wheel).
            13)                           # Four outputs (theta, phi, dtheta, dphi)
        self._DeclareContinuousState(13)  # Four states (theta, phi, dtheta, dphi).

        self.m = float(m)
        self.Ixx = float(Ixx)
        self.Iyy = float(Iyy)
        self.Izz = float(Izz)
        self.g = float(g)
        self.rw_l = float(rw_l)
        self.bw = float(bw)*1.0
        self.bw_bound = float(bw_bound)*1.0
        self.input_max = float(input_max)

        # Go ahead and calculate rotational inertias.
        # Treat the first link as a point mass.
        ### self.I1 = self.m1 * self.l1 ** 2
        # Treat the second link as a disk.
        ### self.I2 = 0.5 * self.m2 * self.r**2

    # This method returns (R(q), E(q), wx (I w), I_inv)
    # according to the dynamics of this system.
    def GetManipulatorDynamics(self, q, qd):
        # Input argument
        #- q = [x, y, z, q0, q1, q2, q3]\in \mathbb{R}^7
        #- qd= [vx, vy, vz, wx, wy, wz ]\in \mathbb{R}^6

        x= q[0]     # (x,y,z) in inertial frame
        y= q[1]
        z= q[2]
        q0= q[3]    # q0+q1i+q2j+qk 
        q1= q[4]
        q2= q[5]
        q3= q[6]
        

        vx= qd[0]   # CoM velocity in inertial frame
        vy= qd[1]
        vz= qd[2]
        wx= qd[3]   # Body velocity in "body frame"
        wy= qd[4]
        wz= qd[5]

        # Stack up the state q
        x = np.array([q, qd])
        # print("x:",x)
        qv = np.array([q1,q2,q3])
        r = np.array([x,y,z])
        # print("qv",qv.shape)
        v = np.array([vx,vy,vz])
        quat_vec = np.vstack([q0,q1,q2,q3])
        # print("quat_vec",quat_vec.shape)
        w =np.array([wx,wy,wz])
        q_norm=np.sqrt(np.dot(quat_vec.T,quat_vec))
        q_normalized = quat_vec/q_norm
        # print("q_norm: ",q_norm)

        quat = Quaternion(q_normalized)    # Quaternion
        Rq = RotationMatrix(quat).matrix()  # Rotational matrix
        # print("\n#######")
        
        # Translation from w to \dot{q}
        Eq = np.zeros((3,4))
        w_hat = np.zeros((3,4))


        # Eq for body frame
        Eq[0,:] = np.array([-1*q1,    q0,  1*q3, -1*q2]) # [-1*q1,    q0, -1*q3,    q2]
        Eq[1,:] = np.array([-1*q2, -1*q3,    q0,  1*q1]) # [-1*q2,    q3,    q0, -1*q1]
        Eq[2,:] = np.array([-1*q3,  1*q2, -1*q1,    q0]) # [-1*q3, -1*q2,    q1,    q0]
        
        # Eq for world frame
        # Eq[0,:] = np.array([-1*q1,    q0, -1*q3,    q2])
        # Eq[1,:] = np.array([-1*q2,    q3,    q0, -1*q1])
        # Eq[2,:] = np.array([-1*q3, -1*q2,    q1,    q0])
        

      #  quat_dot = np.dot(Eq.T,w)/2.   # \dot{quat}=1/2*E(q)^Tw

        # w x (Iw)
        
        I=np.zeros((3,3))       # Intertia matrix
        Ixx= self.Ixx;
        Iyy= self.Iyy;
        Izz= self.Izz;
        I[0,0]=Ixx;
        I[1,1]=Iyy;
        I[2,2]=Izz;
        I_inv = np.linalg.inv(I);
        Iw = np.dot(I,w)
        wIw = np.cross(w.T,Iw.T).T
        
        # Aerodynamic drag and parameters
        e1 = np.zeros(3);
        e1[0]=1;
        e3 = np.zeros(3);
        e3[2]=1;
        r_w = self.rw_l*e3;
        wr_w = np.cross(w.T,r_w) # w x r
        fdw = -self.bw*(v+np.dot(Rq,wr_w))# Gaussian with variance gamma
        # fdw = -self.bw*(v+np.dot(Rq,wr_w))*(1+(2*np.random.rand(1)-1.)*self.bw_bound) # Uniform distribution over [-gamma, gamma]
        # fdw = -self.bw*(v+np.dot(Rq,wr_w))*(1+(np.random.rand(1))*np.sqrt(self.bw_bound)) # Gaussian with variance gamma
        # print("fdw: ", np.shape(v))
        
        RqTfdw=np.dot(Rq.T,fdw)
        taudw = np.cross(r_w,RqTfdw)
        # print("fdw: ", fdw)
        # print("w: ", w)
        # print("Rq.Twr_w :", np.dot(Rq.T,fdw))

        # print("Rqwr_w :", np.dot(Rq,fdw))
        
        taudw_b = -self.bw*(np.dot(Rq.T,v)+np.cross(w,r_w))
        taudw_b = np.cross(r_w,taudw_b)
        # print("taudw - taudw_b", taudw-taudw_b)
        
        vd_aero =  np.dot((-self.g*np.eye(3)),e3) + fdw/self.m # \dot{v} = -ge3 +R(q)e3 u[0] : u[0] Thrust is a unit of acceleration
        wd_aero = -np.dot(I_inv,wIw)+ np.dot(I_inv, taudw)
        # kd = 1.5*1e-1
        # wd = -np.dot(I_inv,wIw)+np.dot(I_inv,-kd*w) + np.dot(I_inv, taudw)
    
        # print("taudw: ", np.dot(Rq.T,fdw.T))
        w_hat = np.zeros((3,3))
        w_hat[0,:] = np.array([     0,   -w[2],     w[1] ])
        w_hat[1,:] = np.array([  w[2],       0,    -w[0] ])
        w_hat[2,:] = np.array([ -w[1],    w[0],        0 ])

        # wrw_byhat = np.dot(w_hat,r_w)

        # print("wr_w - wrw_byhat", wr_w - wrw_byhat)

           

        return (Rq, Eq, wIw, I_inv, fdw, taudw, vd_aero, wd_aero)

    # This helper uses the manipulator dynamics to evaluate
    # \dot{x} = f(x, u). It's just a thin wrapper around
    # the manipulator dynamics. If throw_when_limits_exceeded
    # is true, this function will throw a ValueError when
    # the input limits are violated. Otherwise, it'll clamp
    # u to the input range.
    def evaluate_f(self, u, x, throw_when_limits_exceeded=True):
        # Bound inputs
        if throw_when_limits_exceeded and abs(u[0]) > self.input_max:
            raise ValueError("You commanded an out-of-range input of u=%f"
                              % (u[0]))
        else:
            u[0] = max(-self.input_max, min(self.input_max, u[0]))

        # Use the manipulator equation to get qdd.
        qq = x[0:7]
        qqd = x[7:13]
        wx= qqd[3]   # Body velocity in "body frame"
        wy= qqd[4]
        wz= qqd[5]
        w =np.array([wx,wy,wz])
        
        (Rq, Eq, wIw, I_inv, fdw, taudw, vd_aero, wd_aero) = self.GetManipulatorDynamics(qq, qqd)
        
        e3 = np.array([0,0,1])
        # print("e3,", e3.shape)
        
        # Awkward slice required on tauG to get shapes to agree --        # numpy likes to collapse the other dot products in this expression
        # to vectors.
        # epsilonn = 0.01 # Error pe
        rd = qqd[0:3] # np.vstack([qd[0],qd[1],qd[2]]); 
        qd = np.dot(Eq.T,qqd[3:6])/2.   # \dot{quat}=1/2*E(q)^Tw
        
        vd = vd_aero+np.dot(Rq,e3)*u[0]
        wd = wd_aero+np.dot(I_inv, u[1:4])
        
        # print("Rq", Rq.shape, "u", u.shape)
        # print("rd",rd.shape, "qd", qd.shape, "vd", vd.shape, "wd",wd.shape)

        # print("Eq:",Eq.T)
        # print("qqd:", qqd)
        # print("qd:",qd)
        # print("vd:", vd)
        # print("w:",qqd[3:6])
        # print("w shape:", np.shape(taudw))
        # print("I_inv:",I_inv)
        # print("w:",w)
        # print("v:",v)
        # print("rw:",rw)
        # print("Vw:",Vw)
        # print("fdw_b:",fdw_b)
        # print("fdw:", fdw/self.m)
        # print("taudw:", np.dot(I_inv, taudw))
        # print("taud_w:",taud_w)
        # print("u:",u)
        # # wd_tot = wd+taud_w
        # # vd_tot = vd+fdw/self.m

        return np.hstack([rd, qd, vd, wd])


    # This method calculates the time derivative of the state,
    # which allows the system to be simulated forward in time.
    def _DoCalcVectorTimeDerivatives(self, context, u, x, xdot):
        q = x[0:7]
        qd = x[7:13]
        xdot[:] = self.evaluate_f(u, x, throw_when_limits_exceeded=True)

    # This method calculates the output of the system
    # (i.e. those things that are visible downstream of
    # this system) from the state. In this case, it
    # copies out the full state.
    def _DoCalcVectorOutput(self, context, u, x, y):
        y[:] = x

    # The Drake simulation backend is very careful to avoid
    # algebraic loops when systems are connected in feedback.
    # This system does not feed its inputs directly to its
    # outputs (the output is only a function of the state),
    # so we can safely tell the simulator that we don't have
    # any direct feedthrough.
    def _DoHasDirectFeedthrough(self, input_port, output_port):
        if input_port == 0 and output_port == 0:
            return False
        else:
            # For other combinations of i/o, we will return
            # "None", i.e. "I don't know."
            return None

    # # The method return matrices (A) and (B) that encode the
    # # linearized dynamics of this system around the fixed point
    # # u_f, x_f.
    # def GetLinearizedDynamics(self, u_f, x_f):
    #     x= x_f[0]     # (x,y,z) in inertial frame
    #     y= x_f[1]
    #     z= x_f[2]
    #     q0= x_f[3]    # q0+q1i+q2j+qk 
    #     q1= x_f[4]
    #     q2= x_f[5]
    #     q3= x_f[6]
    #     # print("xf:", x_f)
    #     vx= x_f[7]   # CoM velocity in inertial frame
    #     vy= x_f[8]
    #     vz= x_f[9]
    #     wx= x_f[10]   # Body velocity in "body frame"
    #     wy= x_f[11]
    #     wz= x_f[12]

    #     F_T_f = u_f[0]
    #     tau_f = u_f[1:4]

    #     # You might want at least one of these.
    #     # (M, C_f, tauG_f, B_f) = self.GetManipulatorDynamics(q_f, qd_f)

    #     I_inv = np.zeros((3,3))
    #     I_inv[0,0] = 1/self.Ixx
    #     I_inv[1,1] = 1/self.Iyy
    #     I_inv[2,2] = 1/self.Izz
        

    #     Jac_f_x = np.zeros((15,15))
    #     Jac_f_u = np.zeros((15,4))

    #     W_1 = np.array([[0, -wx, -wy, -wz],
    #                     [wx,  0, -wz,  wy],
    #                     [wy, wz,   0, -wx],
    #                     [wz,-wy,  wx,   0]])
    #     F1 = np.zeros((3,4))
    #     F1[0,:] = np.array([   q2,    q3,    q0,    q1])
    #     F1[1,:] = np.array([-1*q1, -1*q0,    q3,    q2])
    #     F1[2,:] = np.array([   q0, -1*q1, -1*q2,    q3])

    #     Eq = np.zeros((3,4))

    #     # Eq for body frame
    #     Eq[0,:] = np.array([-1*q1,    q0,  1*q3, -1*q2]) # [-1*q1,    q0, -1*q3,    q2]
    #     Eq[1,:] = np.array([-1*q2, -1*q3,    q0,  1*q1]) # [-1*q2,    q3,    q0, -1*q1]
    #     Eq[2,:] = np.array([-1*q3,  1*q2, -1*q1,    q0]) # [-1*q3, -1*q2,    q1,    q0]
        
    #     # Eq for world frame
    #     # Eq[0,:] = np.array([-1*q1,    q0, -1*q3,    q2])
    #     # Eq[1,:] = np.array([-1*q2,    q3,    q0, -1*q1])
    #     # Eq[2,:] = np.array([-1*q3, -1*q2,    q1,    q0])


    #     A1 = -1*(self.Izz-self.Iyy)/self.Ixx
    #     A2 = -1*(self.Ixx-self.Izz)/self.Iyy
    #     A3 = -1*(self.Iyy-self.Ixx)/self.Izz
    #     W_2 = np.array([[    0, A1*wz, A1*wy],
    #                     [A2*wz,     0, A2*wx],
    #                     [A3*wy, A3*wx,   0]])
        
    #     # Jacobian over q
    #     Jac_f_x[3:7,3:7]=W_1
    #     Jac_f_x[8:11,3:7]=2*F1*F_T_f

    #     # Jacobian over xi1
    #     Jac_f_x[8,7]=2*q3*q1 + 2*q0*q2
    #     Jac_f_x[9,7]=2*q3*q2 - 2*q0*q1
    #     Jac_f_x[10,7]=q0*q0   + q3*q3   -q1*q1  -q2*q2
    #     # Jacobian over v
    #     Jac_f_x[0:3,8:11]=np.eye(3)
        
    #     # Jacobian over w
    #     Jac_f_x[3:7,11:14]=Eq.T/2
    #     # print("Eq.T",Eq.T)
    #     # print("Eq.T/2",Eq.T/2)
        
    #     Jac_f_x[11:14,11:14]=W_2

    #     # Jacobian over xi2

    #     Jac_f_x[7,14] = 1
    
    #     # Jacobian over u
    #     Jac_f_u[14,0]= 1
    #     # print("Jac_f_u:", Jac_f_u)
    #     Jac_f_u[11:14,1:4] = I_inv
        
    #     A = Jac_f_x
    #     B = Jac_f_u
    #     return (A, B)

class RobobeeController(VectorSystem):
    ''' System to control the robobee. Must be handed
    a function with signature:
        u = f(t, x)
    that computes control inputs for the pendulum. '''

    def __init__(self, feedback_rule):
        VectorSystem.__init__(self,
            13,                           # Four inputs: full state inertial wheel pendulum..
            4)                           # One output (torque for reaction wheel).
        self.feedback_rule = feedback_rule

    # This method calculates the output of the system from the
    # input by applying the supplied feedback rule.
    def _DoCalcVectorOutput(self, context, u, x, y):
        # Remember that the input "u" of the controller is the
        # state of the plant
        time_t = context.get_time()
        print("time_t:", time_t)

        y[:] = self.feedback_rule(u,time_t)
        # Hybrid switching controller
        # if time_t<2:
        #     y[:] = self.feedback_rule(u,time_t)
        # elif time_t>2 and time_t<2.2:
        #     y[:] = np.zeros(4);
        # else:
        #     y[:] = self.feedback_rule(u,time_t)

# class RigidBodySelection(VectorSystem):
#     ''' System to control the robobee. Must be handed
#     a function with signature:
#         u = f(t, x)
#     that computes control inputs for the pendulum. '''

#     def __init__(self):
#         VectorSystem.__init__(self,
#             18,                           # Four inputs: full state inertial wheel pendulum..
#             13)                           # One output (torque for reaction wheel).
        
#     # This method calculates the output of the system from the
#     # input by applying the supplied feedback rule.
#     def _DoCalcVectorOutput(self, context, u, x, y):
#         # Remember that the input "u" of the controller is the
#         # state of the plant
#         y[:] = np.hstack([u[0:7],u[8:14]])

def RunSimulation(robobee_plantBS_torque, control_law, x0=np.random.random((13, 1)), duration=30):
    robobee_controller = RobobeeController(control_law)

    # Create a simple block diagram containing the plant in feedback
    # with the controller.
    builder = DiagramBuilder()
    # The last pendulum plant we made is now owned by a deleted
    # system, so easiest path is for us to make a new one.
    plant = builder.AddSystem(RobobeePlantAero(
        m = robobee_plantBS_torque.m,
        Ixx = robobee_plantBS_torque.Ixx, 
        Iyy = robobee_plantBS_torque.Iyy, 
        Izz = robobee_plantBS_torque.Izz, 
        g = robobee_plantBS_torque.g, 
        rw_l = robobee_plantBS_torque.rw_l, 
        bw = robobee_plantBS_torque.bw, 
        input_max = robobee_plantBS_torque.input_max))

    # Rigidbody_selector = builder.AddSystem(RigidBodySelection())

    print("1. Connecting plant and controller\n")
    controller = builder.AddSystem(robobee_controller)
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))

    # Create a logger to capture the simulation of our plant
    set_time_interval = 0.01
    time_interval_multiple = 1;
    publish_period = set_time_interval*time_interval_multiple

    print("2. Connecting plant to the logger\n")
    
    input_log = builder.AddSystem(SignalLogger(4))
    # input_log._DeclarePeriodicPublish(publish_period, 0.0)
    builder.Connect(controller.get_output_port(0), input_log.get_input_port(0))

    state_log = builder.AddSystem(SignalLogger(13))
    # state_log._DeclarePeriodicPublish(publish_period, 0.0)
    builder.Connect(plant.get_output_port(0), state_log.get_input_port(0))
    
    # Drake visualization
    print("3. Connecting plant output to DrakeVisualizer\n")
    
    rtree = RigidBodyTree(FindResourceOrThrow("drake/examples/robobee/robobee_arena.urdf"), FloatingBaseType.kQuaternion)
    lcm = DrakeLcm()
    visualizer = builder.AddSystem(DrakeVisualizer(tree=rtree,
       lcm=lcm, enable_playback=True))
    
    builder.Connect(plant.get_output_port(0),visualizer.get_input_port(0))  
    
    print("4. Building diagram\n")
    
    diagram = builder.Build()

    # Set the initial conditions for the simulation.
    context = diagram.CreateDefaultContext()
    state = context.get_mutable_continuous_state_vector()
    state.SetFromVector(x0)

    # Create the simulator.
    print("5. Create simulation\n")
    
    simulator = Simulator(diagram, context)
    simulator.Initialize()
    # simulator.set_publish_every_time_step(False)

    simulator.set_target_realtime_rate(1)
    simulator.get_integrator().set_fixed_step_mode(True)
    simulator.get_integrator().set_maximum_step_size(set_time_interval)

    # Simulate for the requested duration.
    simulator.StepTo(duration)
    
    visualizer.ReplayCachedSimulation()

    return input_log, state_log

