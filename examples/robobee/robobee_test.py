##########################333
#
# Robobee Trajectory optimization
# 
# Direct Collocation method
# 
#  Desired output y = [y_1; y_2]\in\mathbb{R}^4:_
#  y_1 = r
#  y_2 = yaw= atan(e2TR(q)e1/e1TR(q)e1)
#
#  Call RobobeePlantBS : 15 state 4 input
#  \dot{\Xi_1} =\Xi_2
#  \dot{\Xi_2} = u
#
################################


import math
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (DirectCollocation, FloatingBaseType,
                         PiecewisePolynomial, RigidBodyTree, RigidBodyPlant,
                         SolutionResult, Simulator, DrakeVisualizer, FloatingBaseType,
    DiagramBuilder,
    Simulator,
    SignalLogger,
    VectorSystem,
    RotationMatrix,
    Quaternion,
    DrakeLcm,
    RollPitchYaw)

from pydrake.common import FindResourceOrThrow

from pydrake.examples.robobee import RobobeePlant

from pydrake.solvers import mathematicalprogram as mp

# Make numpy printing prettier
np.set_printoptions(precision=3, suppress=True)

#-[0.0] Show figure flag

show_flag_qd = 1;
show_flag_q=1;
show_flag_control =1;

#-[0]  Initial configuration in SE(3)

r0 = np.array([0,0,0])
q0 = np.zeros((4)) # quaternion setup
theta0 = math.pi/4;  # angle of rotation
v0_q=np.array([1.,0.,1.])
q0[0]=math.cos(theta0/2) #q0
# print("q:",q[0])
v0_norm=np.sqrt((np.dot(v0_q,v0_q))) # axis of rotation
v0_normalized =v0_q.T/v0_norm
# print("vnorm:", v_norm)
# print("vnormalized", np.dot(v_normalized,v_normalized.T))
q0[1:4]=math.sin(theta0/2)*v0_q.T/v0_norm

v0 = np.zeros((3))
w0 = np.zeros((3))          # angular velocity
w0[0]=-0.1 #0;
w0[1]=0  #-0.1;
w0[2]=0.1 # 0.2;

#-[0-1] Stack up the state in R^13
# print("r:", r0.shape, "q",q0.shape,"v",v0.shape,"w", w0.shape)
x0= np.hstack([r0, q0, v0, w0])
print("x0:", x0)


#-[1] Final configuration in SE(3) 
rf = np.array([0,0,0.30])
qf = np.zeros((4)) # quaternion setup
thetaf = math.pi/4;  # angle of rotation

vf=np.array([0.,0.,1.])
qf[0]=math.cos(thetaf/2) #q0

# print("q:",q[0])
vf_norm=np.sqrt((np.dot(vf,vf))) # axis of rotation
vf_normalized =vf.T/vf_norm
# print("vnorm:", v_norm)
# print("vnormalized", np.dot(v_normalized,v_normalized.T))
qf[1:4]=math.sin(thetaf/2)*vf.T/vf_norm

vf = np.zeros((3))
wf = np.zeros((3))          # angular velocity
wf[0]=0;
wf[1]=0;
wf[2]=0;

uf_T =0;

xf= np.hstack([rf, qf, vf, wf]) # Fixed point for the state

#-[0-2] Robobee params. From IROS 2015 S.Fuller "Rotating the heading angle of underactuated flapping-wing flyers by wriggle-steering"

m   = 81        # 81 mg
Ixx = 14.2*1e-3      # 14.2 mg m^2
Iyy = 13.4*1e-3      # 13.4 mg m^2
Izz = 4.5*1e-3       # 4.5  mg m^2
g   = 9.80       # 9.8*10^2 m/s^2

input_max = 1000  # N m  

I = np.zeros((3,3))
I[0,0] = Ixx
I[1,1] = Iyy
I[2,2] = Izz

robobee_plant = RobobeePlant(m, I)

builder = DiagramBuilder()
context = robobee_plant.CreateDefaultContext()

dircol = DirectCollocation(robobee_plant, context, num_time_samples=31,
                           minimum_timestep=0.02, maximum_timestep=0.5)

dircol.AddEqualTimeIntervalsConstraints()

# Add input limits.
# torque_limit = 20.0  # mN*m.
u = dircol.input()
# dircol.AddConstraintToAllKnotPoints(-torque_limit <= u[0])
# dircol.AddConstraintToAllKnotPoints(u[0] <= torque_limit)

# initial_state = np.zeros(13)
# initial_state[0:2]= np.array([0,0])
# initial_state[3:7]= np.array([1,0,0,0])
dircol.AddBoundingBoxConstraint(x0, x0,
                                dircol.initial_state())
# More elegant version is blocked on drake #8315:
# dircol.AddLinearConstraint(dircol.initial_state() == initial_state)

# final_state = np.zeros(13)
# final_state[0:3]=np.array([0,0,0.3])
# final_state[3]=1.;
dircol.AddBoundingBoxConstraint(xf, xf,
                                dircol.final_state())
# dircol.AddLinearConstraint(dircol.final_state() == final_state)

R = 20  # Cost on input "effort".
dircol.AddRunningCost(np.dot(u.T,np.dot(R,u)))

# Add a final cost equal to the total duration.
# dircol.AddFinalCost(dircol.time())

initial_x_trajectory = \
    PiecewisePolynomial.FirstOrderHold([0., 10.],
                                       np.column_stack((x0,
                                                        x0)))
dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

tol = 1e-7;
dircol.SetSolverOption(mp.SolverType.kIpopt,"tol", tol);
dircol.SetSolverOption(mp.SolverType.kIpopt,"constr_viol_tol", tol);
dircol.SetSolverOption(mp.SolverType.kIpopt,"acceptable_tol", tol);
dircol.SetSolverOption(mp.SolverType.kIpopt,"acceptable_constr_viol_tol", tol);

dircol.SetSolverOption(mp.SolverType.kIpopt, "print_level", 5) # CAUTION: Assuming that solver used Ipopt


result = dircol.Solve()
assert(result == SolutionResult.kSolutionFound)

x_trajectory = dircol.ReconstructStateTrajectory()
u_trajectory = dircol.ReconstructInputTrajectory()

times = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), 100)
# print("times:",times.shape)

x_lookup = np.vectorize(x_trajectory.value)
u_lookup = np.vectorize(u_trajectory.value)
u_values = np.zeros((4,100))
x_values = np.zeros((13,100))

for i in range(0,100):
    u_values[:,i] = np.reshape(u_lookup(times[i]), (4))
    x_values[:,i] = np.reshape(x_lookup(times[i]), (13))
# print("u_values:", u_values)


# state_source = builder.AddSystem(x_trajectory);


# # Rigidbody_selector = builder.AddSystem(RigidBodySelection())


# # Drake visualization
# print("1. Connecting plant output to DrakeVisualizer\n")
    
# rtree = RigidBodyTree(FindResourceOrThrow("drake/examples/robobee/robobee.urdf"), FloatingBaseType.kQuaternion)
# lcm = DrakeLcm()
# # visualizer = builder.AddSystem(DrakeVisualizer(tree=rtree,
# #        lcm=lcm, enable_playback=True))
    
# # builder.Connect(robobee_plant.get_output_port(0),Rigidbody_selector.get_input_port(0))      
# # builder.Connect(Rigidbody_selector.get_output_port(0), visualizer.get_input_port(0))
    
# print("4. Building diagram\n")
    

# publisher = builder.AddSystem(rtree, lcm);
# publisher.set_publish_period(1.0 / 60.0);

# builder.Connect(state_source.get_output_port(),
#                   publisher.get_input_port(0));

# diagram = builder.Build()
# # Set the initial conditions for the simulation.
# context = diagram.CreateDefaultContext()
# # state = context.get_mutable_continuous_state_vector()
# # state.SetFromVector(x0)

# # # Create the simulator.
# # print("5. Create simulation\n")
    
# simulator = Simulator(diagram, context)
# simulator.set_target_realtime_rate(1)
# simulator.Initialize()
#     # # simulator.set_publish_every_time_step(False)

#     # simulator.get_integrator().set_fixed_step_mode(False)
#     # simulator.get_integrator().set_maximum_step_size(0.005)

#     # Simulate for the requested duration.
# print("Ending time:", x_trajectory.end_time())

# simulator.StepTo(duration)


# Ploting the result

num_iteration = np.size(x_values,1)
num_state=np.size(x_values,0)

print("num_iteration,:", num_iteration)

rpy = np.zeros((3,num_iteration)) # Convert to Euler Angle
ubar = np.zeros((4,num_iteration)) # Convert to Euler Angle
u = np.zeros((4,num_iteration)) # Convert to Euler Angle

for j in range(0,num_iteration):

    # ubar[:,j]=test_Feedback_Linearization_controller_BS(state_out[:,j])
    ubar[:,j]=u_values[:,j]
    q_temp =x_values[3:7,j]
    print(np.dot(q_temp.T, q_temp))
    q_temp = q_temp/np.sqrt(np.dot(q_temp.T, q_temp))
    quat_temp = Quaternion(q_temp)    # Quaternion
    R = RotationMatrix(quat_temp)
    rpy[:,j]=RollPitchYaw(R).vector()
    u[:,j]=ubar[:,j]
    # u[0,j]=x_values[7,j] # Control
    


# Visualize state and input traces
# print("times",state_log.data()[1,:])RollPitchYaw

if show_flag_q==1:
    plt.clf()
    fig = plt.figure(1).set_size_inches(6, 6)
    for i in range(0,3):
        # print("i:%d" %i)
        plt.subplot(3, 1, i+1)
        # print("test:", num_state)
        plt.plot(times, x_values[i, :])
        plt.grid(True)
        if i==0:
            plt.ylabel("x")
        elif i==1:
            plt.ylabel("y")
        elif i==2:
            plt.ylabel("z")
        

    ####- Plot Euler angle
    fig = plt.figure(2).set_size_inches(6, 6)
    for i in range(0,3):
        # print("i:%d" %i)
        plt.subplot(3, 1, i+1)
        # print("test:", num_state)
        plt.plot(times, rpy[i,:])
        plt.grid(True)
        j=i+3
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("roll")
        elif i==1:
            plt.ylabel("pitch")
        elif i==2:
            plt.ylabel("yaw")


####- Plot Quaternion

# fig = plt.figure(2).set_size_inches(6, 6)
# for i in range(0,4):
#   # print("i:%d" %i)
#     plt.subplot(4, 1, i+1)
#     # print("test:", num_state)
#     plt.plot(state_log.sample_times(), state_log.data()[i+3, :])
#     plt.grid(True)
#     j=i+3
#     # plt.ylabel("x[%d]" % j)
#     if i==0:
#       plt.ylabel("q0")
#     elif i==1:
#       plt.ylabel("q1")
#     elif i==2:
#       plt.ylabel("q2")
#     elif i==3:
#       plt.ylabel("q3")

if show_flag_qd==1:
    fig = plt.figure(3).set_size_inches(6, 6)
    for i in range(0,3):
        # print("i:%d" %i)
        plt.subplot(3, 1, i+1)
        # print("test:", num_state)
        plt.plot(times, x_values[i+7, :])
        plt.grid(True)
        j=i+7
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("vx")
        elif i==1:
            plt.ylabel("vy")
        elif i==2:
            plt.ylabel("vz")
    fig = plt.figure(4).set_size_inches(6, 6)
    for i in range(0,3):
        # print("i:%d" %i)
        plt.subplot(3, 1, i+1)
        # print("test:", num_state)
        plt.plot(times, x_values[i+10, :])
        plt.grid(True)
        j=i+10
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("wx")
        elif i==1:
            plt.ylabel("wy")
        elif i==2:
            plt.ylabel("wz")

if show_flag_control==1:
    fig = plt.figure(5).set_size_inches(6, 6)
    for i in range(0,4):
        # print("i:%d" %i)
        plt.subplot(4, 1, i+1)
        # print("test:", num_state)
        plt.plot(times, u[i,:])
        plt.grid(True)
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("Thrust")
        elif i==1:
            plt.ylabel("tau_r")
        elif i==2:
            plt.ylabel("tau_p")
        elif i==3:
            plt.ylabel("tau_y")
# plt.subplot(num_state, 1, num_state)
# plt.plot(input_log.sample_times(), input_log.data()[0, :])
# plt.ylabel("u[0]")
# plt.xlabel("t")

plt.grid(True)
plt.show()
