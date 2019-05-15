import math
import numpy as np
import matplotlib.pyplot as plt

from robobee_plant_example import *

from pydrake.all import (DirectCollocation, FloatingBaseType,
                         PiecewisePolynomial, RigidBodyTree, RigidBodyPlant,
                         SolutionResult, Simulator,
    SignalLogger,
    VectorSystem,
    RotationMatrix,
    Quaternion,
    RollPitchYaw)

from pydrake.examples.acrobot import AcrobotPlant

# Make numpy printing prettier
np.set_printoptions(precision=3, suppress=True)

#-[0.0] Show figure flag

show_flag_qd = 0;
show_flag_q=1;
show_flag_control =1;

#-[0] Initial condition

r0 = np.array([0,0,0])
q0 = np.zeros((4)) # quaternion setup
theta0 = math.pi/2;  # angle of rotation
v0=np.array([1.,-1.,1.])
q0[0]=math.cos(theta0/2) #q0
# print("q:",q[0])
v0_norm=np.sqrt((np.dot(v0,v0))) # axis of rotation
v0_normalized =v0.T/v0_norm
# print("vnorm:", v_norm)
# print("vnormalized", np.dot(v_normalized,v_normalized.T))
q0[1:4]=math.sin(theta0/2)*v0.T/v0_norm

v0 = np.zeros((3))
w0 = np.zeros((3))          # angular velocity
w0[0]=-1;
w0[1]=0;
w0[2]=1;

#-[0-1] Stack up the state in R^13
# print("r:", r0.shape, "q",q0.shape,"v",v0.shape,"w", w0.shape)
x0= np.hstack([r0, q0, v0, w0])

#-[0-2] Robobee params. From IROS 2015 S.Fuller "Rotating the heading angle of underactuated flapping-wing flyers by wriggle-steering"

m   = 81        # 81 mg
Ixx = 14.2*1e-3      # 14.2 mg m^2
Iyy = 13.4*1e-3      # 13.4 mg m^2
Izz = 4.5*1e-3       # 4.5  mg m^2
g   = 9.80       # 9.8*10^2 m/s^2

input_max = 1000  # N m  
robobee_plant = RobobeePlant(
    m = m, Ixx = Ixx, Iyy = Iyy, Izz = Izz, 
    g = g, input_max = input_max)


context = robobee_plant.CreateDefaultContext()

dircol = DirectCollocation(robobee_plant, context, num_time_samples=21,
                           minimum_timestep=0.05, maximum_timestep=0.2)

dircol.AddEqualTimeIntervalsConstraints()

# # Add input limits.
# torque_limit = 8.0  # N*m.
# u = dircol.input()
# dircol.AddConstraintToAllKnotPoints(-torque_limit <= u[0])
# dircol.AddConstraintToAllKnotPoints(u[0] <= torque_limit)

# initial_state = (0., 0., 0., 0.)
# dircol.AddBoundingBoxConstraint(initial_state, initial_state,
#                                 dircol.initial_state())
# # More elegant version is blocked on drake #8315:
# # dircol.AddLinearConstraint(dircol.initial_state() == initial_state)

# final_state = (math.pi, 0., 0., 0.)
# dircol.AddBoundingBoxConstraint(final_state, final_state,
#                                 dircol.final_state())
# # dircol.AddLinearConstraint(dircol.final_state() == final_state)

# R = 10  # Cost on input "effort".
# dircol.AddRunningCost(R*u[0]**2)

# # Add a final cost equal to the total duration.
# dircol.AddFinalCost(dircol.time())

# initial_x_trajectory = \
#     PiecewisePolynomial.FirstOrderHold([0., 4.],
#                                        np.column_stack((initial_state,
#                                                         final_state)))
# dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

# result = dircol.Solve()
# assert(result == SolutionResult.kSolutionFound)

# x_trajectory = dircol.ReconstructStateTrajectory()

# tree = RigidBodyTree(FindResource("acrobot/acrobot.urdf"),
#                      FloatingBaseType.kFixed)
# vis = PlanarRigidBodyVisualizer(tree, xlim=[-4., 4.], ylim=[-4., 4.])
# ani = vis.animate(x_trajectory, repeat=True)

# u_trajectory = dircol.ReconstructInputTrajectory()
# times = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), 100)
# u_lookup = np.vectorize(u_trajectory.value)
# u_values = u_lookup(times)

# plt.figure()
# plt.plot(times, u_values)
# plt.xlabel('time (seconds)')
# plt.ylabel('force (Newtons)')

# plt.show()