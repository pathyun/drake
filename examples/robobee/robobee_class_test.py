##########################333
#
# Test script for LQR controller for robobee using pydr
#
#
################################
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

from robobee_plant_example import *

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    SignalLogger,
    VectorSystem,
    RotationMatrix,
    Quaternion,
    LinearQuadraticRegulator,
    RollPitchYaw
    )

# Make numpy printing prettier
np.set_printoptions(precision=3, suppress=True)


#-[0] Initial condition

r0 = np.array([0,0,0])
q0 = np.zeros((4)) # quaternion setup
theta0 = math.pi/2;  # angle of rotation
v0=np.array([1.,0.,1.])
q0[0]=math.cos(theta0/2) #q0
# print("q:",q[0])
v0_norm=np.sqrt((np.dot(v0,v0))) # axis of rotation
v0_normalized =v0.T/v0_norm
# print("vnorm:", v_norm)
# print("vnormalized", np.dot(v_normalized,v_normalized.T))
q0[1:4]=math.sin(theta0/2)*v0.T/v0_norm

v0 = np.zeros((3))
w0 = np.zeros((3))  		# angular velocity
w0[0]=0;
w0[1]=2;
w0[2]=1;

#-[0-1] Stack up the state in R^13
print("r:", r0.shape, "q",q0.shape,"v",v0.shape,"w", w0.shape)
x0= np.hstack([r0, q0, v0, w0])

# Robobee params. You're free to play with these,
# of course, but we'll be expecting you to use the
# default values when answering questions, where
# varying these values might make a difference.
m = 0.5 # Default 1
Ixx = 0.0023 # Default 1
Iyy = 0.0023 # Default 2
Izz = 0.0040 # Default 2
g = 9.8  # Default 10

input_max = 1000
robobee_plant = RobobeePlant(
    m = m, Ixx = Ixx, Iyy = Iyy, Izz = Izz, 
    g = g, input_max = input_max)

'''
Code submission for 3.1: 
Edit this method in `inertial_wheel_pendulum.py`
and ensure it produces reasonable A and B
'''

#-[1] Fixed point for Linearization



F_T = g;
tau = np.array([0,0,0]) 

rf = np.array([0,0,0.5])
qf = np.zeros((4)) # quaternion setup
thetaf = math.pi/4;  # angle of rotation
#-test of theta=0
# thetaf =0;
vf=np.array([0.,0.,1.])
qf[0]=math.cos(thetaf/2) #q0
# print("q:",q[0])
vf_norm=np.sqrt((np.dot(vf,vf))) # axis of rotation
vf_normalized =vf.T/vf_norm
# print("vnorm:", v_norm)
# print("vnormalized", np.dot(v_normalized,v_normalized.T))
qf[1:4]=math.sin(thetaf/2)*vf.T/vf_norm

vf = np.zeros((3))
wf = np.zeros((3))  		# angular velocity
wf[0]=0;
wf[1]=0;
wf[2]=0;


xf= np.hstack([rf, qf, vf, wf]) # Fixed point for the state
print("xf:", xf)
uf = np.hstack([F_T,tau])  # Hovering
print("uf:",uf)

xstackf=robobee_plant.evaluate_f(uf,xf)
print("xstack",xstackf)


#-[2] Linearization and get (K,S) for LQR

A, B =robobee_plant.GetLinearizedDynamics(uf, xf)

print("A:", A, "B:", B)

Q = 1*np.eye(13)
Q[0:3,0:3] = 10*np.eye(3)
Q[10:13,10:13] = 1*np.eye(3)

R = np.eye(4)
N = np.zeros((13,4))

M_lqr = solve_continuous_are(A,B,Q,R)
# print("M_lqr:", M_lqr)
K_py = np.dot(np.dot(np.linalg.inv(R),B.T),M_lqr) 
print("K_py",K_py)
print("K_py size:", K_py.shape)


K_, S_ = LinearQuadraticRegulator(A,B,Q,R)

# print("K_:", K_, "S_:", S_)

ControllabilityMatrix = np.zeros((13, 4*13))
for i in range(0,13):
	if i==0:
		TestAB = B
		ControllabilityMatrix= TestAB
		print("ControllabilityMatrix:", ControllabilityMatrix)
	else:
		TestAB = np.dot(A,TestAB)
		ControllabilityMatrix =np.hstack([ControllabilityMatrix, TestAB])
# print("ContrbM: ", ControllabilityMatrix)
# print("Contrb size:", ControllabilityMatrix.shape)
rankContrb = np.linalg.matrix_rank(ControllabilityMatrix)
print("Contrb rank:", rankContrb)


def test_controller(x):
    # This should return a 1x1 u that is bounded
    # between -input_max and input_max.
    # Remember to wrap the angular values back to
    # [-pi, pi].
    u = np.zeros(4)
    global g, xf, uf, K
    
    u[0]=g+0.01
    u[1] =0;
    u[3]=-2
    ''' 
    Code submission for 3.3: fill in the code below
    to use your computed LQR controller (i.e. gain matrix
    K) to stabilize the robot by setting u appropriately.
    '''
    
    return u

def test_LQRcontroller(x):
    # This should return a 1x1 u that is bounded
    # between -input_max and input_max.
    # Remember to wrap the angular values back to
    # [-pi, pi].
    global g, xf, uf, K_, K_py
    q0=x[3]
    q1=x[4]
    q2=x[5]
    q3=x[6]
    quat_vec = np.vstack([q0,q1,q2,q3])
    q_norm=np.sqrt(np.dot(quat_vec.T,quat_vec))
    # print("x:",x)
    # print("q_norm: ",q_norm)
    u = np.zeros(4)

    u = uf - np.dot(K_py,x-xf)
    # print("\n######################")
    # print("\n u:", u)
    # print("\n####################33")
    return u 

# Test LQR


# Run forward simulation from the specified initial condition
duration =5.

input_log, state_log = \
    RunSimulation(robobee_plant,
              test_LQRcontroller,
              x0=x0,
              duration=duration)



num_iteration = np.size(state_log.data(),1)
num_state=np.size(state_log.data(),0)

state_out =state_log.data();

# print("num_iteration,:", num_iteration)
# print("state_out dimension:,:", state_out.shape)
rpy = np.zeros((3,num_iteration)) # Convert to Euler Angle
u_lqr = np.zeros((4,num_iteration)) # Convert to Euler Angle

for j in range(0,num_iteration):
	q_temp =state_out[3:7,j]
	quat_temp = Quaternion(q_temp)    # Quaternion
	R = RotationMatrix(quat_temp)
	rpy[:,j]=RollPitchYaw(R).vector()
	u_lqr[:,j]=test_LQRcontroller(state_out[:,j]) # Control
	


# Visualize state and input traces
# print("times",state_log.data()[1,:])RollPitchYaw
plt.clf()
fig = plt.figure(1).set_size_inches(6, 6)
for i in range(0,3):
	# print("i:%d" %i)
    plt.subplot(3, 1, i+1)
    # print("test:", num_state)
    plt.plot(state_log.sample_times(), state_log.data()[i, :])
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
    plt.plot(state_log.sample_times(), rpy[i,:])
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
# 	# print("i:%d" %i)
#     plt.subplot(4, 1, i+1)
#     # print("test:", num_state)
#     plt.plot(state_log.sample_times(), state_log.data()[i+3, :])
#     plt.grid(True)
#     j=i+3
#     # plt.ylabel("x[%d]" % j)
#     if i==0:
#     	plt.ylabel("q0")
#     elif i==1:
#     	plt.ylabel("q1")
#     elif i==2:
#     	plt.ylabel("q2")
#     elif i==3:
#     	plt.ylabel("q3")
fig = plt.figure(3).set_size_inches(6, 6)
for i in range(0,3):
	# print("i:%d" %i)
    plt.subplot(3, 1, i+1)
    # print("test:", num_state)
    plt.plot(state_log.sample_times(), state_log.data()[i+7, :])
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
    plt.plot(state_log.sample_times(), state_log.data()[i+10, :])
    plt.grid(True)
    j=i+10
    # plt.ylabel("x[%d]" % j)
    if i==0:
    	plt.ylabel("wx")
    elif i==1:
    	plt.ylabel("wy")
    elif i==2:
    	plt.ylabel("wz")
fig = plt.figure(5).set_size_inches(6, 6)
for i in range(0,4):
	# print("i:%d" %i)
    plt.subplot(4, 1, i+1)
    # print("test:", num_state)
    plt.plot(state_log.sample_times(), u_lqr[i,:])
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
