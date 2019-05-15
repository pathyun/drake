##########################333
#
# Test script for feedback_linearizaiton with BS method 
# Script originated from robobee_class_feedback_linearization.py
# 
#  Desired output y = [y_1; y_2]\in\mathbb{R}^4:
#
#  Call RobobeePlantBS : 15 state 4 input
#  \dot{\Xi_1} =\Xi_2
#  \dot{\Xi_2} = u
#
################################
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

from robobee_plantBS_example import *

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
np.set_printoptions(precision=4, suppress=True)

#-[0.0] Show figure flag

show_flag_qd = 1;
show_flag_q=1;
show_flag_control =1;

#-[0] Intialization
#-[0-1] Robobee params. From IROS 2015 S.Fuller "Rotating the heading angle of underactuated flapping-wing flyers by wriggle-steering"

m   = 81        # 81 mg
Ixx = 14.2*1e-2      # 14.2 mg m^2
Iyy = 13.4*1e-2      # 13.4 mg m^2
Izz = 4.5*1e-2       # 4.5  mg m^2
g   = 9.8       # 9.8*10^2 m/s^2

#-[0] Initial condition

r0 = np.array([0,0,0])
q0 = np.zeros((4)) # quaternion setup
theta0 = math.pi/2;  # angle of rotation
v0_q=np.array([1.,-1.,1.])
q0[0]=math.cos(theta0/2) #q0
# print("q:",q[0])
v0_norm=np.sqrt((np.dot(v0_q,v0_q))) # axis of rotation
v0_normalized =v0_q.T/v0_norm
# print("vnorm:", v_norm)
# print("vnormalized", np.dot(v_normalized,v_normalized.T))
q0[1:4]=math.sin(theta0/2)*v0_q.T/v0_norm

xi10 = g;
v0 = np.zeros((3))
w0 = np.zeros((3))          # angular velocity
w0[0]=0;
w0[1]=-1;
w0[2]=2;

xi20 =0;
#-[0-1] Stack up the state in R^13
# print("r:", r0.shape, "q",q0.shape,"v",v0.shape,"w", w0.shape)
x0= np.hstack([r0, q0, xi10, v0, w0, xi20])



input_max = 10000000  # N m  
robobee_plantBS = RobobeePlantBS(
    m = m, Ixx = Ixx, Iyy = Iyy, Izz = Izz, 
    g = g, input_max = input_max)


#-[1] Fixed point for Linearization

F_T = g;
tau = np.array([0,0,0]) 

rf = np.array([0,0,0.30])
qf = np.zeros((4)) # quaternion setup
thetaf = math.pi/4;  # angle of rotation

vf=np.array([0.,0.,1.])
qf[0]=math.cos(thetaf/2) #q0

xi1f=F_T;
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

xi2f=0;

# u =0 Backstepoped input

uf_T =0;

xf= np.hstack([rf, qf, xi1f, vf, wf, xi2f]) # Fixed point for the state
print("xf:", xf)
uf = np.hstack([uf_T,tau])  # Hovering
print("uf:",uf)

xstackf=robobee_plantBS.evaluate_f(uf,xf)
xstackf_norm = np.dot(xstackf,xstackf)
if xstackf_norm<1e-6:
    print("\n\n1. Set point is a fixed point")
else:
    print("\n\n1. Set point is not a fixed point")

#-[2] Output dynamics Lyapunov function
# 
#  V(eta)= 1/2eta^T M_out eta
#  for deta = Aout eta + Bout u
#
#  Solving CARE to get output feedback controller
#
#  dVdt < -eta^T Q eta< 0  where Q is positive definite

Q = 1*np.eye(13)
Q[0:3,0:3] = 100*np.eye(3)
Q[12,12] = 1
# Q[10:13,10:13] = 1*np.eye(3)

R = 1*np.eye(4)

Aout = np.zeros((13,13))
Aout[0:3,3:6]=np.eye(3)
Aout[3:6,6:9]=np.eye(3)
Aout[6:9,9:12]=np.eye(3)
    
Bout = np.zeros((13,4))
Bout[9:13,0:4]= np.eye(4)
# print("A:", A, "B:", B)
Mout = solve_continuous_are(Aout,Bout,Q,R)
# print("M_out:", Mout)

# #-[2] Linearization and get (K,S) for LQR

# A, B =robobee_plantBS.GetLinearizedDynamics(uf, xf)

# print("A:", A, "B:", B)

# ControllabilityMatrix = np.zeros((15, 4*15))
# for i in range(0,15):
#     if i==0:
#         TestAB = B
#         ControllabilityMatrix= TestAB
#         # print("ControllabilityMatrix:", ControllabilityMatrix)
#     else:
#         TestAB = np.dot(A,TestAB)
#         ControllabilityMatrix =np.hstack([ControllabilityMatrix, TestAB])
# print("ContrbM: ", ControllabilityMatrix)
# print("Contrb size:", ControllabilityMatrix.shape)
# rankContrb = np.linalg.matrix_rank(ControllabilityMatrix)
# print("\n Contrb rank: ", rankContrb)

# #-[2-1] LQR gain matrix by solving CARE

# Q = 1*np.eye(15)
# Q[0:3,0:3] = 10*np.eye(3)
# # Q[10:13,10:13] = 1*np.eye(3)

# R = np.eye(4)
# N = np.zeros((15,4))

# M_lqr = solve_continuous_are(A,B,Q,R)
# # print("M_lqr:", M_lqr)
# K_py = np.dot(np.dot(np.linalg.inv(R),B.T),M_lqr) 
# print("K_py",K_py)
# # print("K_py size:", K_py.shape)




# K_, S_ = LinearQuadraticRegulator(A,B,Q,R)

# print("K_:", K_, "S_:", S_)




def test_controller(x):
    # This should return a 4x1 u that is bounded
    # between -input_max and input_max.
    # Remember to wrap the angular values back to
    # [-pi, pi].
    u = np.zeros(4)
    global g, xf, uf, K
    
    u[0]=1
    u[1] =0;
    u[3]=-2

    return u

def test_LQRcontroller(x):
    # This should return a 4x1 u that is bounded
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

def test_Feedback_Linearization_controller_BS(x):
    # Output feedback linearization 2
    #
    # y1= ||r-rf||^2/2
    # y2= wx
    # y3= wy
    # y4= wz
    #
    # Analysis: The zero dynamics is unstable.
    global g, xf, Aout, Bout, Mout

    e1=np.array([1,0,0]) # e3 elementary vector
    e2=np.array([0,1,0]) # e3 elementary vector
    e3=np.array([0,0,1]) # e3 elementary vector

    # epsilonn= 1e-0
    # alpha = 100;
    # kp1234=alpha*1/math.pow(epsilonn,4) # gain for y
    # kd1=4/math.pow(epsilonn,3) # gain for y^(1)
    # kd2=12/math.pow(epsilonn,2) # gain for y^(2)
    # kd3=4/math.pow(epsilonn,1)  # gain for y^(3)

    # kp5= 10;                    # gain for y5 

    q=np.zeros(7)
    qd=np.zeros(6)
    q=x[0:8]
    qd=x[8:15]

    print("qnorm:", np.dot(q[3:7],q[3:7]))

    q0=q[3]
    q1=q[4]
    q2=q[5]
    q3=q[6]
    xi1=q[7]

    v=qd[0:3]
    w=qd[3:6]
    xi2=qd[6]
    

    xd=xf[0]
    yd=xf[1]
    zd=xf[2]
    wd=xf[11:14]

    # Useful vectors and matrices
    
    (Rq, Eq, wIw, I_inv)=robobee_plantBS.GetManipulatorDynamics(q, qd)

    F1q = np.zeros((3,4))
    F1q[0,:] = np.array([   q2,    q3,    q0,    q1])
    F1q[1,:] = np.array([-1*q1, -1*q0,    q3,    q2])
    F1q[2,:] = np.array([   q0, -1*q1, -1*q2,    q3])
    
    Rqe3 = np.dot(Rq,e3)
    Rqe3_hat = np.zeros((3,3))
    Rqe3_hat[0,:] = np.array([        0,   -Rqe3[2],     Rqe3[1] ])
    Rqe3_hat[1,:] = np.array([  Rqe3[2],          0,    -Rqe3[0] ])
    Rqe3_hat[2,:] = np.array([ -Rqe3[1],    Rqe3[0],           0 ])

    w_hat = np.zeros((3,3))
    w_hat[0,:] = np.array([     0,   -w[2],     w[1] ])
    w_hat[1,:] = np.array([  w[2],       0,    -w[0] ])
    w_hat[2,:] = np.array([ -w[1],    w[0],        0 ])

    #- Checking the derivation

    # print("F1qEqT-(-Rqe3_hat)",np.dot(F1q,Eq.T)-(-Rqe3_hat))
    # Rqe3_cal = np.zeros(3)
    # Rqe3_cal[0] = 2*(q3*q1+q0*q2)
    # Rqe3_cal[1] = 2*(q3*q2-q0*q1)
    # Rqe3_cal[2] = (q0*q0+q3*q3-q1*q1-q2*q2)

    # print("Rqe3 - Rqe3_cal", Rqe3-Rqe3_cal)

    # Four output
    y1 = x[0]-xd
    y2 = x[1]-yd
    y3 = x[2]-zd
    y4 = np.dot(w-wd,w-wd)/2 +np.dot(v[2],v[2])/2

    eta1 = np.zeros(3)
    eta1 = np.array([y1,y2,y3])
    eta5=y4
    
    # First derivative of first three output
    eta2 = np.zeros(3)
    eta2 = v
    dy1 = eta2[0]
    dy2 = eta2[1]
    dy3 = eta2[2]    

    # Second derivative of first three output
    eta3 = np.zeros(3)
    eta3 = -g*e3+Rqe3*xi1
    ddy1 = eta3[0]
    ddy2 = eta3[1]
    ddy3 = eta3[2]

    # Third derivative of first three output
    eta4 = np.zeros(3)
    eta4 = Rqe3*xi2+np.dot(np.dot(F1q,Eq.T),w)*xi1
    dddy1 = eta4[0]
    dddy2 = eta4[1]
    dddy3 = eta4[2]

    # Fourth derivative of first three output
    B_qw_temp = np.zeros(3)
    B_qw_temp = xi1*(-np.dot(w_hat,np.dot(Rqe3_hat,w))+np.dot(Rqe3_hat,np.dot(I_inv,wIw))) # np.dot(I_inv,wIw)*xi1-2*w*xi2
    B_qw      = B_qw_temp+xi2*(-2*np.dot(Rqe3_hat,w))    #np.dot(Rqe3_hat,B_qw_temp)
    # B_qw = B_qw_temp - np.dot(w_hat,np.dot(Rqe3_hat,w))*xi1

    wIwIw = -np.dot(w-wd,np.dot(I_inv,wIw)) -g

    # Decoupling matrix A(x)\in\mathbb{R}^4

    A_fl = np.zeros((4,4))
    A_fl[0:3,0] = Rqe3
    A_fl[0:3,1:4] = -np.dot(Rqe3_hat,I_inv)*xi1
    A_fl[3,1:4]=np.dot(w-wd,I_inv)
    A_fl[3,0] = np.dot(e3,Rqe3)

    A_fl_inv = np.linalg.inv(A_fl)
    A_fl_det = np.linalg.det(A_fl)
    # print("I_inv:", I_inv)
    # print("A_fl:", A_fl)
    print("A_fl_det:", A_fl_det)

    # Output dyamics

    eta = np.hstack([eta1, eta2, eta3, eta4, eta5])

    # Full feedback controller

    U_temp = np.zeros(4)
    U_temp[0:3]  = B_qw
    U_temp[3]    = wIwIw

    # print("x:", x)
    print("eta_norm:", np.dot(eta, np.dot(Mout,eta) ))
    # print("eta1:", eta1)
    # print("eta2:", eta2)
    # print("eta3:", eta3)
    # print("eta4:", eta4)
    # print("eta5:", eta5)
            
    mu = np.zeros(4)
    k = np.zeros((4,13))
    k = np.dot(Bout.T,Mout)
    # print("k:", k)

    mu = -np.dot(k,eta)

    v=-U_temp+mu
    
    U_fl = np.dot(A_fl_inv,v)       # Feedback controller

    # U_fl_zero = np.dot(A_fl_inv,-U_temp)

    # dx = robobee_plantBS.evaluate_f(U_fl,x)
    
    # print("dx", dx)

    u = np.zeros(4)

    u[0] = U_fl[0]
    u[1:4] = U_fl[1:4]

    # print("\n######################")
    # # print("qe3:", A_fl[0,0])
    # print("u:", u)
    # print("\n####################33")
    
    # deta4 = B_qw+Rqe3*U_fl_zero[0]+np.dot(-np.dot(Rqe3_hat,I_inv),U_fl_zero[1:4])*xi1
    # print("deta4:",deta4)

    return u 


# Test LQR


# Run forward simulation from the specified initial condition
duration =15.

input_log, state_log = \
    RunSimulation(robobee_plantBS,
              test_Feedback_Linearization_controller_BS,
              x0=x0,
              duration=duration)



num_iteration = np.size(state_log.data(),1)
num_state=np.size(state_log.data(),0)

state_out =state_log.data();

# print("num_iteration,:", num_iteration)
# print("state_out dimension:,:", state_out.shape)
rpy = np.zeros((3,num_iteration)) # Convert to Euler Angle
ubar = np.zeros((4,num_iteration)) # Convert to Euler Angle
u = np.zeros((4,num_iteration)) # Convert to Euler Angle

for j in range(0,num_iteration):

    ubar[:,j]=test_Feedback_Linearization_controller_BS(state_out[:,j])
    q_temp =state_out[3:7,j]
    quat_temp = Quaternion(q_temp)    # Quaternion
    R = RotationMatrix(quat_temp)
    rpy[:,j]=RollPitchYaw(R).vector()
    u[:,j]=ubar[:,j]
    u[0,j]=state_out[7,j] # Control
    


# Visualize state and input traces
# print("times",state_log.data()[1,:])RollPitchYaw

if show_flag_q==1:
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

if show_flag_control==1:
    fig = plt.figure(5).set_size_inches(6, 6)
    for i in range(0,4):
        # print("i:%d" %i)
        plt.subplot(4, 1, i+1)
        # print("test:", num_state)
        plt.plot(state_log.sample_times(), u[i,:])
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
