##########################333
#
# Feedback_linearizaiton with Back Stepping method with quaternion
# 
# Script originated from robobee_class_feedback_linearization.py
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

# Pick mg unit
m   = 81        # 81 mg
Ixx = 1.42*1e-3      # 1.42 x 10^-3  mg m^2                     | 1.42 x 10 mg cm^2
Iyy = 1.34*1e-3      # 1.34 x 10^-3  mg m^2                     | 1.34 x 10 mg cm^2             
Izz = 0.45*1e-3      # 0.45 x 10^-3  mg m^2                     | 0.45 x 10 mg cm^2
g   = 9.80*1e0       # 9.8 m/s^2    use 1 \tau = 0.1s           | 9.8  x 1 cm/s^2

#-[0] Initial condition

r0 = np.array([0,0,0.05])
q0 = np.zeros((4)) # quaternion setup
theta0 = math.pi/4;  # angle of rotation
# v0_q=np.array([1.,0.,1.])
v0_q=np.array([0.,0.,1.])
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
w0[0]=-1 # 0.1 #0;
w0[1]=0 # 0  #-0.1;
w0[2]=1 # 0.1 # 0.2;

xi20 =0;
#-[0-1] Stack up the state in R^13
# print("r:", r0.shape, "q",q0.shape,"v",v0.shape,"w", w0.shape)
x0= np.hstack([r0, q0, xi10, v0, w0, xi20])
print("x0:", x0)


input_max = 1e12  # N m  
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

#- Tracking parameter
time_gain = 1; # 10\tau = 1s

T_period = 5*time_gain # 
w_freq = 2*math.pi/T_period
radius = .5
#-[2] Output dynamics Lyapunov function
# 
#  V(eta)= 1/2eta^T M_out eta
#  for deta = Aout eta + Bout u
#
#  Solving CARE to get output feedback controller
#
#  dVdt < -eta^T Q eta< 0  where Q is positive definite

Q = 1e0*np.eye(14)
Q[0:3,0:3] = 1e4*np.eye(3) # eta1
Q[3:6,3:6] = 1e3*np.eye(3) # eta2
Q[6:9,6:9] = 1e2*np.eye(3) # eta3
Q[9,9] = 1e1     # eta5
Q[10:13,10:13] = 1e1*np.eye(3) #eta 4


# Q[9,9] = 1000
# Q[10:13,10:13] = 1*np.eye(3)

R = 1e0*np.eye(4)
R[0,]

Aout = np.zeros((14,14))
Aout[0:3,3:6]=np.eye(3)
Aout[3:6,6:9]=np.eye(3)
Aout[6:9,10:13]=np.eye(3)
Aout[9,13]=1
    
Bout = np.zeros((14,4))
Bout[10:14,0:4]= np.eye(4)

print("A:", Aout, "B:", Bout)
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




def test_controller(x, t):
    # This should return a 4x1 u that is bounded
    # between -input_max and input_max.
    # Remember to wrap the angular values back to
    # [-pi, pi].
    u = np.zeros(4)
    global g, xf, uf, K
    
    u[0]=1
    u[1]=0
    u[2]=-0.001
    u[3]=0

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

def test_Feedback_Linearization_controller_BS(x,t):
    # Output feedback linearization 2
    #
    # y1= ||r-rf||^2/2
    # y2= wx
    # y3= wy
    # y4= wz
    #
    # Analysis: The zero dynamics is unstable.
    
    global g, xf, Aout, Bout, Mout, T_period, w_freq, radius

    print("t:", t)
    
    # # # # Example 1 : Circle

    # x_f = radius*math.cos(w_freq*t)
    # y_f = radius*math.sin(w_freq*t)
    # # print("x_f:",x_f)
    # # print("y_f:",y_f)
    # dx_f = -radius*math.pow(w_freq,1)*math.sin(w_freq*t)
    # dy_f = radius*math.pow(w_freq,1)*math.cos(w_freq*t)
    # ddx_f = -radius*math.pow(w_freq,2)*math.cos(w_freq*t)
    # ddy_f = -radius*math.pow(w_freq,2)*math.sin(w_freq*t)
    # dddx_f = radius*math.pow(w_freq,3)*math.sin(w_freq*t)
    # dddy_f = -radius*math.pow(w_freq,3)*math.cos(w_freq*t)
    # ddddx_f = radius*math.pow(w_freq,4)*math.cos(w_freq*t)
    # ddddy_f = radius*math.pow(w_freq,4)*math.sin(w_freq*t)

    # Example 2 : Lissajous curve a=1 b=2
    ratio_ab=4./3.;
    a=3;
    b=ratio_ab*a;
    delta_lissajous = math.pi/2;

    x_f = radius*math.sin(a*w_freq*t+delta_lissajous)
    y_f = radius*math.sin(b*w_freq*t)
    print("a:", a)
    print("b:", b)
    # print("x_f:",x_f)
    # print("y_f:",y_f)
    dx_f = radius*math.pow(a*w_freq,1)*math.cos(a*w_freq*t+delta_lissajous)
    dy_f = radius*math.pow(b*w_freq,1)*math.cos(b*w_freq*t)
    ddx_f = -radius*math.pow(a*w_freq,2)*math.sin(a*w_freq*t+delta_lissajous)
    ddy_f = -radius*math.pow(b*w_freq,2)*math.sin(b*w_freq*t)
    dddx_f = -radius*math.pow(a*w_freq,3)*math.cos(a*w_freq*t+delta_lissajous)
    dddy_f = -radius*math.pow(b*w_freq,3)*math.cos(b*w_freq*t)
    ddddx_f = radius*math.pow(a*w_freq,4)*math.sin(a*w_freq*t+delta_lissajous)
    ddddy_f = radius*math.pow(b*w_freq,4)*math.sin(b*w_freq*t)


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

    Rqe1 = np.dot(Rq,e1)
    Rqe1_hat = np.zeros((3,3))
    Rqe1_hat[0,:] = np.array([        0,   -Rqe1[2],     Rqe1[1] ])
    Rqe1_hat[1,:] = np.array([  Rqe1[2],          0,    -Rqe1[0] ])
    Rqe1_hat[2,:] = np.array([ -Rqe1[1],    Rqe1[0],           0 ])

    Rqe1_x=np.dot(e2.T,Rqe1)
    Rqe1_y=np.dot(e1.T,Rqe1)

    w_hat = np.zeros((3,3))
    w_hat[0,:] = np.array([     0,   -w[2],     w[1] ])
    w_hat[1,:] = np.array([  w[2],       0,    -w[0] ])
    w_hat[2,:] = np.array([ -w[1],    w[0],        0 ])

    Rw = np.dot(Rq,w);
    Rw_hat = np.zeros((3,3))
    Rw_hat[0,:] = np.array([     0,   -Rw[2],     Rw[1] ])
    Rw_hat[1,:] = np.array([  Rw[2],       0,    -Rw[0] ])
    Rw_hat[2,:] = np.array([ -Rw[1],    Rw[0],        0 ])


    #- Checking the derivation

    # print("F1qEqT-(-Rqe3_hat)",np.dot(F1q,Eq.T)-(-Rqe3_hat))
    # Rqe3_cal = np.zeros(3)
    # Rqe3_cal[0] = 2*(q3*q1+q0*q2)
    # Rqe3_cal[1] = 2*(q3*q2-q0*q1)
    # Rqe3_cal[2] = (q0*q0+q3*q3-q1*q1-q2*q2)

    # print("Rqe3 - Rqe3_cal", Rqe3-Rqe3_cal)

    # Four output
    y1 = x[0]-x_f
    y2 = x[1]-y_f
    y3 = x[2]-zd
    y4 = math.atan2(Rqe1_x,Rqe1_y)-math.pi/8
    # print("Rqe1_x:", Rqe1_x)

    eta1 = np.zeros(3)
    eta1 = np.array([y1,y2,y3])
    eta5 = y4
    
    # print("y4", y4)
   
    # First derivative of first three output and yaw output
    eta2 = np.zeros(3)
    eta2 = v - np.array([dx_f,dy_f,0])
    dy1 = eta2[0]
    dy2 = eta2[1]
    dy3 = eta2[2]
    dy4 = 0

    x2y2 = (math.pow(Rqe1_x,2)+math.pow(Rqe1_y,2)) # x^2+y^2

    eta6_temp = np.zeros(3)     #eta6_temp = (ye2T-xe1T)/(x^2+y^2)
    eta6_temp = (Rqe1_y*e2.T-Rqe1_x*e1.T)/x2y2  
    # print("eta6_temp:", eta6_temp)
    # Body frame w  ( multiply R)
    eta6 = np.dot(eta6_temp,np.dot(-Rqe1_hat,np.dot(Rq,w))) -dy4 

    # World frame w
    # eta6 = np.dot(eta6_temp,np.dot(-Rqe1_hat,w))
    print("Rqe1_hat:", Rqe1_hat)

    # Second derivative of first three output
    eta3 = np.zeros(3)
    eta3 = -g*e3+Rqe3*xi1 - np.array([ddx_f,ddy_f,0])
    ddy1 = eta3[0]
    ddy2 = eta3[1]
    ddy3 = eta3[2]

    # Third derivative of first three output
    eta4 = np.zeros(3)
    # Body frame w ( multiply R)
    eta4 = Rqe3*xi2+np.dot(-Rqe3_hat,np.dot(Rq,w))*xi1 - np.array([dddx_f,dddy_f,0])

    # World frame w 
    # eta4 = Rqe3*xi2+np.dot(np.dot(F1q,Eq.T),w)*xi1 - np.array([dddx_f,dddy_f,0])
    dddy1 = eta4[0]
    dddy2 = eta4[1]
    dddy3 = eta4[2]

    # Fourth derivative of first three output
    B_qw_temp = np.zeros(3)
    # Body frame w 
    B_qw_temp = xi1*(-np.dot(Rw_hat,np.dot(Rqe3_hat,Rw))+np.dot(Rqe3_hat,np.dot(Rq,np.dot(I_inv,wIw))) ) # np.dot(I_inv,wIw)*xi1-2*w*xi2
    B_qw      = B_qw_temp+xi2*(-2*np.dot(Rqe3_hat,Rw)) - np.array([ddddx_f,ddddy_f,0])   #np.dot(Rqe3_hat,B_qw_temp)

    # World frame w
    # B_qw_temp = xi1*(-np.dot(w_hat,np.dot(Rqe3_hat,w))+np.dot(Rqe3_hat,np.dot(I_inv,wIw))) # np.dot(I_inv,wIw)*xi1-2*w*xi2
    # B_qw      = B_qw_temp+xi2*(-2*np.dot(Rqe3_hat,w)) - np.array([ddddx_f,ddddy_f,0])   #np.dot(Rqe3_hat,B_qw_temp)

    # B_qw = B_qw_temp - np.dot(w_hat,np.dot(Rqe3_hat,w))*xi1

    # Second derivative of yaw output

    # Body frame w
    dRqe1_x = np.dot(e2,np.dot(-Rqe1_hat,Rw)) # \dot{x}
    dRqe1_y = np.dot(e1,np.dot(-Rqe1_hat,Rw)) # \dot{y}
    alpha1 = 2*(Rqe1_x*dRqe1_x+Rqe1_y*dRqe1_y)/x2y2 # (2xdx +2ydy)/(x^2+y^2)
    
    # World frame w
    # dRqe1_x = np.dot(e2,np.dot(-Rqe1_hat,w)) # \dot{x}
    # dRqe1_y = np.dot(e1,np.dot(-Rqe1_hat,w)) # \dot{y}

    # alpha1 = 2*(Rqe1_x*dRqe1_x+Rqe1_y*dRqe1_y)/x2y2 # (2xdx +2ydy)/(x^2+y^2)
    # # alpha2 = math.pow(dRqe1_y,2)-math.pow(dRqe1_x,2)

    # Body frame w

    B_yaw_temp3 =np.zeros(3)
    B_yaw_temp3 = alpha1*np.dot(Rqe1_hat,Rw)+np.dot(Rqe1_hat,np.dot(Rq,np.dot(I_inv,wIw)))-np.dot(Rw_hat,np.dot(Rqe1_hat,Rw))

    B_yaw = np.dot(eta6_temp,B_yaw_temp3) # +alpha2 :Could be an error in math.
    g_yaw = np.zeros(3)
    g_yaw = -np.dot(eta6_temp,np.dot(Rqe1_hat,np.dot(Rq,I_inv)))

    # World frame w
    # B_yaw_temp3 =np.zeros(3)
    # B_yaw_temp3 = alpha1*np.dot(Rqe1_hat,w)+np.dot(Rqe1_hat,np.dot(I_inv,wIw))-np.dot(w_hat,np.dot(Rqe1_hat,w))

    # B_yaw = np.dot(eta6_temp,B_yaw_temp3) # +alpha2 :Could be an error in math.
    # g_yaw = np.zeros(3)
    # g_yaw = -np.dot(eta6_temp,np.dot(Rqe1_hat,I_inv))

    print("g_yaw:", g_yaw)
    # Decoupling matrix A(x)\in\mathbb{R}^4

    A_fl = np.zeros((4,4))
    A_fl[0:3,0] = Rqe3
    # Body frame w
    A_fl[0:3,1:4] = -np.dot(Rqe3_hat,np.dot(Rq,I_inv))*xi1
    # World frame w
    # A_fl[0:3,1:4] = -np.dot(Rqe3_hat,I_inv)*xi1
    A_fl[3,1:4]=g_yaw

    A_fl_inv = np.linalg.inv(A_fl)
    A_fl_det = np.linalg.det(A_fl)
    # print("I_inv:", I_inv)
    print("A_fl:", A_fl)
    print("A_fl_det:", A_fl_det)

    # Output dyamics

    eta = np.hstack([eta1, eta2, eta3, eta5, eta4, eta6])

    # Full feedback controller

    U_temp = np.zeros(4)
    U_temp[0:3]  = B_qw
    U_temp[3]    = B_yaw

    # print("x:", x)
    print("eta_norm:", np.dot(eta, np.dot(Mout,eta) ))
    # print("eta1:", eta1)
    # print("eta2:", eta2)
    # print("eta3:", eta3)
    # print("eta4:", eta4)
    # print("eta5:", eta5)
    # print("eta6:", eta6)
            
    mu = np.zeros(4)
    k = np.zeros((4,14))
    k = np.dot(Bout.T,Mout)
    # print("k:", k)

    mu = -np.dot(k,eta)

    v=-U_temp+mu
    
    U_fl = np.dot(A_fl_inv,v)       # Feedback controller

    U_fl_zero = np.dot(A_fl_inv,-U_temp)

    # dx = robobee_plantBS.evaluate_f(U_fl,x)
    
    # print("dx", dx)

    u = np.zeros(4)

    u[0] = U_fl[0]
    u[1:4] = U_fl[1:4]

    # print("\n######################")
    # # print("qe3:", A_fl[0,0])
    # print("u:", u)
    # print("\n####################33")
    
    deta4 = B_qw+Rqe3*U_fl_zero[0]+np.dot(-np.dot(Rqe3_hat,I_inv),U_fl_zero[1:4])*xi1
    deta6 = B_yaw+np.dot(g_yaw,U_fl_zero[1:4])
    # print("deta4:",deta4)
    # print("deta6:",deta6)

    return u 


# Test LQR


# Run forward simulation from the specified initial condition
duration =40.

input_log, state_log = \
    RunSimulation(robobee_plantBS,
              test_Feedback_Linearization_controller_BS,
              x0=x0,
              duration=duration)



num_iteration = np.size(state_log.data(),1)
num_state=np.size(state_log.data(),0)

state_out =state_log.data();
# time_out =input_log.times();
input_out=input_log.data();

# print("num_iteration,:", num_iteration)
# print("state_out dimension:,:", state_out.shape)
rpy = np.zeros((3,num_iteration)) # Convert to Euler Angle
ubar = np.zeros((4,num_iteration)) # Convert to Euler Angle
u = np.zeros((4,num_iteration)) # Convert to Euler Angle

for j in range(0,num_iteration):

    # ubar[:,j]=test_Feedback_Linearization_controller_BS(state_out[:,j])
    ubar[:,j]=input_out[:,j]
    q_temp =state_out[3:7,j]
    q_temp_norm =math.sqrt(np.dot(q_temp,q_temp));
    q_temp = q_temp/q_temp_norm;
    quat_temp = Quaternion(q_temp)    # Quaternion
    R = RotationMatrix(quat_temp)
    rpy[:,j]=RollPitchYaw(R).vector()
    u[:,j]=ubar[:,j]
    u[0,j]=state_out[7,j] # Control

# print(u)

# Visualize state and input traces
# print("times",state_log.data()[1,:])RollPitchYaw

if show_flag_q==1:
    plt.clf()
    fig = plt.figure(1).set_size_inches(6, 6)
    for i in range(0,3):
        # print("i:%d" %i)
        plt.subplot(3, 1, i+1)
        # print("test:", num_state)
        plt.plot(state_log.sample_times()*time_gain, state_log.data()[i, :])
        plt.grid(True)
        if i==0:
            plt.ylabel("x (m) ")
        elif i==1:
            plt.ylabel("y (m) ")
        elif i==2:
            plt.ylabel("z (m) ")
    plt.xlabel("Time (s)")

    ####- Plot Euler angle
    fig = plt.figure(2).set_size_inches(6, 6)
    for i in range(0,3):
        # print("i:%d" %i)
        plt.subplot(3, 1, i+1)
        # print("test:", num_state)
        plt.plot(state_log.sample_times()*time_gain, rpy[i,:])
        plt.grid(True)
        j=i+3
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("roll (rad)")
        elif i==1:
            plt.ylabel("pitch (rad)")
        elif i==2:
            plt.ylabel("yaw (rad)")
    plt.xlabel("Time (s)")


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
        tau_sec_gain = 1; # 10\tau =1 s
        com_vel_body = tau_sec_gain*state_log.data()[i+8, :];
        plt.plot(state_log.sample_times()*time_gain, com_vel_body)
        plt.grid(True)
        j=i+7
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("vx (m/s)")
        elif i==1:
            plt.ylabel("vy (m/s)")
        elif i==2:
            plt.ylabel("vz (m/s)")
    plt.xlabel("Time (s)")

    fig = plt.figure(4).set_size_inches(6, 6)
    for i in range(0,3):
        # print("i:%d" %i)
        plt.subplot(3, 1, i+1)
        # print("test:", num_state)
        tau_sec_gain = 1; # 10\tau =1 s
        angular_vel_body = tau_sec_gain*state_log.data()[i+11, :];
        plt.plot(state_log.sample_times()*time_gain, angular_vel_body)
        plt.grid(True)
        j=i+10
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("wx (rad/s)")
        elif i==1:
            plt.ylabel("wy (rad/s)")
        elif i==2:
            plt.ylabel("wz (rad/s)")
    plt.xlabel("Time (s)")


if show_flag_control==1:
    fig = plt.figure(5).set_size_inches(6, 6)
    for i in range(0,4):
        # print("i:%d" %i)
        plt.subplot(4, 1, i+1)
        # print("test:", num_state)
        if i==0:
            thrust_mg_gain=1/g; #  1cm = 0.01m
            control = u[i,:]*thrust_mg_gain;
        else:
            mg_gain=1e0; # 1000mg =1g
            control = mg_gain*u[i,:];

        plt.plot(state_log.sample_times()*time_gain, control)
        plt.grid(True)
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("T-W ratio")
        elif i==1:
            plt.ylabel("tau_r (mNmm)")
        elif i==2:
            plt.ylabel("tau_p (mNmm)")
        elif i==3:
            plt.ylabel("tau_y (mNmm)")
    plt.xlabel("Time (s)")

# plt.subplot(num_state, 1, num_state)
# plt.plot(input_log.sample_times(), input_log.data()[0, :])
# plt.ylabel("u[0]")
# plt.xlabel("t")

    plt.grid(True)
    plt.show()
