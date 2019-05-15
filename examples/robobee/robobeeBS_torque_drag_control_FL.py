##########################333
#
# Feedback_linearizaiton with Back Stepping method with quaternion including Aerodynamics drag
# 
# Script originated from robobee_class_feedback_linearization_Lissajous.py
# 
#  CLF-QP controller
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
from pydrake.all import MathematicalProgram
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.gurobi import GurobiSolver
from pydrake.solvers.mosek import MosekSolver



from robobee_plantBS_torque_example import *

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
np.set_printoptions(precision=4, suppress=False)

#-[0.0] Show figure flag

show_flag_qd = 1;
show_flag_q=1;
show_flag_control =1;
# Run forward simulation from the specified initial condition
duration =1.04050


#-[0] Intialization
#-[0-1] Robobee params. From IROS 2015 S.Fuller "Rotating the heading angle of underactuated flapping-wing flyers by wriggle-steering"

# Pick mg unit
m   = 81        # 81 mg
Ixx = 1.42*1e-3      # 1.42 x 10^-3  mg m^2                     | 1.42 x 10 mg cm^2
Iyy = 1.34*1e-3      # 1.34 x 10^-3  mg m^2                     | 1.34 x 10 mg cm^2             
Izz = 0.45*1e-3      # 0.45 x 10^-3  mg m^2                     | 0.45 x 10 mg cm^2
g   = 9.80*1e0       # 9.8 m/s^2    use 1 \tau = 0.1s          | 9.8  x 1 cm/s^2
rw_l = 7.0*1e-3      # 7 x 10^-3 m
bw   = 2.*1e2       # 2 x 10^-4 mg / s

I = np.zeros((3,3))
I[0,0] = Ixx
I[1,1] = Iyy
I[2,2] = Izz

e3= np.array([0,0,1])
r_w = rw_l*e3; # Length to the wing from CoM

#-[0] Initial condition

r0 = np.array([0,0,2.05])
q0 = np.zeros((4)) # quaternion setup
theta0 = 1*math.pi/4;  # angle of rotation
# v0_q=np.array([1.,0.,1.])
v0_q=np.array([0.,-1.,1.])
q0[0]=math.cos(theta0/2) #q0
# print("q:",q[0])
v0_norm=np.sqrt((np.dot(v0_q,v0_q))) # axis of rotation
v0_normalized =v0_q.T/v0_norm
# print("vnorm:", v_norm)
# print("vnormalized", np.dot(v_normalized,v_normalized.T))
q0[1:4]=math.sin(theta0/2)*v0_q.T/v0_norm

xi10 =0.*g; # g;
v0 = np.zeros((3))
w0 = np.zeros((3))          # angular velocity
v0[0] =0.;
v0[1] = -0.;
v0[2] = 0.0001-m*g/bw;
w0[0]=5.0 #-0.1 # 0.1 #0;
w0[1]=0.#10.5 # 0  #-0.1;s
w0[2]=-5. #.1 # 0.1 # 0.2;

xi20 =0;

# Initial Torque
xi30 = 0;
xi31 = 0;
xi32 = 0;

#-[0-1] Stack up the state in R^13
# print("r:", r0.shape, "q",q0.shape,"v",v0.shape,"w", w0.shape)
x0= np.hstack([r0, q0, xi10, v0, w0, xi20, xi30, xi31, xi32])
print("x0:", x0)


input_max = 1e22  # N m  
robobee_plantBS = RobobeePlantBSTorque(
    m = m, Ixx = Ixx, Iyy = Iyy, Izz = Izz, 
    g = g, rw_l= rw_l, bw=bw, input_max = input_max)


#-[1] Fixed point for Linearization

F_T = 0;
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
xi30f =0;
xi31f =0;
xi32f =0;
# u =0 Backstepoped input

uf_T =0;

xf= np.hstack([rf, qf, xi1f, vf, wf, xi2f, xi30f, xi31f, xi32f]) # Fixed point for the state
print("xf:", xf)
uf = np.hstack([uf_T,tau])  # Hovering
print("uf:",uf)

xstackf=robobee_plantBS.evaluate_f(uf,xf)
xstackf_norm = np.dot(xstackf,xstackf)
if xstackf_norm<1e-6:
    print("\n\n1. Set point is a fixed point")
else:
    print("\n\n1. Set point is not a fixed point")


# #- Tracking parameter
time_gain = 1; # 10\tau = 1s

T_period = 5*time_gain # 
w_freq = 2*math.pi/T_period
radius = 0.0 #.5

#-[2] Output dynamics Lyapunov function
# 
#  V(eta)= 1/2eta^T M_out eta
#  for deta = Aout eta + Bout u
#
#  Solving CARE to get Control Lyapunov function v(eta)
#  dVdt < -eta^T Q eta< 0  where Q is positive definite

Q = 1e0*np.eye(15)
Q[0:3,0:3] = 1e2*np.eye(3) # eta1
Q[3:6,3:6] = 1e1*np.eye(3) # eta2
Q[6:9,6:9] = 1e0*np.eye(3) # eta3
Q[9,9] = 1e0     # eta5
Q[10,10] = 1e-1     # eta6
Q[11:14,11:14] = 1e-2*np.eye(3) #eta 4
Q[14,14] = 1e-2     # eta7


# Q[9,9] = 1000
# Q[10:13,10:13] = 1*np.eye(3)

R = 1e0*np.eye(4)

Aout = np.zeros((15,15))
Aout[0:3,3:6]=np.eye(3)
Aout[3:6,6:9]=np.eye(3)
Aout[6:9,11:14]=np.eye(3)
Aout[9,10]=1
Aout[10,14]=1
    
Bout = np.zeros((15,4))
Bout[11:15,0:4]= np.eye(4)

# Observ_matrix=obsv(A, Q)

print("A:", Aout, "B:", Bout)
Mout = solve_continuous_are(Aout,Bout,Q,R)
# print("Mout",Mout)
# print("M_out:", Mout)

# Minimum and maximum eigenvalues for Q and Mout used for CLF-QP constraint

eval_Q = np.linalg.eigvals(Q)
eval_P = np.linalg.eigvals(Mout)

min_e_Q = np.min(eval_Q)
max_e_P = np.max(eval_P)
min_e_P = np.min(eval_P)

print("Evalues: ", [min_e_Q, max_e_P])
# Set up for QP problem
prog = MathematicalProgram()
u_var = prog.NewContinuousVariables(4, "u_var")
solverid = prog.GetSolverId()

tol = 1e-10
prog.SetSolverOption(mp.SolverType.kIpopt,"tol", tol);
prog.SetSolverOption(mp.SolverType.kIpopt,"constr_viol_tol", tol);
prog.SetSolverOption(mp.SolverType.kIpopt,"acceptable_tol", tol);
prog.SetSolverOption(mp.SolverType.kIpopt,"acceptable_constr_viol_tol", tol);

prog.SetSolverOption(mp.SolverType.kIpopt, "print_level", 2) # CAUTION: Assuming that solver used Ipopt

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
    
    q=np.zeros(7)
    qd=np.zeros(10)
    q=x[0:8]
    qd=x[8:18]

    print("qnorm:", np.dot(q[3:7],q[3:7]))

    q0=q[3]
    q1=q[4]
    q2=q[5]
    q3=q[6]
    xi1=q[7]

    r=q[0:3]
    v=qd[0:3]
    w=qd[3:6]
    xi2=qd[6]
    xi30=qd[7]
    xi31=qd[8]
    xi32=qd[9]

    xi3 = np.zeros(3)
    xi3[0]=xi30
    xi3[1]=xi31
    xi3[2]=xi32


    u[0]=0
    # u[1]=0.
    # u[2]=-0.001
    # u[3]=0.01
    u[1:4]= -(2.*1e-1)*w

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
    global g, m, I, r_w, rw_l, bw, xf, Aout, Bout, Mout, T_period, w_freq, radius


    print("%%%%%%%%%%%%%%%%%%%%%")
    print("%%CLF-QP  %%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%")

    print("t:", t)
    prog = MathematicalProgram()
    u_var = prog.NewContinuousVariables(4, "u_var")
    
    # # # Example 1 : Circle

    x_f = radius*math.cos(w_freq*t)
    y_f = radius*math.sin(w_freq*t)
    # print("x_f:",x_f)
    # print("y_f:",y_f)
    dx_f = -radius*math.pow(w_freq,1)*math.sin(w_freq*t)
    dy_f = radius*math.pow(w_freq,1)*math.cos(w_freq*t)
    ddx_f = -radius*math.pow(w_freq,2)*math.cos(w_freq*t)
    ddy_f = -radius*math.pow(w_freq,2)*math.sin(w_freq*t)
    dddx_f = radius*math.pow(w_freq,3)*math.sin(w_freq*t)
    dddy_f = -radius*math.pow(w_freq,3)*math.cos(w_freq*t)
    ddddx_f = radius*math.pow(w_freq,4)*math.cos(w_freq*t)
    ddddy_f = radius*math.pow(w_freq,4)*math.sin(w_freq*t)

    # # Example 2 : Lissajous curve a=1 b=2
    # ratio_ab=2;
    # a=1;
    # b=ratio_ab*a;
    # delta_lissajous = 0 #math.pi/2;

    # x_f = radius*math.sin(a*w_freq*t+delta_lissajous)
    # y_f = radius*math.sin(b*w_freq*t)
    # # print("x_f:",x_f)
    # # print("y_f:",y_f)
    # dx_f = radius*math.pow(a*w_freq,1)*math.cos(a*w_freq*t+delta_lissajous)
    # dy_f = radius*math.pow(b*w_freq,1)*math.cos(b*w_freq*t)
    # ddx_f = -radius*math.pow(a*w_freq,2)*math.sin(a*w_freq*t+delta_lissajous)
    # ddy_f = -radius*math.pow(b*w_freq,2)*math.sin(b*w_freq*t)
    # dddx_f = -radius*math.pow(a*w_freq,3)*math.cos(a*w_freq*t+delta_lissajous)
    # dddy_f = -radius*math.pow(b*w_freq,3)*math.cos(b*w_freq*t)
    # ddddx_f = radius*math.pow(a*w_freq,4)*math.sin(a*w_freq*t+delta_lissajous)
    # ddddy_f = radius*math.pow(b*w_freq,4)*math.sin(b*w_freq*t)


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
    qd=np.zeros(10)
    q=x[0:8]
    qd=x[8:18]

    print("qnorm:", np.dot(q[3:7],q[3:7]))

    q0=q[3]
    q1=q[4]
    q2=q[5]
    q3=q[6]
    xi1=q[7]

    r=q[0:3]
    v=qd[0:3]
    v_vel=v; 

    w=qd[3:6]
    xi2=qd[6]
    xi30=qd[7]
    xi31=qd[8]
    xi32=qd[9]

    xi3 = np.zeros(3)
    xi3[0]=xi30
    xi3[1]=xi31
    xi3[2]=xi32

    

    # Desired hovering point
    zd=xf[2]
    # xd=xf[0]
    # yd=xf[1]
    # wd=xf[11:14]

    # Useful vectors and matrices
    
    (Rq, Eq, wIw, I_inv, fdw, taudw, vd, wd)=robobee_plantBS.GetManipulatorDynamics(q, qd)
    
    w_hat = np.zeros((3,3))
    w_hat[0,:] = np.array([     0,   -w[2],     w[1] ])
    w_hat[1,:] = np.array([  w[2],       0,    -w[0] ])
    w_hat[2,:] = np.array([ -w[1],    w[0],        0 ])

    Iw = np.dot(I,w)
    r_w = rw_l*e3; # Length to the wing from CoM
    wr_w = np.cross(w,r_w) # w x r
    wr_w_hat = np.dot(w_hat,r_w)
    print("wr_w-wr_w_hat", wr_w-wr_w_hat)

    wwr_w = np.cross(w,wr_w) # w x (w x r)
    # w_hat2 = np.dot(w_hat,w_hat)
    # wwr_w_hat = np.dot(w_hat2,r_w)
    # print("wwr_w-wwr_w_hat", wwr_w-wwr_w_hat)
    
    wdr_w = np.cross(wd,r_w)
    wdwr_w = np.cross(wd,wr_w)
    wwdr_w = np.cross(w,wdr_w)
    Iwd = np.dot(I,wd) 
    wdIw = np.cross(wd,Iw)
    wIwd = np.cross(w,Iwd)
    we3 = np.cross(w,e3)
    wwe3 = np.cross(w,we3)
    wde3 = np.cross(wd,e3)
    we1 = np.cross(w,e1)
    wwe1 = np.cross(w,we1)
    wde1 = np.cross(wd,e1)
    wdwe1 = np.cross(wd,we1)
    wwde1 = np.cross(w,wde1)

    # print("wdr_w shape : ", wdr_w.shape)
    # print("wdwr_w shape : ", wdwr_w.shape)
    # print("wwdr_w shape : ", wwdr_w.shape)
    # print("Iwd shape : ", Iwd.shape)
    # print("wdIw shape : ", wdIw.shape)
    # print("wIwd shape : ", wIwd.shape)
    # print("we3 shape : ", we3.shape)
    # print("wwe3 shape : ", wwe3.shape)
    # print("wde3 shape : ", wde3.shape)
    # print("we1 shape : ", we1.shape)
    # print("wwe1 shape : ", wwe1.shape)
    # print("wde1 shape : ", wde1.shape)
    # print("wdwe1 shape : ", wdwe1.shape)
    # print("wwde1 shape : ", wwde1.shape)


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

    Rqr_w = np.dot(Rq,r_w)
    Rqr_w_hat = np.zeros((3,3))
    Rqr_w_hat[0,:] = np.array([        0,   -Rqr_w[2],     Rqr_w[1] ])
    Rqr_w_hat[1,:] = np.array([  Rqr_w[2],          0,    -Rqr_w[0] ])
    Rqr_w_hat[2,:] = np.array([ -Rqr_w[1],    Rqr_w[0],           0 ])

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

    Rdot = np.zeros((3,3))
    Rdot = np.dot(Rq,w_hat)

    # Precalculation
    dfdw_bar = np.zeros(3)
    dfdw = np.zeros(3)
    vdd = np.zeros(3)
    
    dfdw_bar=-1*(vd+np.dot(Rq,wwr_w+wdr_w))
    dfdw = dfdw_bar*bw
    vdd = np.dot(Rq,e3)*xi2+np.dot(Rq,we3)*xi1  + dfdw/m
    
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

    eta1_old = np.zeros(3)
    eta1_old = np.array([y1,y2,y3])
    eta5_old = y4
    
    # # print("y4", y4)
   
    # First derivative of first three output and yaw output
    eta2_old = np.zeros(3)
    eta2_old = v - np.array([dx_f,dy_f,0])
    dy1 = eta2_old[0]
    dy2 = eta2_old[1]
    dy3 = eta2_old[2]

    x2y2 = (math.pow(Rqe1_x,2)+math.pow(Rqe1_y,2)) # x^2+y^2
    
    alpha = np.zeros(3)     #alpha = (ye2T-xe1T)/(x^2+y^2)
    alpha = (Rqe1_y*e2.T-Rqe1_x*e1.T)/x2y2    
    # print("alpha:", alpha)
    # Body frame w  ( multiply R)
    eta6_we1 = np.dot(alpha,np.dot(-Rqe1_hat,np.dot(Rq,w)))
    eta6_old = np.dot(alpha,np.dot(Rq,we1))
    print("eta6-eta6_we1 :", eta6_old-eta6_we1)

    # Second derivative of first three output
    eta3_old = np.zeros(3)
    eta3_old = -g*e3+Rqe3*xi1+fdw/m - np.array([ddx_f,ddy_f,0])
    ddy1 = eta3_old[0]
    ddy2 = eta3_old[1]
    ddy3 = eta3_old[2]

    print("fdw: ", fdw/m)
    print("taudw:", np.dot(I_inv, taudw))
    print("velocity: ", v)
    print("angular vel :", w )
    # Second derivative of yaw output

    # Body frame w
    # dRqe1_x = np.dot(e2,np.dot(-Rqe1_hat,Rw)) # \dot{x}
    # dRqe1_y = np.dot(e1,np.dot(-Rqe1_hat,Rw)) # \dot{y}

    # print("dRqe1_x-dRqe1_x_we1 ", dRqe1_x-dRqe1_x_we1)
    # print("dRqe1_y-dRqe1_y_we1 ", dRqe1_y-dRqe1_y_we1)
    
    dRqe1_x = np.dot(e2,np.dot(Rq,we1)) # \dot{x}
    dRqe1_y = np.dot(e1,np.dot(Rq,we1)) # \dot{y}
    

    alpha1 = 2*(Rqe1_x*dRqe1_x+Rqe1_y*dRqe1_y)/x2y2 # (2xdx +2ydy)/(x^2+y^2)
    
    eta7_temp = np.dot(Rq,-1*alpha1*we1+wwe1+wde1)
    eta7_old = np.dot(alpha, eta7_temp) 

    
    # # Third derivative of first three output
    eta4_old = np.zeros(3)
    vdd = np.zeros(3)
    # Body frame w ( multiply R)
    dfdw_bar=-1*(vd+np.dot(Rq,wwr_w+wdr_w))
    dfdw = dfdw_bar*bw
    vdd = Rqe3*xi2+np.dot(Rq,we3)*xi1  + dfdw/m
    eta4_old = vdd  - np.array([dddx_f,dddy_f,0])
    # eta4 = Rqe3*xi2+np.dot(-Rqe3_hat,np.dot(Rq,w))*xi1 - np.array([dddx_f,dddy_f,0])
    # print("eta4_we3-eta4",eta4_we3-eta4)
    
    dddy1 = eta4_old[0]
    dddy2 = eta4_old[1]
    dddy3 = eta4_old[2]

    # Fourth derivative of first three output
    B_qw_temp = np.zeros(3)

    ddfdw_bar_star = np.zeros(3)
    ddfdw_bar_star_temp = np.zeros(3)
    ddfdw_bar_star_temp2 = np.zeros(3)
    RqTdfdw = np.zeros(3)
    r_wRqdfdw = np.zeros(3)

    RqTdfdw=np.dot(Rq.T,dfdw)
    RqTfdw=np.dot(Rq.T,fdw)
    r_wRqTdfdw=np.cross(r_w,RqTdfdw)
    wRqTfdw=np.cross(w,RqTfdw)
    r_wwRqTfdw=np.cross(r_w,wRqTfdw)
    ddw_temp2 = -1*np.dot(I_inv,wdIw+wIwd)+np.dot(I_inv,r_wRqTdfdw-r_wwRqTfdw) # ddw without the control : used in thrid derivative of yaw as well.

    ddfdw_bar_star_temp = np.cross(w,wwr_w)+wdwr_w+2*wwdr_w-np.cross(r_w,ddw_temp2)
    ddfdw_bar_star = -1*vdd-np.dot(Rq,ddfdw_bar_star_temp)
    B_qw_old = np.dot(Rq,wwe3*xi1+wde3*xi1+2*we3*xi2)+ddfdw_bar_star*bw/m - np.array([ddddx_f,ddddy_f,0]) 

    # Body frame w 
    B_qw_temp = xi1*(-np.dot(Rw_hat,np.dot(Rqe3_hat,Rw))+np.dot(Rqe3_hat,np.dot(Rq,np.dot(I_inv,wIw))) ) # np.dot(I_inv,wIw)*xi1-2*w*xi2
    B_qw      = B_qw_temp+xi2*(-2*np.dot(Rqe3_hat,Rw)) - np.array([ddddx_f,ddddy_f,0])   #np.dot(Rqe3_hat,B_qw_temp)

    # Thrid derivative of yaw output
    
    
    ddRqe1_x = np.dot(e2,np.dot(Rq,wwe1+wde1)) # \dot{x}
    ddRqe1_y = np.dot(e1,np.dot(Rq,wwe1+wde1)) # \dot{y}
    dalpha = (dRqe1_y*e2-dRqe1_x*e1)/x2y2-alpha*alpha1 # Remember that eta6_temp is alpha   
    dalpha1 = 2*(dRqe1_x*dRqe1_x+dRqe1_y*dRqe1_y+Rqe1_x*ddRqe1_x+Rqe1_y*ddRqe1_y)/x2y2 -alpha1*alpha1 # (2xdx +2ydy)/(x^2+y^2)

    ddRqe1_x = np.dot(e2,np.dot(Rq,wwe1+wde1)) # \dot{x}
    ddRqe1_y = np.dot(e1,np.dot(Rq,wwe1+wde1)) # \dot{y}
    dalpha = (dRqe1_y*e2-dRqe1_x*e1)/x2y2-alpha*alpha1 # Remember that eta6_temp is alpha   
    dalpha1 = 2*(dRqe1_x*dRqe1_x+dRqe1_y*dRqe1_y+Rqe1_x*ddRqe1_x+Rqe1_y*ddRqe1_y)/x2y2 -alpha1*alpha1 # (2xdx +2ydy)/(x^2+y^2)

    B_yaw_temp = np.zeros(3)
    B_yaw_temp1 = np.zeros(3)
    B_yaw_temp3 =np.zeros(3)

    ddw_temp3 = -1*np.dot(I_inv,wdIw+wIwd)+np.dot(I_inv,r_wRqdfdw-r_wwRqTfdw) # ddw without the control : used in thrid derivative of yaw as well.
    
    # B_yaw_temp3 = (np.dot(dalpha.T,Rq)+np.dot(alpha.T,np.dot(Rq,w_hat)))
    # B_yaw_temp2 = np.dot(B_yaw_temp3, -1*alpha1*we1+wwe1+wde1) # Scalar number
    # B_yaw_temp1 = -1*dalpha1*we1-1*alpha1*wde1+(wdwe1+wwde1)-np.cross(e1,ddw_temp3)
    # B_yaw_temp = np.dot(Rq,B_yaw_temp1)
    # B_yaw_old = B_yaw_temp2+np.dot(alpha.T,B_yaw_temp)

    B_yaw_temp4 =(-alpha1*we1+wwe1+wde1)
    B_yaw_temp5 = np.dot(np.dot(dalpha.T,Rq),B_yaw_temp4)
    B_yaw_temp6 = np.dot(np.dot(alpha.T,Rdot),B_yaw_temp4)
    B_yaw_temp7 = np.dot(np.dot(alpha.T,Rq),-dalpha1*we1-alpha1*wde1+wdwe1+wwde1)+np.dot(np.dot(alpha.T,Rq),np.cross(ddw_temp2,e1))
    B_yaw_temp3 = (np.dot(dalpha.T,Rq)+np.dot(alpha.T,np.dot(Rq,w_hat)))
    B_yaw_temp2 = np.dot(B_yaw_temp3.T, -1*alpha1*we1+wwe1+wde1) # Scalar number
    B_yaw_temp1 = -1*dalpha1*we1-1*alpha1*wde1+(wdwe1+wwde1)-np.cross(e1,ddw_temp3)
    B_yaw_temp = np.dot(Rq,B_yaw_temp1)
    # B_yaw = B_yaw_temp2+np.dot(alpha.T,B_yaw_temp)
    B_yaw_old = B_yaw_temp5+B_yaw_temp6+B_yaw_temp7

    g_yaw_old = np.zeros(3)
    g_yaw_old = -np.dot(alpha,np.dot(Rqe1_hat,np.dot(Rq,I_inv)))
    

    # # print("g_yaw:", g_yaw)

    # Decoupling matrix A(x)\in\mathbb{R}^4

    A_fl = np.zeros((4,4))
    A_fl[0:3,0] = Rqe3
    A_fl[0:3,1:4] = -np.dot(Rqr_w_hat,np.dot(Rq,I_inv))*bw/m
    A_fl[3,1:4]=g_yaw_old
    print("A_fl_old:", A_fl)
    

    # ###########################################33
    #
    # Correct math....log
    # ?############################################ 

    e1=np.array([1,0,0]) # e3 elementary vector
    e2=np.array([0,1,0]) # e3 elementary vector
    e3=np.array([0,0,1]) # e3 elementary vector
    Rqr_w = np.dot(Rq,r_w)
    Rqr_w_hat = np.zeros((3,3))
    Rqr_w_hat[0,:] = np.array([        0,   -Rqr_w[2],     Rqr_w[1] ])
    Rqr_w_hat[1,:] = np.array([  Rqr_w[2],          0,    -Rqr_w[0] ])
    Rqr_w_hat[2,:] = np.array([ -Rqr_w[1],    Rqr_w[0],           0 ])
    Rqe1=np.dot(Rq,e1)
    Rqe1_x=np.dot(e2.T,Rqe1)
    Rqe1_y=np.dot(e1.T,Rqe1)

    w_hat = np.zeros((3,3))
    w_hat[0,:] = np.array([     0,   -w[2],     w[1] ])
    w_hat[1,:] = np.array([  w[2],       0,    -w[0] ])
    w_hat[2,:] = np.array([ -w[1],    w[0],        0 ])
    print("we3 error", np.array([w[1],-w[0],0])-np.cross(w,e3))

    Rqe1 = np.dot(Rq,e1)
    Rqe1_hat = np.zeros((3,3))
    Rqe1_hat[0,:] = np.array([        0,   -Rqe1[2],     Rqe1[1] ])
    Rqe1_hat[1,:] = np.array([  Rqe1[2],          0,    -Rqe1[0] ])
    Rqe1_hat[2,:] = np.array([ -Rqe1[1],    Rqe1[0],           0 ])
    
    Rqe3 = np.dot(Rq,e3)
    Rqe3_hat = np.zeros((3,3))
    Rqe3_hat[0,:] = np.array([        0,   -Rqe3[2],     Rqe3[1] ])
    Rqe3_hat[1,:] = np.array([  Rqe3[2],          0,    -Rqe3[0] ])
    Rqe3_hat[2,:] = np.array([ -Rqe3[1],    Rqe3[0],           0 ])

    Rdot = np.zeros((3,3))
    Rdot = np.dot(Rq,w_hat)
    zd=xf[2]

    Iw = np.dot(I,w)
    r_w = rw_l*e3; # Length to the wing from CoM
    wr_w = np.cross(w,r_w) # w x r
    wwr_w = np.cross(w,wr_w) # w x (w x r)
    wdr_w = np.cross(wd,r_w)

    wdwr_w = np.cross(wd,wr_w) # wd x (w x r_w)
    # wd_hat = np.zeros((3,3))
    # wd_hat[0,:] = np.array([     0,   -wd[2],     wd[1] ])
    # wd_hat[1,:] = np.array([  wd[2],       0,    -wd[0] ])
    # wd_hat[2,:] = np.array([ -wd[1],    wd[0],        0 ])
    # wdwr_w_hat = np.dot(wd_hat,wr_w)
    # print("wdwr_w - wdwr_w_hat", wdwr_w - wdwr_w_hat)

    wwdr_w = np.cross(w,wdr_w) # w x (wd x r_w)

    Iwd = np.dot(I,wd) 
    wdIw = np.cross(wd,Iw)
    wIwd = np.cross(w,Iwd)
    we3 = np.cross(w,e3.T)
    wwe3 = np.cross(w,we3)
    wde3 = np.cross(wd,e3)
    we1 = np.cross(w,e1.T)
    wwe1 = np.cross(w,we1)
    wde1 = np.cross(wd,e1)
    wdwe1 = np.cross(wd,we1)
    wwde1 = np.cross(w,wde1)

    
    dfdw_bar=-1*(vd+np.dot(Rq,wwr_w+wdr_w))
    dfdw = dfdw_bar*bw
    vdd = Rqe3*xi2+np.dot(Rq,we3)*xi1  + dfdw/m
    
    ddfdw_bar_star = np.zeros(3)
    ddfdw_bar_star_temp = np.zeros(3)
    ddfdw_bar_star_temp2 = np.zeros(3)
    RqTdfdw = np.zeros(3)
    r_wRqdfdw = np.zeros(3)

    RqTdfdw=np.dot(Rq.T,dfdw)
    RqTfdw=np.dot(Rq.T,fdw)
    r_wRqTdfdw=np.cross(r_w,RqTdfdw)
    wRqTfdw=np.cross(w,RqTfdw)
    r_wwRqTfdw=np.cross(r_w,wRqTfdw)
    ddw_temp2 = -1*np.dot(I_inv,wdIw+wIwd)+np.dot(I_inv,r_wRqTdfdw-r_wwRqTfdw) # ddw without the control : used in thrid derivative of yaw as well.
    
    y4 = math.atan2(Rqe1_x,Rqe1_y)-math.pi/8
    # print("Rqe1_x:", Rqe1_x)
    x2y2 = (math.pow(Rqe1_x,2)+math.pow(Rqe1_y,2)) # x^2+y^2
    
    alpha = np.zeros(3)     #alpha = (ye2T-xe1T)/(x^2+y^2)
    alpha = (Rqe1_y*e2-Rqe1_x*e1)/x2y2
    print("alpha 3: ", np.dot(alpha,e3))    
    dRqe1_x = np.dot(e2,np.dot(Rq,we1)) # \dot{x}
    dRqe1_y = np.dot(e1,np.dot(Rq,we1)) # \dot{y}
    
    # print("dRqe1_x-dRqe1_x_we1 ", dRqe1_x-dRqe1_x_we1)
    # print("dRqe1_y-dRqe1_y_we1 ", dRqe1_y-dRqe1_y_we1)
    alpha1 = 2*(Rqe1_x*dRqe1_x+Rqe1_y*dRqe1_y)/x2y2 # (2xdx +2ydy)/(x^2+y^2)
    
    eta7_temp = np.dot(Rq,-1*alpha1*we1+wwe1+wde1)
    eta7 = np.dot(alpha, eta7_temp) 
    
    ddfdw_bar_star_temp = np.cross(w,wwr_w)+wdwr_w+2*wwdr_w-np.cross(r_w,ddw_temp2)
    ddfdw_bar_star = -1*(vdd+np.dot(Rdot,wwr_w+wdr_w)+np.dot(Rq,wdwr_w+wwdr_w)+np.dot(Rq,-np.cross(r_w,ddw_temp2)))
    ddfdw_star= ddfdw_bar_star*bw

    B_qw = np.dot(Rq,wwe3*xi1+wde3*xi1+2*we3*xi2)+ddfdw_star/m 
    ddRqe1_x = np.dot(e2,np.dot(Rq,wwe1+wde1)) # \dot{x}
    ddRqe1_y = np.dot(e1,np.dot(Rq,wwe1+wde1)) # \dot{y}
    dalpha = (dRqe1_y*e2-dRqe1_x*e1)/x2y2-alpha*alpha1 # Remember that eta6_temp is alpha   
    dalpha1 = 2*(dRqe1_x*dRqe1_x+dRqe1_y*dRqe1_y+Rqe1_x*ddRqe1_x+Rqe1_y*ddRqe1_y)/x2y2 -alpha1*alpha1 # (2xdx +2ydy)/(x^2+y^2)

    B_yaw_temp = np.zeros(3)
    B_yaw_temp1 = np.zeros(3)
    B_yaw_temp3 =np.zeros(3)

    ddw_temp3 = -1*np.dot(I_inv,wdIw+wIwd)+np.dot(I_inv,r_wRqdfdw-r_wwRqTfdw) # ddw without the control : used in thrid derivative of yaw as well.
    
    B_yaw_temp4 =(-alpha1*we1+wwe1+wde1)
    B_yaw_temp5 = np.dot(np.dot(dalpha.T,Rq),B_yaw_temp4)
    B_yaw_temp6 = np.dot(np.dot(alpha.T,Rdot),B_yaw_temp4)
    B_yaw_temp7 = np.dot(np.dot(alpha.T,Rq),-dalpha1*we1-alpha1*wde1+wdwe1+wwde1)+np.dot(np.dot(alpha.T,Rq),np.cross(ddw_temp2,e1))
    B_yaw_temp3 = (np.dot(dalpha.T,Rq)+np.dot(alpha.T,np.dot(Rq,w_hat)))
    B_yaw_temp2 = np.dot(B_yaw_temp3.T, -1*alpha1*we1+wwe1+wde1) # Scalar number
    B_yaw_temp1 = -1*dalpha1*we1-1*alpha1*wde1+(wdwe1+wwde1)-np.cross(e1,ddw_temp3)
    B_yaw_temp = np.dot(Rq,B_yaw_temp1)
    # B_yaw = B_yaw_temp2+np.dot(alpha.T,B_yaw_temp)
    B_yaw = B_yaw_temp5+B_yaw_temp6+B_yaw_temp7
    g_yaw = np.zeros(3)
    g_yaw = -np.dot(alpha,np.dot(Rqe1_hat,np.dot(Rq,I_inv)))

    A_fl = np.zeros((4,4))
    A_fl[0:3,0] = np.dot(Rq,e3)
    A_fl[0:3,1:4] = np.dot(Rqr_w_hat,np.dot(Rq,I_inv))*bw/m
    A_fl[3,1:4]=g_yaw


    # ############################################# 




    A_fl_inv = np.linalg.inv(A_fl)
    A_fl_det = np.linalg.det(A_fl)
    # print("I_inv:", I_inv)
    print("A_fl:", A_fl)
    print("A_fl_det:", A_fl_det)

    # Drift in the output dynamics

    U_temp = np.zeros(4)
    U_temp[0:3]  = B_qw
    U_temp[3]    = B_yaw

    # Output dyamics
    eta1 = r   - np.array([x_f,y_f,zd])
    eta2 = v   - np.array([dx_f,dy_f,0])
    eta3 = vd  - np.array([ddx_f,ddy_f,0])
    eta4 = vdd - np.array([dddx_f,dddy_f,0])
    # deta4_all[:,j] = B_qw + A_flu[0:3]
    eta5 = y4
    eta6 = np.dot(alpha.T,np.dot(Rq,we1))
    eta7 = eta7
    # deta7_all[:,j] = B_yaw + A_flu[3]

    eta = np.hstack([eta1, eta2, eta3, eta5, eta6, eta4, eta7])
    # deta =np.hstack([eta2.T, eta3.T, eta4.T, eta6.T, eta7.T, 0,0,0,0])
    # print("deta- d_eta", np.dot(Aout,eta)-deta)
    eta_norm = np.dot(eta,eta)
    print("eta_norm:", eta_norm)
    
    # v-CLF QP controller

    FP_PF = np.dot(Aout.T,Mout)+np.dot(Mout,Aout)
    PG = np.dot(Mout, Bout)
    
    L_FVx = np.dot(eta,np.dot(FP_PF,eta))
    L_GVx = 2*np.dot(eta.T,PG) # row vector
    L_fhx_star = U_temp;

    Vx = np.dot(eta, np.dot(Mout,eta) )
    # phi0_exp = L_FVx+np.dot(L_GVx,L_fhx_star)+(min_e_Q/max_e_P)*Vx*1    # exponentially stabilizing
    phi0_exp = L_FVx+np.dot(L_GVx,L_fhx_star)+min_e_Q*eta_norm      # more exact bound - exponentially stabilizing
    phi1_decouple = np.dot(L_GVx,A_fl)
    
    # # Solve QP
    v_var = np.dot(A_fl,u_var)  + L_fhx_star
    Quadratic_Positive_def = np.matmul(A_fl.T,A_fl)
    QP_det = np.linalg.det(Quadratic_Positive_def)
    c_QP = 2*np.dot(L_fhx_star.T,A_fl)
    
    
    # CLF_QP_cost_v = np.dot(v_var,v_var) // Exact quadratic cost
    CLF_QP_cost_v_effective = np.dot(u_var, np.dot(Quadratic_Positive_def,u_var))+np.dot(c_QP,u_var) # Quadratic cost without constant term
    
    # CLF_QP_cost_u = np.dot(u_var,u_var)
    
    phi1 = np.dot(phi1_decouple,u_var)

    # #----Printing intermediate states

    # # print("L_fhx_star: ",L_fhx_star)
    # # print("c_QP:", c_QP)
    # # print("Qp : ",Quadratic_Positive_def)
    # # print("Qp det: ", QP_det)
    # # print("c_QP", c_QP)

    # # print("phi0_exp: ", phi0_exp)
    # # print("PG:", PG)
    # # print("L_GVx:", L_GVx)
    # # print("eta6", eta6)
    # # print("d : ", phi1_decouple)
    # # print("Cost expression:", CLF_QP_cost_v)
    # # print("Const expression:", phi0_exp+phi1)

    # #----Different solver option // Gurobi did not work with python at this point (some binding issue 8/8/2018)
    # # solver = IpoptSolver()
    # # solver = GurobiSolver()
    # # print solver.available()
    # # assert(solver.available()==True)
    # # assertEqual(solver.solver_type(), mp.SolverType.kGurobi)
    # # solver.Solve(prog)
    # # assertEqual(result, mp.SolutionResult.kSolutionFound)
    
    # # mp.AddLinearConstraint()
    # # print("x:", x)
    # # print("phi_0_exp:", phi0_exp)
    # # print("phi1_decouple:", phi1_decouple)
    
    print("eta1:", eta1)
    print("eta2:", eta2)
    print("eta3:", eta3)
    print("eta4:", eta4)
    print("eta5:", eta5)
    print("eta6:", eta6)
    print("eta7:", eta7)

    print("eta1 error:", eta1-eta1_old)
    print("eta2 error:", eta2-eta2_old)
    print("eta3 error:", eta3-eta3_old)
    print("eta4 error:", eta4-eta4_old)
    print("eta5 error:", eta5-eta5_old)
    print("eta6 error:", eta6-eta6_old)
    print("eta7 error:", eta7-eta7_old)

    print("B_qw error:", B_qw-B_qw_old)
    print("B_yaw error:", B_yaw-B_yaw_old)
    print('g_yaw error:', g_yaw-g_yaw_old)
      
    # Output Feedback Linearization controller 
    mu = np.zeros(4)
    k = np.zeros((4,15))
    k = np.matmul(Bout.T,Mout)
    # k = np.matmul(np.linalg.inv(R),k)

    mu = -1/2*np.matmul(k,eta)

    # # Testing the feedback 
    # mu = np.zeros(4)
    # epsilonnn=1e-1
    # mu[0:3] = -1/math.pow(epsilonnn,3)*eta2 -3/math.pow(epsilonnn,2)*eta3 -3/math.pow(epsilonnn,1)*eta4 
    # mu[3] = -1/math.pow(epsilonnn,2)*eta6-2/math.pow(epsilonnn,1)*eta7
    # u_correct = np.zeros(4)
    # kd = 2*1e-1
    # u_correct[1:4] = -kd*w
    
    v=-U_temp+mu

    
    print("Utemp: ", np.matmul(A_fl_inv,U_temp))
    print("mu: ", np.matmul(A_fl_inv,mu))
    U_fl = np.matmul(A_fl_inv,v)           # Output Feedback controller to comare the result with CLF-QP solver

    # Set up the QP problem
    prog.AddQuadraticCost(CLF_QP_cost_v_effective)
    prog.AddConstraint(phi0_exp+phi1<=0)

    prog.SetSolverOption(mp.SolverType.kIpopt, "print_level", 5) # CAUTION: Assuming that solver used Ipopt

    print("CLF value:", Vx) # Current CLF value

    # prog.SetInitialGuess(u_var, U_fl)
    # prog.Solve() # Solve with default osqp
    
    # # solver.Solve(prog)
    # print("Optimal u : ", prog.GetSolution(u_var))
    # U_CLF_QP = prog.GetSolution(u_var)

    # #---- Printing for debugging
    # # dx = robobee_plantBS.evaluate_f(U_fl,x)    
    # # print("dx", dx)
    # # print("\n######################")
    # # # print("qe3:", A_fl[0,0])
    # # print("u:", u)
    # # print("\n####################33")
    
    # # deta4 = B_qw+Rqe3*U_fl_zero[0]+np.matmul(-np.matmul(Rqe3_hat,I_inv),U_fl_zero[1:4])*xi1
    # # deta6 = B_yaw+np.dot(g_yaw,U_fl_zero[1:4])
    # # print("deta4:",deta4)
    # # print("deta6:",deta6)


    # phi1_opt = np.dot(phi1_decouple, U_CLF_QP)
    phi1_opt_FL = np.dot(phi1_decouple, U_fl)

    print("FL u: ", U_fl)
    # print("CLF u:", U_CLF_QP)
    print("Cost FL: ", np.dot(v,v))

    # v_CLF = np.dot(A_fl,U_CLF_QP)+L_fhx_star
    # print("Cost CLF: ", np.dot(v_CLF,v_CLF))
    print("Constraint FL : ", phi0_exp+phi1_opt_FL)
    # print("Constraint CLF : ", phi0_exp+phi1_opt)
    u = U_fl

    print("eigenvalues minQ maxP:", [min_e_Q, max_e_P])
    # print("minP eval : ", min_e_P)

    # Null test output
    u = np.zeros(4)
    global g, xf, uf, K
    
    u[0]=0
    # u[1:4]= -(2.*1e-1)*w
    u[1]=0.0
    # u[2]=-0.
    # u[3]=0.

    return u 


# Test LQR



flag_FL_controller =1;
if flag_FL_controller==1:
    input_log, state_log = \
        RunSimulation(robobee_plantBS,
              test_Feedback_Linearization_controller_BS,
              x0=x0,
              duration=duration)
elif flag_FL_controller==0:
    input_log, state_log = \
        RunSimulation(robobee_plantBS,
              test_controller,
              x0=x0,
              duration=duration)



num_iteration = np.size(state_log.data(),1)
num_state=np.size(state_log.data(),0)

state_out =state_log.data();
# time_out =input_log.times();
input_out=input_log.data();
time_out=state_log.sample_times()*1.
print("size of inputout", input_out.shape)
print("size of time-out", time_out.shape)

# print("num_iteration,:", num_iteration)
# print("state_out dimension:,:", state_out.shape)
rpy = np.zeros((3,num_iteration)) # Convert to Euler Angle
ubar = np.zeros((4,num_iteration)) # Convert to Euler Angle
u = np.zeros((4,num_iteration)) # Convert to Euler Angle
fdw_all = np.zeros((3,num_iteration))
d_fdw_all = np.zeros((3,num_iteration))
dfdw_all = np.zeros((3,num_iteration))
vdd_all = np.zeros((3,num_iteration))
wdd_all = np.zeros((3,num_iteration))
d_wd_all = np.zeros((3,num_iteration))
vd_all = np.zeros((3,num_iteration))
d_vd_all = np.zeros((3,num_iteration))
wd_all = np.zeros((3,num_iteration))
ddfdw_all =np.zeros((3,num_iteration))
d_dfdw_all =np.zeros((3,num_iteration))
eta1_all = np.zeros((3,num_iteration))
d_eta1_all = np.zeros((3,num_iteration))
eta2_all = np.zeros((3,num_iteration))
d_eta2_all = np.zeros((3,num_iteration))
eta3_all =np.zeros((3,num_iteration))
d_eta3_all =np.zeros((3,num_iteration))
eta4_all = np.zeros((3,num_iteration))
d_eta4_all = np.zeros((3,num_iteration))
deta4_all = np.zeros((3,num_iteration))
eta5_all =np.zeros((3,num_iteration))
d_eta5_all =np.zeros((3,num_iteration))
eta6_all = np.zeros((3,num_iteration))
d_eta6_all = np.zeros((3,num_iteration))
eta7_all = np.zeros((3,num_iteration))
d_eta7_all = np.zeros((3,num_iteration))
deta7_all =np.zeros((3,num_iteration))
alpha_all = np.zeros((3,num_iteration))
dalpha_all =np.zeros((3,num_iteration))
d_alpha_all =np.zeros((3,num_iteration))
u_all = np.zeros((4,num_iteration))
alpha1_all=np.zeros(num_iteration)
dalpha1_all=np.zeros(num_iteration)
d_alpha1_all=np.zeros(num_iteration)
Kinetic_all=np.zeros(num_iteration)
Kinetic_all_compare=np.zeros(num_iteration)
V_lyapu_upper=np.zeros(num_iteration)
V1_lyapu=np.zeros(num_iteration)
V2_lyapu=np.zeros(num_iteration)
V3_lyapu=np.zeros(num_iteration)
d_Kinetic_all=np.zeros(num_iteration)
vd_Kinetic_all =np.zeros(num_iteration)
int_V2 = np.zeros(num_iteration)
Total_energy = np.zeros(num_iteration)
Angular_momentum = np.zeros(num_iteration)
int_CoM = np.zeros(num_iteration)
int_Rot = np.zeros(num_iteration)
Work_done = np.zeros(num_iteration)
Conservation_energy = np.zeros(num_iteration)

xi1_all = np.zeros(num_iteration)
xi2_all = np.zeros(num_iteration)
d_xi1_all = np.zeros(num_iteration)
d_xi2_all = np.zeros(num_iteration)

Vx_all = np.zeros(num_iteration)
eta_norm_all = np.zeros(num_iteration)
for j in range(0,num_iteration):

    # ubar[:,j]=test_Feedback_Linearization_controller_BS(state_out[:,j])
    # print("r_w:", r_w)
    ubar[:,j]=input_out[:,j]
    q_temp =state_out[3:7,j]
    q_temp_norm =math.sqrt(np.dot(q_temp,q_temp));
    q_temp = q_temp/q_temp_norm;
    quat_temp = Quaternion(q_temp)    # Quaternion

    R = RotationMatrix(quat_temp)
    rpy[:,j]=RollPitchYaw(R).vector()
    u[1:4,j]=state_out[15:18,j]
    u[0,j]=state_out[7,j] # Control

    (Rq, Eq, wIw, I_inv, fdw, taudw, vd, wd)=robobee_plantBS.GetManipulatorDynamics(state_out[0:8,j], state_out[8:18,j])
    
    xi1=state_out[7,j]
    r=state_out[0:3,j]
    q_eval = state_out[3:7,j] 
    v=state_out[8:11,j]
    v_vel = v;
    w=state_out[11:14,j]
    xi2=state_out[14,j]

    e1=np.array([1,0,0]) # e3 elementary vector
    e2=np.array([0,1,0]) # e3 elementary vector
    e3=np.array([0,0,1]) # e3 elementary vector
    Rqr_w = np.dot(Rq,r_w)
    Rqr_w_hat = np.zeros((3,3))
    Rqr_w_hat[0,:] = np.array([        0,   -Rqr_w[2],     Rqr_w[1] ])
    Rqr_w_hat[1,:] = np.array([  Rqr_w[2],          0,    -Rqr_w[0] ])
    Rqr_w_hat[2,:] = np.array([ -Rqr_w[1],    Rqr_w[0],           0 ])
    Rqe1=np.dot(Rq,e1)
    Rqe1_x=np.dot(e2.T,Rqe1)
    Rqe1_y=np.dot(e1.T,Rqe1)

    w_hat = np.zeros((3,3))
    w_hat[0,:] = np.array([     0,   -w[2],     w[1] ])
    w_hat[1,:] = np.array([  w[2],       0,    -w[0] ])
    w_hat[2,:] = np.array([ -w[1],    w[0],        0 ])
    # print("we3 error", np.array([w[1],-w[0],0])-np.cross(w,e3))

    e3_hat = np.zeros((3,3))
    e3_hat[0,:] = np.array([        0,   -e3[2],     e3[1] ])
    e3_hat[1,:] = np.array([  e3[2],          0,    -e3[0] ])
    e3_hat[2,:] = np.array([ -e3[1],    e3[0],           0 ])

    Rqe1 = np.dot(Rq,e1)
    Rqe1_hat = np.zeros((3,3))
    Rqe1_hat[0,:] = np.array([        0,   -Rqe1[2],     Rqe1[1] ])
    Rqe1_hat[1,:] = np.array([  Rqe1[2],          0,    -Rqe1[0] ])
    Rqe1_hat[2,:] = np.array([ -Rqe1[1],    Rqe1[0],           0 ])
    
    Rqe3 = np.dot(Rq,e3)
    Rqe3_hat = np.zeros((3,3))
    Rqe3_hat[0,:] = np.array([        0,   -Rqe3[2],     Rqe3[1] ])
    Rqe3_hat[1,:] = np.array([  Rqe3[2],          0,    -Rqe3[0] ])
    Rqe3_hat[2,:] = np.array([ -Rqe3[1],    Rqe3[0],           0 ])

    Rdot = np.zeros((3,3))
    Rdot = np.dot(Rq,w_hat)
    zd=xf[2]

    Iw = np.dot(I,w)

    Iwd = np.dot(I,wd) 
    r_w = rw_l*e3; # Length to the wing from CoM
    wr_w = np.cross(w,r_w) # w x r
    wwr_w = np.cross(w,wr_w) # w x (w x r)
    wdr_w = np.cross(wd,r_w)

    wdwr_w = np.cross(wd,wr_w) # wd x (w x r_w)
    # wd_hat = np.zeros((3,3))
    # wd_hat[0,:] = np.array([     0,   -wd[2],     wd[1] ])
    # wd_hat[1,:] = np.array([  wd[2],       0,    -wd[0] ])
    # wd_hat[2,:] = np.array([ -wd[1],    wd[0],        0 ])
    # wdwr_w_hat = np.dot(wd_hat,wr_w)
    # print("wdwr_w - wdwr_w_hat", wdwr_w - wdwr_w_hat)

    wwdr_w = np.cross(w,wdr_w) # w x (wd x r_w)

    Iwd = np.dot(I,wd) 
    wdIw = np.cross(wd,Iw)
    wIwd = np.cross(w,Iwd)
    we3 = np.cross(w,e3.T)
    wwe3 = np.cross(w,we3)
    wde3 = np.cross(wd,e3)
    we1 = np.cross(w,e1.T)
    wwe1 = np.cross(w,we1)
    wde1 = np.cross(wd,e1)
    wdwe1 = np.cross(wd,we1)
    wwde1 = np.cross(w,wde1)

    fdw_all[:,j] = fdw
    dfdw_bar=-1*(vd+np.dot(Rq,wwr_w+wdr_w))
    dfdw = dfdw_bar*bw
    dfdw_all[:,j] = dfdw
    vdd = Rqe3*xi2+np.dot(Rq,we3)*xi1  + dfdw/m
    vdd_all[:,j] = vdd
    vd_all[:,j] = vd
    wd_all[:,j] = wd

    ddfdw_bar_star = np.zeros(3)
    ddfdw_bar_star_temp = np.zeros(3)
    ddfdw_bar_star_temp2 = np.zeros(3)
    RqTdfdw = np.zeros(3)
    r_wRqdfdw = np.zeros(3)

    RqTdfdw=np.dot(Rq.T,dfdw)
    RqTfdw=np.dot(Rq.T,fdw)
    r_wRqTdfdw=np.cross(r_w,RqTdfdw)
    wRqTfdw=np.cross(w,RqTfdw)
    r_wwRqTfdw=np.cross(r_w,wRqTfdw)
    ddw_temp2 = -1*np.dot(I_inv,wdIw+wIwd)+np.dot(I_inv,r_wRqTdfdw-r_wwRqTfdw) # ddw without the control : used in thrid derivative of yaw as well.
    wdd = ddw_temp2+np.dot(I_inv,ubar[1:4,j])
    wdd_all[:,j]=wdd

    y4 = math.atan2(Rqe1_x,Rqe1_y)-math.pi/8
    # print("Rqe1_x:", Rqe1_x)
    x2y2 = (math.pow(Rqe1_x,2)+math.pow(Rqe1_y,2)) # x^2+y^2
    
    alpha = np.zeros(3)     #alpha = (ye2T-xe1T)/(x^2+y^2)
    alpha = (Rqe1_y*e2-Rqe1_x*e1)/x2y2
    # print("alpha 3: ", np.dot(alpha,e3))    
    dRqe1_x = np.dot(e2,np.dot(Rq,we1)) # \dot{x}
    dRqe1_y = np.dot(e1,np.dot(Rq,we1)) # \dot{y}
    
    # print("dRqe1_x-dRqe1_x_we1 ", dRqe1_x-dRqe1_x_we1)
    # print("dRqe1_y-dRqe1_y_we1 ", dRqe1_y-dRqe1_y_we1)
    alpha1 = 2*(Rqe1_x*dRqe1_x+Rqe1_y*dRqe1_y)/x2y2 # (2xdx +2ydy)/(x^2+y^2)
    
    eta7_temp = np.dot(Rq,-1*alpha1*we1+wwe1+wde1)
    eta7 = np.dot(alpha, eta7_temp) 
    
    ddfdw_bar_star_temp = np.cross(w,wwr_w)+wdwr_w+2*wwdr_w-np.cross(r_w,ddw_temp2)
    ddfdw_bar_star = -1*(vdd+np.dot(Rdot,wwr_w+wdr_w)+np.dot(Rq,wdwr_w+wwdr_w)+np.dot(Rq,-np.cross(r_w,ddw_temp2)))
    ddfdw_star= ddfdw_bar_star*bw

    B_qw = np.dot(Rq,wwe3*xi1+wde3*xi1+2*we3*xi2)+ddfdw_star/m 
    ddRqe1_x = np.dot(e2,np.dot(Rq,wwe1+wde1)) # \dot{x}
    ddRqe1_y = np.dot(e1,np.dot(Rq,wwe1+wde1)) # \dot{y}
    dalpha = (dRqe1_y*e2-dRqe1_x*e1)/x2y2-alpha*alpha1 # Remember that eta6_temp is alpha   
    dalpha1 = 2*(dRqe1_x*dRqe1_x+dRqe1_y*dRqe1_y+Rqe1_x*ddRqe1_x+Rqe1_y*ddRqe1_y)/x2y2 -alpha1*alpha1 # (2xdx +2ydy)/(x^2+y^2)

    alpha_all[:,j]=alpha
    dalpha_all[:,j]=dalpha
    alpha1_all[j]=alpha1
    dalpha1_all[j]=dalpha1
    B_yaw_temp = np.zeros(3)
    B_yaw_temp1 = np.zeros(3)
    B_yaw_temp3 =np.zeros(3)

    ddw_temp3 = -1*np.dot(I_inv,wdIw+wIwd)+np.dot(I_inv,r_wRqdfdw-r_wwRqTfdw) # ddw without the control : used in thrid derivative of yaw as well.
    
    B_yaw_temp4 =(-alpha1*we1+wwe1+wde1)
    B_yaw_temp5 = np.dot(np.dot(dalpha.T,Rq),B_yaw_temp4)
    B_yaw_temp6 = np.dot(np.dot(alpha.T,Rdot),B_yaw_temp4)
    B_yaw_temp7 = np.dot(np.dot(alpha.T,Rq),-dalpha1*we1-alpha1*wde1+wdwe1+wwde1)+np.dot(np.dot(alpha.T,Rq),np.cross(ddw_temp2,e1))
    B_yaw_temp3 = (np.dot(dalpha.T,Rq)+np.dot(alpha.T,np.dot(Rq,w_hat)))
    B_yaw_temp2 = np.dot(B_yaw_temp3.T, -1*alpha1*we1+wwe1+wde1) # Scalar number
    B_yaw_temp1 = -1*dalpha1*we1-1*alpha1*wde1+(wdwe1+wwde1)-np.cross(e1,ddw_temp3)
    B_yaw_temp = np.dot(Rq,B_yaw_temp1)
    # B_yaw = B_yaw_temp2+np.dot(alpha.T,B_yaw_temp)
    B_yaw = B_yaw_temp5+B_yaw_temp6+B_yaw_temp7
    g_yaw = np.zeros(3)
    g_yaw = -np.dot(alpha,np.dot(Rqe1_hat,np.dot(Rq,I_inv)))

    A_fl = np.zeros((4,4))
    A_fl[0:3,0] = np.dot(Rq,e3)
    A_fl[0:3,1:4] = np.dot(Rqr_w_hat,np.dot(Rq,I_inv))*bw/m
    A_fl[3,1:4]=g_yaw
    A_flu = np.dot(A_fl,ubar[:,j])
    # ddfdw_all[:,j] =(ddfdw_star -bw*np.dot(Rq,-np.cross(r_w,np.dot(I_inv,ubar[1:4,j]))))
    ddfdw_all[:,j] =(ddfdw_star +np.dot(np.dot(Rqr_w_hat,np.dot(Rq,I_inv))*bw,ubar[1:4,j]))
    # deta7_all[:,j] = B_yaw_temp5+B_yaw_temp6+B_yaw_temp7
    # ddfdw_all[:,j] =-bw*(ddfdw_bar_star+np.dot(Rq,np.cross(wdd.T,r_w.T))) 
    # ddfdw_all[:,j] =-bw*(vdd+np.dot(Rdot, wwr_w+wdr_w)+np.dot(Rq,wdwr_w+wwdr_w+ np.cross(wdd.T,r_w.T))) 
    # Output dynamics
    
    # print("alpha:", alpha)
    # Body frame w  ( multiply R)
    # eta6_we1 = np.dot(alpha,np.dot(-Rqe1_hat,np.dot(Rq,w)))

    # Body frame w
    # dRqe1_x = np.dot(e2,np.dot(-Rqe1_hat,Rw)) # \dot{x}
    # dRqe1_y = np.dot(e1,np.dot(-Rqe1_hat,Rw)) # \dot{y}
    
    
    eta1 = r - zd*e3
    eta2 = v
    eta3 = vd
    eta4 = vdd
    eta5 = y4
    eta6 = np.dot(alpha.T,np.dot(Rq,we1))
    eta7 = eta7 

    eta1_all[:,j] = r - zd*e3
    eta2_all[:,j] = v
    eta3_all[:,j] = vd
    xi1_all[j] = xi1
    xi2_all[j] = xi2
    eta4_all[:,j] = vdd
    deta4_all[:,j] = B_qw + A_flu[0:3]
    eta5_all[:,j] = y4
    eta6_all[:,j] = np.dot(alpha.T,np.dot(Rq,we1))
    eta7_all[:,j] = eta7
    deta7_all[:,j] = B_yaw + A_flu[3]

    eta = np.hstack([eta1, eta2, eta3, eta5, eta6, eta4, eta7])
    eta_norm_all[j] = np.dot(eta,eta)
    
    Vx_all[j] = np.dot(eta, np.dot(Mout,eta) )
    # print("d_alpha1_all", d_alpha1_all.shape)
    # print("j", j)
    # print("time_out", time_out[j])


    A_fl_inv = np.linalg.inv(A_fl)
    A_fl_det = np.linalg.det(A_fl)
      # Drift in the output dynamics

    U_temp = np.zeros(4)
    U_temp[0:3]  = B_qw
    U_temp[3]    = B_yaw
      
    # Output Feedback Linearization controller 
    mu = np.zeros(4)
    k = np.zeros((4,15))
    k = np.matmul(Bout.T,Mout)
    # k = np.matmul(np.linalg.inv(R),k)

    mu = -1/2*np.matmul(k,eta)

    v=-U_temp+mu
    
    u_all[:,j] = np.matmul(A_fl_inv,v)            # Output Feedback controller to comare the result with CLF-QP solver



    # if j>0:
       
    if (time_out[j]-time_out[j-1])>0:

        d_eta1_all[:,j]= (eta1_all[:,j]-eta1_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_eta2_all[:,j]= (eta2_all[:,j]-eta2_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_eta3_all[:,j]= (eta3_all[:,j]-eta3_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_eta4_all[:,j]= (eta4_all[:,j]-eta4_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_eta5_all[:,j]= (eta5_all[:,j]-eta5_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_eta6_all[:,j]= (eta6_all[:,j]-eta6_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_eta7_all[:,j]= (eta7_all[:,j]-eta7_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_alpha_all[:,j] = (alpha_all[:,j]-alpha_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_alpha1_all[j] = (alpha1_all[j]-alpha1_all[j-1])/(time_out[j]-time_out[j-1])
        d_wd_all[:,j] = (wd_all[:,j]-wd_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_dfdw_all[:,j] =(dfdw_all[:,j]-dfdw_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_fdw_all[:,j] =(fdw_all[:,j]-fdw_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_vd_all[:,j] = (vd_all[:,j]-vd_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_xi1_all[j] = (xi1_all[j]-xi1_all[j-1])/(time_out[j]-time_out[j-1])
        d_xi2_all[j] = (xi2_all[j]-xi2_all[j-1])/(time_out[j]-time_out[j-1])
        
    else:

        print("time step: ",(time_out[j]-time_out[j-1]) )
        print("j", j)
        print("timeeout[j],", time_out[j])

    v_equiv=v_vel-(-m*g/bw*e3)
    V_equiv=v_equiv+np.dot(Rq,wr_w)
    w_equiv = np.cross(e3,Iw)
    w_equiv2 = np.cross(e3,w)
    R_equiv = np.cross(e3,Rqe3)

    d_w_equiv = np.cross(e3,Iwd)
    q_eval_vec = q_eval[1:4]
    q_eval_vec_norm=np.sqrt(np.dot(q_eval_vec.T,q_eval_vec))
    q_eval_vec_normalized = q_eval_vec/q_eval_vec_norm
    q_equiv = q_eval_vec_normalized-e3

    temp_V = -bw*rw_l*np.cross(Rqe3,v_vel)
    if j==0:
        int_V2[j] = np.dot(w,np.dot(Rq.T,temp_V))
        int_CoM[j]= np.dot(v_vel,fdw)
        int_Rot[j]= np.dot(w,taudw)
    else:
        int_V2[j] = int_V2[j-1]+(time_out[j]-time_out[j-1])*(np.dot(w,np.dot(Rq.T,temp_V)))
        int_CoM[j] = int_CoM[j-1]+(time_out[j]-time_out[j-1])*(np.dot(v_vel,fdw))
        int_Rot[j] = int_Rot[j-1]+(time_out[j]-time_out[j-1])*(np.dot(w,taudw))

    e3_hat_square = -np.dot(e3_hat,e3_hat)
    V1_lyapu[j] = 1./2.*m*np.dot(v_equiv,v_equiv)#math.pow(m,2)*math.pow(g,2)/math.pow(bw,1)+np.dot(v_vel,fdw)+np.dot(w,taudw)#1./2.*bw*np.dot(V_equiv,V_equiv)
    V2_lyapu[j] = 1./2.*np.dot(w,Iw) 
    V3_lyapu[j] = m*g*rw_l*(1-np.dot(e3,Rqe3))
    Kinetic_all[j]= V1_lyapu[j]+V2_lyapu[j]+V3_lyapu[j]
    Kinetic_all_compare[j]= V1_lyapu[j]+V2_lyapu[j]
    
    d_Kinetic_all[j]=m*np.dot(v_equiv,vd) + np.dot(w,Iwd) #np.dot(Iw, Iwd) #+1./2.*np.dot(vd,vd) 
    vd_Kinetic_all[j]=m*np.dot(v_equiv,vd) + np.dot(w,Iwd) -(m*g*rw_l*np.dot(e3,np.dot(Rdot,e3)))
    V_lyapu_upper[j] = vd_Kinetic_all[j] #math.exp(-bw/m*time_out[j])*V1_lyapu[0]+math.exp(-bw*rw_l*time_out[j])*V2_lyapu[0]
    # Kinetic_all[j]=1./2.*np.dot(fdw,fdw)
    # print("fdw :", fdw)
    Total_energy[j] = 1./2.*m*np.dot(v_vel,v_vel) + m*g*r[2] + 1./2.*np.dot(w,Iw)
    Work_done[j] = int_CoM[j]+int_Rot[j]
    Conservation_energy[j] = Total_energy[j]-Work_done[j]
    Angular_momentum[j] =  1./2.*np.dot(w,Iw)

    # V1_lyapu[j] = m*np.dot(v_vel,vd)+np.dot(w,Iwd)+m*g*v_vel[2]#np.dot(v_vel,fdw)+np.dot(w,taudw)# Total_energy[0]+Work_done[j]

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

fig = plt.figure(2).set_size_inches(6, 6)
for i in range(0,4):
  # print("i:%d" %i)
    plt.subplot(4, 1, i+1)
    # print("test:", num_state)
    plt.plot(state_log.sample_times(), state_log.data()[i+3, :])
    plt.grid(True)
    j=i+3
    # plt.ylabel("x[%d]" % j)
    if i==0:
      plt.ylabel("q0")
    elif i==1:
      plt.ylabel("q1")
    elif i==2:
      plt.ylabel("q2")
    elif i==3:
      plt.ylabel("q3")

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
    # fig = plt.figure(5).set_size_inches(6, 6)
    # for i in range(0,4):
    #     # print("i:%d" %i)
    #     plt.subplot(4, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1; #  1cm = 0.01m
    #         control = ubar[i,:]*thrust_mg_gain;
    #         compare = u_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1e0; # 1000mg =1g
    #         control = mg_gain*ubar[i,:];
    #         compare = mg_gain*u_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.plot(state_log.sample_times()*time_gain, compare, color='red')
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("Thrust accel (m/s^2)")
    #     elif i==1:
    #         plt.ylabel("tau_r (mNmm)")
    #     elif i==2:
    #         plt.ylabel("tau_p (mNmm)")
    #     elif i==3:
    #         plt.ylabel("tau_y (mNmm)")
    # plt.xlabel("Time (s)")

    # fig = plt.figure(6).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(3, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1./m; #  1cm = 0.01m
    #         control = fdw_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1./m; # 1000mg =1g
    #         control = mg_gain*fdw_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("Drag acc x")
    #     elif i==1:
    #         plt.ylabel("Drag acc y")
    #     elif i==2:
    #         plt.ylabel("Drag acc z")
    # plt.xlabel("Time (s)")
    # fig = plt.figure(7).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(3, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1./m; #  1cm = 0.01m
    #         control = dfdw_all[i,:]*thrust_mg_gain;
    #         compare = d_fdw_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1./m; # 1000mg =1g
    #         control = mg_gain*dfdw_all[i,:];
    #         compare = mg_gain*d_fdw_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control, color='blue')
    #     plt.plot(state_log.sample_times()*time_gain, compare, color='red')
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("derv Drag acc x")
    #     elif i==1:
    #         plt.ylabel("derv Drag acc y")
    #     elif i==2:
    #         plt.ylabel("derv Drag acc z")
    # plt.xlabel("Time (s)")
    # fig = plt.figure(8).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(3, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1.; #  1cm = 0.01m
    #         control = vd_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1.; # 1000mg =1g
    #         control = mg_gain*vd_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("acc x")
    #     elif i==1:
    #         plt.ylabel("acc y")
    #     elif i==2:
    #         plt.ylabel("acc z")
    # plt.xlabel("Time (s)")
    # fig = plt.figure(9).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(3, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1.; #  1cm = 0.01m
    #         control = vdd_all[i,:]*thrust_mg_gain;
    #         compare = d_vd_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1.; # 1000mg =1g
    #         control = mg_gain*vdd_all[i,:];
    #         compare = mg_gain*d_vd_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("jerk x")
    #     elif i==1:
    #         plt.ylabel("jerk y")
    #     elif i==2:
    #         plt.ylabel("jerk z")
    # plt.xlabel("Time (s)")
    # fig = plt.figure(10).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(3, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1./m; #  1cm = 0.01m
    #         control = ddfdw_all[i,:]*thrust_mg_gain;
    #         compare = d_dfdw_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1./m; # 1000mg =1g
    #         control = mg_gain*ddfdw_all[i,:];
    #         compare = mg_gain*d_dfdw_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.plot(state_log.sample_times()*time_gain, compare, color='red')
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("sec derv Drag acc x")
    #     elif i==1:
    #         plt.ylabel("sec derv Drag acc y")
    #     elif i==2:
    #         plt.ylabel("sec derv Drag acc z")
    # plt.xlabel("Time (s)")
    # fig = plt.figure(11).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(3, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1./m; #  1cm = 0.01m
    #         control = wdd_all[i,:]*thrust_mg_gain;
    #         compare = d_wd_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1./m; # 1000mg =1g
    #         control = mg_gain*wdd_all[i,:];
    #         compare = mg_gain*d_wd_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("sec derv ang vel x")
    #     elif i==1:
    #         plt.ylabel("sec derv ang vel y")
    #     elif i==2:
    #         plt.ylabel("sec derv ang vel z")
    # plt.xlabel("Time (s)")
    # fig = plt.figure(12).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(3, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1./m; #  1cm = 0.01m
    #         control = wd_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1./m; # 1000mg =1g
    #         control = mg_gain*wd_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
        
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("derv ang vel x")
    #     elif i==1:
    #         plt.ylabel("derv ang vel y")
    #     elif i==2:
    #         plt.ylabel("derv ang vel z")
    # plt.xlabel("Time (s)")
    # fig = plt.figure(13).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(4, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1.; #  1cm = 0.01m
    #         control = eta1_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1e0; # 1000mg =1g
    #         control = mg_gain*eta1_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("eta1 x")
    #     elif i==1:
    #         plt.ylabel("eta1 y")
    #     elif i==2:
    #         plt.ylabel("eta1 z")
    # plt.xlabel("Time (s)")
    # fig = plt.figure(14).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(3, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1.; #  1cm = 0.01m
    #         control = eta2_all[i,:]*thrust_mg_gain;
    #         compare = d_eta1_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1e0; # 1000mg =1g
    #         control = mg_gain*eta2_all[i,:];
    #         compare = mg_gain*d_eta1_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.plot(state_log.sample_times()*time_gain, compare, color='red')
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("eta2 x")
    #     elif i==1:
    #         plt.ylabel("eta2 y")
    #     elif i==2:
    #         plt.ylabel("eta2 z")
    # plt.xlabel("Time (s)")
    # fig = plt.figure(15).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(3, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1.; #  1cm = 0.01m
    #         control = eta3_all[i,:]*thrust_mg_gain;
    #         compare = d_eta2_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1e0; # 1000mg =1g
    #         control = mg_gain*eta3_all[i,:];
    #         compare = mg_gain*d_eta2_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("eta3 x")
    #     elif i==1:
    #         plt.ylabel("eta3 y")
    #     elif i==2:
    #         plt.ylabel("eta3 z")
    # plt.xlabel("Time (s)")
    # fig = plt.figure(16).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(3, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1.; #  1cm = 0.01m
    #         control = eta4_all[i,:]*thrust_mg_gain;
    #         compare = d_eta3_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1e0; # 1000mg =1g
    #         control = mg_gain*eta4_all[i,:];
    #         compare = mg_gain*d_eta3_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("eta4 x")
    #     elif i==1:
    #         plt.ylabel("eta4 y")
    #     elif i==2:
    #         plt.ylabel("eta4 z")
    # plt.xlabel("Time (s)")
    # fig = plt.figure(17).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(3, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1.; #  1cm = 0.01m
    #         control = deta4_all[i,:]*thrust_mg_gain;
    #         compare = d_eta4_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1e0; # 1000mg =1g
    #         control = mg_gain*deta4_all[i,:];
    #         compare = mg_gain*d_eta4_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.plot(state_log.sample_times()*time_gain, compare, color='red') 
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("deta4 x")
    #     elif i==1:
    #         plt.ylabel("deta4 y")
    #     elif i==2:
    #         plt.ylabel("deta4 z")
    # plt.xlabel("Time (s)")
#     fig = plt.figure(18).set_size_inches(6, 6)
#     for i in range(0,3):
#         # print("i:%d" %i)
#         plt.subplot(3, 1, i+1)
#         # print("test:", num_state)
#         if i==0:
#             thrust_mg_gain=1.; #  1cm = 0.01m
#             control = eta5_all[i,:]*thrust_mg_gain;
#         else:
#             mg_gain=1e0; # 1000mg =1g
#             control = mg_gain*eta5_all[i,:];

#         plt.plot(state_log.sample_times()*time_gain, control)
        
#         plt.grid(True)
#         # plt.ylabel("x[%d]" % j)
#         if i==0:
#             plt.ylabel("eta5 x")
#         elif i==1:
#             plt.ylabel("eta5 y")
#         elif i==2:
#             plt.ylabel("eta5 z")
#     plt.xlabel("Time (s)")
#     fig = plt.figure(19).set_size_inches(6, 6)
#     for i in range(0,3):
#         # print("i:%d" %i)
#         plt.subplot(3, 1, i+1)
#         # print("test:", num_state)
#         if i==0:
#             thrust_mg_gain=1.; #  1cm = 0.01m
#             control = eta6_all[i,:]*thrust_mg_gain;
#             compare = d_eta5_all[i,:]*thrust_mg_gain;
#         else:
#             mg_gain=1e0; # 1000mg =1g
#             control = mg_gain*eta6_all[i,:];
#             compare = mg_gain*d_eta5_all[i,:];

#         plt.plot(state_log.sample_times()*time_gain, control)
#         plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        
#         plt.grid(True)
#         # plt.ylabel("x[%d]" % j)
#         if i==0:
#             plt.ylabel("eta6 x")
#         elif i==1:
#             plt.ylabel("eta6 y")
#         elif i==2:
#             plt.ylabel("eta6 z")
#     plt.xlabel("Time (s)")
#     fig = plt.figure(20).set_size_inches(6, 6)
#     for i in range(0,3):
#         # print("i:%d" %i)
#         plt.subplot(3, 1, i+1)
#         # print("test:", num_state)
#         if i==0:
#             thrust_mg_gain=1.; #  1cm = 0.01m
#             control = eta7_all[i,:]*thrust_mg_gain;
#             compare = d_eta6_all[i,:]*thrust_mg_gain;
#         else:
#             mg_gain=1e0; # 1000mg =1g
#             control = mg_gain*eta7_all[i,:];
#             compare = mg_gain*d_eta6_all[i,:];

#         plt.plot(state_log.sample_times()*time_gain, control)
#         plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        
#         plt.grid(True)
#         # plt.ylabel("x[%d]" % j)
#         if i==0:
#             plt.ylabel("eta7 x")
#         elif i==1:
#             plt.ylabel("eta7 y")
#         elif i==2:
#             plt.ylabel("eta7 z")
#     plt.xlabel("Time (s)")
    # fig = plt.figure(23).set_size_inches(6, 6)
    # for i in range(0,3):
    #     # print("i:%d" %i)
    #     plt.subplot(3, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1.; #  1cm = 0.01m
    #         control = deta7_all[i,:]*thrust_mg_gain;
    #         compare = d_eta7_all[i,:]*thrust_mg_gain;
    #     else:
    #         mg_gain=1e0; # 1000mg =1g
    #         control = mg_gain*deta7_all[i,:];
    #         compare = mg_gain*d_eta7_all[i,:];

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("deta7 x")
    #     elif i==1:
    #         plt.ylabel("deta7 y")
    #     elif i==2:
    #         plt.ylabel("deta7 z")
    # plt.xlabel("Time (s)")

    # fig = plt.figure(22).set_size_inches(6, 6)
    # for i in range(0,2):
    #     # print("i:%d" %i)
    #     plt.subplot(2, 1, i+1)
    #     # print("test:", num_state)
    #     if i==0:
    #         thrust_mg_gain=1.; #  1cm = 0.01m
    #         control = alpha1_all*thrust_mg_gain;
    #     else:
    #         mg_gain=1e0; # 1000mg =1g
    #         control = mg_gain*dalpha1_all;

    #     plt.plot(state_log.sample_times()*time_gain, control)
    #     plt.grid(True)
    #     # plt.ylabel("x[%d]" % j)
    #     if i==0:
    #         plt.ylabel("alpha1")
    #     elif i==1:
    #         plt.ylabel("dalpha1")
    #         compare = mg_gain*d_alpha1_all;
    #         plt.plot(state_log.sample_times()*time_gain, compare, color='red')
    #     elif i==2:
    #         plt.ylabel("deta7 z")
    # plt.xlabel("Time (s)")

#     # fig = plt.figure(23).set_size_inches(6, 6)
#     # for i in range(0,3):
#     #     # print("i:%d" %i)
#     #     plt.subplot(3, 1, i+1)
#     #     # print("test:", num_state)
#     #     if i==0:
#     #         thrust_mg_gain=1.; #  1cm = 0.01m
#     #         control = alpha_all[i,:]*thrust_mg_gain;
#     #     else:
#     #         mg_gain=1e0; # 1000mg =1g
#     #         control = mg_gain*alpha_all[i,:];

#     #     plt.plot(state_log.sample_times()*time_gain, control)
#     #     plt.grid(True)
#     #     # plt.ylabel("x[%d]" % j)
#     #     if i==0:
#     #         plt.ylabel("alpha x")
#     #     elif i==1:
#     #         plt.ylabel("alpha y")
#     #     elif i==2:
#     #         plt.ylabel("alpha z")
#     plt.xlabel("Time (s)")
#     fig = plt.figure(24).set_size_inches(6, 6)
#     for i in range(0,3):
#         # print("i:%d" %i)
#         plt.subplot(3, 1, i+1)
#         # print("test:", num_state)
#         if i==0:
#             thrust_mg_gain=1.; #  1cm = 0.01m
#             control = dalpha_all[i,:]*thrust_mg_gain;
#             compare = d_alpha_all[i,:]*thrust_mg_gain;
#         else:
#             mg_gain=1e0; # 1000mg =1g
#             control = mg_gain*dalpha_all[i,:];
#             compare = mg_gain*d_alpha_all[i,:];

#         plt.plot(state_log.sample_times()*time_gain, control)
#         plt.plot(state_log.sample_times()*time_gain, compare, color='red')
#         plt.grid(True)
#         # plt.ylabel("x[%d]" % j)
#         if i==0:
#             plt.ylabel("dalpha_all x")
#         elif i==1:
#             plt.ylabel("dalpha_all y")
#         elif i==2:
#             plt.ylabel("dalpha_all z")
#     plt.xlabel("Time (s)")
#     fig = plt.figure(25).set_size_inches(6, 6)
#     for i in range(0,2):
#         # print("i:%d" %i)
#         plt.subplot(2, 1, i+1)
#         # print("test:", num_state)
#         if i==0:
#             thrust_mg_gain=1.; #  1cm = 0.01m
#             control = xi2_all*thrust_mg_gain;
#             compare = d_xi1_all*thrust_mg_gain;
#         else:
#             mg_gain=1e0; # 1000mg =1g
#             control = mg_gain*ubar[0,:];
#             compare = mg_gain*d_xi2_all;

#         plt.plot(state_log.sample_times()*time_gain, control)
#         plt.plot(state_log.sample_times()*time_gain, compare, color='red')
#         plt.grid(True)
#         # plt.ylabel("x[%d]" % j)
#         if i==0:
#             plt.ylabel("dxi1")
#         elif i==1:
#             plt.ylabel("dxi2")
#     plt.xlabel("Time (s)")

    fig = plt.figure(26).set_size_inches(6, 6)
    for i in range(0,6):
        # print("i:%d" %i)
        plt.subplot(6, 1, i+1)
        # print("test:", num_state)
        if i==0:
            thrust_mg_gain=1.; #  1cm = 0.01m
            control = eta_norm_all*thrust_mg_gain;

        elif i==1:
            mg_gain=1e0; # 1000mg =1g
            control = mg_gain*Vx_all;
        elif i==2:
            control = Kinetic_all;
            compare = Kinetic_all_compare;
            # plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        elif i==3:
            control = V1_lyapu;
        elif i==4:
            control = 1/1.*V2_lyapu;
            compare = 1/1.*V3_lyapu;
            # plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        elif i==5:
            control = d_Kinetic_all;
            compare = vd_Kinetic_all;
            plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        plt.plot(state_log.sample_times()*time_gain, control)
        plt.grid(True)
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("Total energy")
        elif i==1:
            plt.ylabel("Vx")
        elif i==2:
            plt.ylabel("Kinetic energy")
        elif i==3:
            plt.ylabel("V1")
        elif i==4:
            plt.ylabel("V2")
        elif i==5:
            plt.ylabel("derivative of Kinetic energy")
    plt.xlabel("Time (s)")

    fig = plt.figure(27).set_size_inches(6, 6)
    for i in range(0,6):
        # print("i:%d" %i)
        plt.subplot(6, 1, i+1)
        # print("test:", num_state)
        if i==0:
            thrust_mg_gain=1.; #  1cm = 0.01m
            control = Total_energy;
            compare = Work_done;
            compare2 = Conservation_energy;
            plt.plot(state_log.sample_times()*time_gain, compare, color='red')
            plt.plot(state_log.sample_times()*time_gain, compare2, color='green')
        elif i==1:
            mg_gain=1e0; # 1000mg =1g
            control = Angular_momentum;
        elif i==2:
            control = Kinetic_all;
            compare = Kinetic_all_compare;
            # plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        elif i==3:
            control = V1_lyapu;
        elif i==4:
            control = 1/1.*V2_lyapu;
            compare = 1/1.*V3_lyapu;
            # plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        elif i==5:
            control = vd_Kinetic_all;
            # compare = vd_Kinetic_all;
            # plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        plt.plot(state_log.sample_times()*time_gain, control)
        plt.grid(True)
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("Total energy")
        elif i==1:
            plt.ylabel("Angular_momentum")
        elif i==2:
            plt.ylabel("Lyapunov")
        elif i==3:
            plt.ylabel("1/2v^2")
        elif i==4:
            plt.ylabel("1/2wIW")
        elif i==5:
            plt.ylabel("derivative of Lyapu")
    plt.xlabel("Time (s)")

plt.grid(True)
plt.show()
