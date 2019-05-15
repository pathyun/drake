# Testing Python 
import math
import numpy as np
# from sympy import *

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    SignalLogger,
    VectorSystem,
    RotationMatrix,
    Quaternion,
    RollPitchYaw
    )
from pydrake.all import MathematicalProgram
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.gurobi import GurobiSolver

##################### 
# Quaternion test
#####################

q = np.zeros((4,1)) # quaternion setup
theta = math.pi/2;  # angle of rotation
v=np.array([0.,1.,0])
q[0]=math.cos(theta/2) #q0
# print("q:",q[0])
v_norm=np.sqrt((np.dot(v,v))) # axis of rotation
v_normalized =v.T/v_norm
# print("vnorm:", v_norm)
# print("vnormalized", np.dot(v_normalized,v_normalized.T))
q[1:4,0]=math.sin(theta/2)*v.T/v_norm
print("q: ", q)
# print("q_norm: ", np.dot(q.T,q))

quat = Quaternion(q)

# unit quaternion to Rotation matrix
R = RotationMatrix(quat)
print("R:", R.matrix())
rpy=RollPitchYaw(R)
print("rpy",rpy.vector())
# print(R.matrix().dtype) # data type of Rotation matrix
# print(np.dot(R.matrix().T,R.matrix()))

# R\in SO(3) check
Iden = np.dot(R.matrix().T,R.matrix())
print("R^TR=:",Iden)

# get quaternion from rotationmatrix
quat_from_rot = RotationMatrix.ToQuaternion(R)
# print("quaternion_from_rotmax:", quat_from_rot.wxyz())

# Get E(q) matrix given unit quaternion q

E = np.zeros((3,4))
E[0,:] = np.array([-1*q[1],    q[0], -1*q[3],    q[2]]).T
E[1,:] = np.array([-1*q[2],    q[3],    q[0], -1*q[1]]).T
E[2,:] = np.array([-1*q[3], -1*q[2],    q[1],    q[0]]).T

print("E(q)^T:", E.T)

# compute quaternion derivative
w= np.zeros((3,1))
w[0]=1.
print("w:", w)
quat_dot = np.dot(E.T,w)/2.   # \dot{quat}=1/2*E(q)^Tw
print("quat_dot:", quat_dot)

##################### 
# Euler equation test
#####################

w=np.zeros((3,1))  		# angular velocity
tau =np.zeros((3,1))	# torque
w[0]=1;
w[1]=2;
w[2]=3;
tau[0] =4;
tau[1] = 5;
tau[2] = 6;

I=np.zeros((3,3)) 		# Intertia matrix
Ixx= 1;
Iyy= 2;
Izz= 3;

I[0,0]=Ixx;
I[1,1]=Iyy;
I[2,2]=Izz;
I_inv = np.linalg.inv(I);

print("I:",I)
print("I^-1",I_inv)
Iw = np.dot(I,w)
print("Iw", Iw)
what = np.cross(np.eye(3),w.T) # Caution, hat operation needs to be flipped. eye(3) first in the argument.

print("what:",what)

wIw_w_hat = np.dot(what,Iw)     # w^Iw
wIw_wo_hat = np.cross(w.T,Iw.T) # w x (Iw)

print("wIw_w_hat:",wIw_w_hat)
print("wIw_wo_hat:", wIw_wo_hat)

w_dot = -np.dot(I_inv,wIw_w_hat)+np.dot(I_inv,tau) # \dot{w}=-I^{-1} w x (Iw) + I^{-1] tau}
print("w_dot:",w_dot)

####################################
# Symbolic test
####################################

# xSym = Symbol('x')
# ySym = Symbol('y')

####################################
# Mathematical programming test
####################################

# Set up for QP problem

prog = MathematicalProgram()
u_var = prog.NewContinuousVariables(4, "u_var")
solverid = prog.GetSolverId()

tol = 1e-6
prog.SetSolverOption(mp.SolverType.kIpopt,"tol", tol);
prog.SetSolverOption(mp.SolverType.kIpopt,"constr_viol_tol", tol);
prog.SetSolverOption(mp.SolverType.kIpopt,"acceptable_tol", tol);
prog.SetSolverOption(mp.SolverType.kIpopt,"acceptable_constr_viol_tol", tol);

prog.SetSolverOption(mp.SolverType.kIpopt, "print_level", 2) # CAUTION: Assuming that solver used Ipopt

A_fl = np.eye(4)
# A_fl[0:3,0] = [1, 0, 0]
# A_fl[0:3,1:4] = -np.dot(Rqe3_hat,I_inv)*xi1
# A_fl[3,1:4]=g_yaw
A_fl = np.array([[ 1.9430e-02, -0.0000e+00,  7.8752e+03,  4.5375e+03],
       [-1.8993e-01, -7.4316e+03, -0.0000e+00,  4.6419e+02],
       [ 9.8161e-01, -1.4379e+03, -1.5589e+02, -0.0000e+00],
       [ 0.0000e+00,  4.6609e+01, -1.3138e+02,  2.2222e+03]])

A_fl_inv = np.linalg.inv(A_fl)
A_fl_det = np.linalg.det(A_fl)
# print("I_inv:", I_inv)
print("A_fl:", A_fl)
print("A_fl_det:", A_fl_det)

Quadratic_Positive_def = np.matmul(A_fl.T,A_fl)*1e0  # Q =A^TA 
QP_det = np.linalg.det(Quadratic_Positive_def)

L_fhx_star = np.zeros(4)
L_fhx_star[0]=1;
L_fhx_star[1]=1;
L_fhx_star[2]=1;
L_fhx_star[3]=1;

c = np.array([-8.3592e+00, -8.3708e+06, -5.1451e+05,  2.0752e+05])   # c

d = np.zeros(4)
d[0]=1;
d[1]=-1;
d[2]=1;
d[3]=-1;

d= np.array([5.0752e+01, 4.7343e+05, 8.4125e+05, 6.2668e+05])

phi0= -36332.36234347365;

print("Qp : ",Quadratic_Positive_def)
print("Qp det: ", QP_det)

# Quadratic cost : u^TQu + c^Tu
CLF_QP_cost_v_effective = np.dot(u_var, np.dot(Quadratic_Positive_def,u_var))+np.dot(c,u_var)

    
prog.AddQuadraticCost(CLF_QP_cost_v_effective)
prog.AddConstraint(np.dot(d,u_var)+phi0<=0)

solver = IpoptSolver()


prog.Solve()
# solver.Solve(prog)
print("Optimal u : ", prog.GetSolution(u_var))
u_CLF_QP = prog.GetSolution(u_var)

# ('A_fl:', )
# ('A_fl_det:', 137180180557.17741)
# ('Qp : ', array([[ 1.0000e+00, -1.5475e-13,  4.0035e-14,  3.7932e-15],
#        [-1.5475e-13,  5.7298e+07,  2.1803e+05, -3.3461e+06],
#        [ 4.0035e-14,  2.1803e+05,  6.2061e+07,  3.5442e+07],
#        [ 3.7932e-15, -3.3461e+06,  3.5442e+07,  2.5742e+07]]))
# ('Qp det: ', 1.8818401937699662e+22)
# ('c_QP', array([-8.3592e+00, -8.3708e+06, -5.1451e+05,  2.0752e+05]))
# ('phi0_exp: ', -36332.36234347365)
# ('d : ', array([5.0752e+01, 4.7343e+05, 8.4125e+05, 6.2668e+05]))
