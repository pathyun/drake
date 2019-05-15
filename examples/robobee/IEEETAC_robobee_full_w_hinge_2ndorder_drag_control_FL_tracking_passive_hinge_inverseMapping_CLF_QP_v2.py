##########################333
#
# Feedback_linearizaiton with Back Stepping method with quaternion including Aerodynamics drag
# 
# Script originated from robobee_drag_control_FL.py
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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy import signal

# mpl.font_manager._rebuild()


from scipy.linalg import solve_continuous_are
from pydrake.all import MathematicalProgram
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.gurobi import GurobiSolver
from pydrake.solvers.mosek import MosekSolver



from robobee_plant_full_w_hinge_2ndorder_aero_passive_hinge import *

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
mpl.rcParams['legend.fontsize'] = 20
# plt.rc('font',family='Times New Roman')

font = {'family' : 'times new roman',
        # 'weight' : 'bold',
        'size'   : 23}

mpl.rc('font', **font)

#-[0.0] Show figure flag
show_flag =1
show_flag_qd = 1;
show_flag_q=1;
show_flag_control =1;

# Run forward simulation from the specified initial condition
duration =2.5000
flag_FL_controller =3;
flag_discrete_control_500Hz=1;
# Reference trajectory # #- Tracking parameter

circle_flag = 0 #  0: circle 1: Lisssajous figure
time_gain = 1.0; # 10\tau = 1s 

T_period = 5*time_gain # 
w_freq = 2*math.pi/T_period
radius = 0.1 #.5
ratio_ab=2./3.;
delta_lissajous = math.pi/2;

# Default mean stroke angle
phi_default = 0.# math.pi/4. #math.pi/3
w_resonance = 2.*math.pi*170;

num_sample_cycle = 15;
time_step_cycle = 2*math.pi/w_resonance*1/num_sample_cycle

num_cycle_filter =2;
filter_window_size=num_sample_cycle*num_cycle_filter;
r_filtered =np.zeros((3,filter_window_size))
q_filtered =np.zeros((4,filter_window_size))
v_filtered =np.zeros((3,filter_window_size))
w_filtered =np.zeros((3,filter_window_size))
Thrust_filtered=np.zeros(filter_window_size)
tau_aero_filtered=np.zeros((3,filter_window_size))
# Wing hinge angle
psi_default = 0 #math.pi/3

# Average voltage 
V_l_avg = np.zeros(num_sample_cycle*3)
V_r_avg = np.zeros(num_sample_cycle*3)

u_sampled = np.zeros(6)
sample_flag=0
#-[0] Intialization
#-[0-1] Robobee params. From IROS 2015 S.Fuller "Rotating the heading angle of underactuated flapping-wing flyers by wriggle-steering"

# Pick mg unit
m   = 81        # 81 mg
Ixx = 1.42*1e-3      # 1.42 x 10^-3  mg m^2                     | 1.42 x 10 mg cm^2
Iyy = 1.34*1e-3      # 1.34 x 10^-3  mg m^2                     | 1.34 x 10 mg cm^2             
Izz = 0.45*1e-3      # 0.45 x 10^-3  mg m^2                     | 0.45 x 10 mg cm^2
g   = 9.80*1e0       # 9.8 m/s^2    use 1 \tau = 0.1s          | 9.8  x 1 cm/s^2
rw_l = 9.0*1e-3      # 7 x 10^-3 m
bw   = 2.*1e2       # 2 x 10^-4 mg / s


# Acutator and Transmission model (2nd order approximation)
m_a = 25e-6;
J_phi = 51.1e-12;
T = 2666;
r_cp = 1.42*9.56e-3;
b = 2.03e-6;
k_a = 300;
k_t = 28.2e-6;

m_eq = m_a+T*T*J_phi;
b_eq = T*T*r_cp*b;
k_eq = k_a+T*T*k_t;

sys = signal.TransferFunction([1], [m_eq, b_eq, k_eq])
w_array = np.linspace(0,3*w_resonance, 10000)
w_bode, mag_bode, phase_bode = signal.bode(sys,w_array)
mag_max = np.max(mag_bode)
mag_max_decimal = math.pow(10.,mag_max/20)

phi_max = 90.*math.pi/180.;   # ~90 deg peak-to-peak max (at resonance) expected at V_pp = 180 during normal operation
V_pp = 180.;
H_mag = 1/mag_max_decimal;
H_mag = H_mag*phi_max/2/V_pp;

w_r_zero, mag_r_zero, phase_r_zero = signal.bode(sys,w_resonance*0.) # DC Gain
w_r, mag_r, phase_r = signal.bode(sys,w_resonance)
w_r_half, mag_r_half, phase_r_half = signal.bode(sys,w_resonance/2.)
w_r_twice, mag_r_twice, phase_r_twice = signal.bode(sys,w_resonance*2.)
w_r_third, mag_r_third, phase_r_third = signal.bode(sys,w_resonance*3.)

print("mag_r_zero", mag_r_zero)
Transfer_mag_zero =  H_mag/k_eq # math.pow(10.,mag_r_zero/20)*H_mag # DC Gain in linear scale
Transfer_mag_reso =  math.pow(10.,mag_r/20)*H_mag
# print("mag_r_zerp -A :", Transfer_mag_zero-H_mag/k_eq)
Transfer_mag_reso_half =  math.pow(10.,mag_r_half/20)*H_mag
Transfer_mag_reso_twice =  math.pow(10.,mag_r_twice/20)*H_mag
Transfer_mag_reso_third =  math.pow(10.,mag_r_third/20)*H_mag

print('resonacne mag:', Transfer_mag_reso)
print('resonacne half mag:', Transfer_mag_reso_half)

# Aerodynamics and Wing parameter

rho = 1.2041;
Aw = 54.59; # 54.59 mm^2
beta_geometry = 2.0993e-9 #1.1462e-9 #2.0993e-9; from Kevin Influence of wing design paper.
C_R_chord = 7.0e-3
CL = 1.8;
CL0 =1.8;

CD0 = 0.4;
CDmax = 3.4;

wing_R = 12e-3
wing_xr = 3.74e-3

Force_scale = 1e-6

Ixx_wing = 1.54 # 1.54 mg mm^2
m_wing = 0.52 # mg
l_wing=0.52 # 0.52mm
k_wing=1.4/Force_scale # 1.4 mg m^2/s^2 / 1.4*1e6 mg mm^2/s^2

w_2_wing =0.43;
b_max = 1.25*3.4;
W_wing = 4.5; #4.5mm
W_cp1_wing = 0.38*1.93;

damp_wing = 4.*(0.5*rho*Aw*W_cp1_wing*math.pow(w_2_wing*W_wing,2)*b_max)

print("damp_wing:", damp_wing);

r_cp_wing = r_cp*1e3 # in mm scale

#update bw
bw=bw*0.62 #0.82


bw_gap = 0.01;
bw_max = bw*(1.+1.*bw_gap)
bw_min = bw*(1.-bw_gap)

# 100 blades
# chordlength_r = np.loadtxt('chordlength_list_r.txt')
# chordlength_cr = np.loadtxt('chordlength_list_cr.txt')
# chordlength_yLE = np.loadtxt('chordlength_list_yLE.txt')

# 50 blades
# chordlength_r = np.loadtxt('chordlength_list_r_50.txt')
# chordlength_cr = np.loadtxt('chordlength_list_cr_50.txt')
# chordlength_yLE = np.loadtxt('chordlength_list_yLE_50.txt')

# 30 blades
chordlength_r = np.loadtxt('/home/patrick/Research/drake/examples/robobee/chordlength_list_r_10.txt')
chordlength_cr = np.loadtxt('/home/patrick/Research/drake/examples/robobee/chordlength_list_cr_10.txt')
chordlength_yLE = np.loadtxt('/home/patrick/Research/drake/examples/robobee/chordlength_list_yLE_10.txt')


num_blade = np.size(chordlength_r)
blade_width = chordlength_r[1]-chordlength_r[0]

# Inverse mapping Gain

gain_amp = 0.82; #0.82
gain_offset=1.5; #3
gain_diff=0.3; #0.2

I = np.zeros((3,3))
I[0,0] = Ixx
I[1,1] = Iyy
I[2,2] = Izz

e3= np.array([0,0,1])
e2= np.array([0,1,0])
e1= np.array([1,0,0])
r_w = rw_l*e3; # Length to the wing from CoM

#-[0-2] Control set up

ratio_rw =3#3.0#7.0;
decaying_rate = 0.#1.5;
base_ratio =1.#3.

k_yaw = Izz;

radius_original = radius
# radius = radius -(base_ratio+0.2)*rw_l

#-[0] Initial condition

r0 = np.array([0.0,0.,0.05])
q0 = np.zeros((4)) # quaternion setup
theta0 = 0.*math.pi/4.;  # angle of rotation
# v0_q=np.array([1.,0.,1.])
v0_q=np.array([0.,0.,1.])
q0[0]=math.cos(theta0/2) #q0
# print("q:",q[0])
v0_norm=np.sqrt((np.dot(v0_q,v0_q))) # axis of rotation
v0_normalized =v0_q.T/v0_norm
# print("vnorm:", v_norm)
# print("vnormalized", np.dot(v_normalized,v_normalized.T))
q0[1:4]=math.sin(theta0/2)*v0_q.T/v0_norm

v0 = np.zeros((3))
w0 = np.zeros((3))          # angular velocity
v0[0] =0.0 #0.1;
v0[1] =-0.0 # -0.2;
v0[2] = 0. #-m*g/bw# 0.#10.0001-m*g/bw;
w0[0]=0.# 2.0 #-0.1 # 0.1 #0;
w0[1]=0.# -2.#10.5 # 0  #-0.1;s
w0[2]=0. #.1 # 0.1 # 0.2;

#-[0-1] Stack up the state in R^13
# print("r:", r0.shape, "q",q0.shape,"v",v0.shape,"w", w0.shape)
x0= np.hstack([r0, q0, phi_default, -phi_default, psi_default, -psi_default, v0, w0, 0, 0, 0, 0])
print("x0:", x0)


input_max = 1e22  # N m  
robobee_plant_aero = RobobeePlantAero(
    m = m, Ixx = Ixx, Iyy = Iyy, Izz = Izz, 
    g = g, rw_l= rw_l, bw=bw, m_eq=m_eq, b_eq=b_eq, k_eq=k_eq, H_mag=H_mag, k_wing=k_wing, Ixx_wing=Ixx_wing, m_wing=m_wing 
    ,l_wing=l_wing,r_cp_wing=r_cp_wing, damp_wing=damp_wing,  input_max = input_max)


#-[1] Fixed point for Linearization

F_T = 0;
tau = np.array([0,0,0]) 

rf = np.array([0.05,0.15,0.3])+rw_l*e3
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

# u =0 Backstepoped input

uf_T =0;

xf= np.hstack([rf, qf, phi_default, -phi_default, psi_default, psi_default, vf, wf, 0, 0, 0, 0]) # Fixed point for the state
print("xf:", xf)
uf = np.hstack([uf_T,0,0,tau,0,0,0,0])  # Hovering
print("uf:",uf)

xstackf=robobee_plant_aero.evaluate_f(uf,xf)
xstackf_norm = np.dot(xstackf,xstackf)
if xstackf_norm<1e-6:
    print("\n\n1. Set point is a fixed point")
else:
    print("\n\n1. Set point is not a fixed point")


#-----Reduced output dynamics LQR (3x4)

Q = 1e0*np.eye(6)
Q[0:3,0:3] = 4*1e2*np.eye(3) # eta1
Q[3:6,3:6] =2e2*np.eye(3) # eta2

kq =1e3

kr = 1e0
Rout = kr*np.eye(3)

Aout = np.zeros((6,6))
Aout[0:3,3:6]=np.eye(3)
    
Bout = np.zeros((6,3))
Bout[3:6,0:3]= np.eye(3)

# Observ_matrix=obsv(A, Q)

print("A:", Aout, "B:", Bout)
Mout = solve_continuous_are(Aout,Bout,Q,Rout)
R_inv = np.linalg.inv(Rout)
k = np.matmul(R_inv,np.matmul(Bout.T,Mout))
alpha_a = 2.*(1.+ratio_rw)*rw_l;

beta_a = bw/m;

BAB= np.zeros((6,6));
I6 = np.eye(6);
BBT = np.matmul(Bout,Bout.T)
I_P=(beta_a*I6-Mout/kr);
I_P_Rinv = (beta_a*Bout.T-k)/(2*(1+ratio_rw)*rw_l);
BAB = np.matmul(I_P,BBT)
BAB = np.matmul(BAB,I_P)/(alpha_a*alpha_a)*(1+ratio_rw)*m/(bw*ratio_rw);

BAB_new = ((1+ratio_rw)*m)/(bw*ratio_rw)*np.matmul(I_P_Rinv.T,I_P_Rinv)
print(BAB_new)
eval_BAB_new = np.linalg.eigvals(BAB_new)
max_eval_BAB_new = np.max(eval_BAB_new)
eval_Q = np.linalg.eigvals(Q)
min_e_Q = np.min(eval_Q)
kq = max_eval_BAB_new/min_e_Q*2.
print("kq", kq)

P_ISS = 1e0*np.eye(9)
P_ISS_reduced = 1e0*np.eye(8)
P_ISS[0:3,0:3] = bw*ratio_rw/((1+ratio_rw)*m)*np.eye(3)
P_ISS[3:9,3:9] = kq*Q;
P_ISS[0:3,3:9] =I_P_Rinv;
P_ISS[3:9,0:3] =I_P_Rinv.T;

P_ISS_reduced[0:2,0:2] = P_ISS[0:2,0:2]
P_ISS_reduced[2:8,2:8] = P_ISS[3:9,3:9]
P_ISS_reduced[0:2,2:8] = P_ISS[0:2,3:9]
P_ISS_reduced[2:8,0:2] = P_ISS[3:9,0:2]
eval_P_ISS = np.linalg.eigvals(P_ISS)
min_eval_P_ISS = np.min(eval_P_ISS)

# Different bound calculation

kq_new = max_eval_BAB_new/min_e_Q*3.
P_ISS_new = 1e0*np.eye(9)
P_ISS_new[0:3,0:3] = bw*ratio_rw/((1+ratio_rw)*m)*np.eye(3)
P_ISS_new[3:9,3:9] = kq_new*Q;
P_ISS_new[0:3,3:9] =I_P_Rinv;
P_ISS_new[3:9,0:3] =I_P_Rinv.T;

eval_P_ISS_new = np.linalg.eigvals(P_ISS_new)
min_eval_P_ISS_new = np.min(eval_P_ISS_new)

ratio_eval_P_ISS = min_eval_P_ISS/min_eval_P_ISS_new
####

max_eval_P_ISS = np.max(eval_P_ISS)
Q_BAB = Mout-BAB;

eval_Q_BAB = np.linalg.eigvals(Q_BAB)
min_e_Q_BAB = np.min(eval_Q_BAB)

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

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def test_controller(x, t):
    # This should return a 4x1 u that is bounded
    # between -input_max and input_max.
    # Remember to wrap the angular values back to
    # [-pi, pi].
    u = np.zeros(6)
    global g, xf, uf, K, e3
    
    q=np.zeros(7)
    qd=np.zeros(6)
    q=x[0:7]
    qd=x[7:13]


    print("qnorm:", np.dot(q[3:7],q[3:7]))
    (Rq, Eq, wIw, I_inv, fdw, taudw, vd_aero, wd_aero)=robobee_plant_aero.GetManipulatorDynamics(q, qd)
    

    print("wdaero- Iinv_Tau", wd_aero +wIw-np.dot(I_inv,taudw))
    q0=q[3]
    q1=q[4]
    q2=q[5]
    q3=q[6]
    
    r=q[0:3]
    v=qd[0:3]
    w=qd[3:6]


    u[0]=0
    # u[1]=0.
    # u[2]=-0.001
    # u[3]=0.01
    # u[1:4]= -(2.*1e-1)*w

    kd = 2*1e-1
    u[1:4] = 0 #-1*kd*w #-2.005*w[2]*e3
    u[0] = 0 #2*g

    return u


def test_harmonic_controller(x, t):
    # This should return a 4x1 u that is bounded
    # between -input_max and input_max.
    # Remember to wrap the angular values back to
    # [-pi, pi].
    u = np.zeros(8)
    global g, m, I, r_w, rw_l, bw, xf, Aout, Bout, Mout, T_period, w_freq, radius, ratio_ab, delta_lissajous, ratio_rw, circle_flag, w_resonance, u_sampled, rho, beta_geometry, CL, chordlength_r, chordlength_cr, blade_width, torque_flag, r_filtered, q_filtered, v_filtered, w_filtered

    ratio_rw_timevarying =base_ratio+ratio_rw*(math.exp(-decaying_rate*t))
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print("%% Aerodynamics Lift and Drag  %%%%%%%%%%%")
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    print("t:", t)
    prog = MathematicalProgram()
    u_var = prog.NewContinuousVariables(4, "u_var")
    
    zd=xf[2]
    # # # # Example 1 : Circle
    if circle_flag==0:
        x_f = radius*math.cos(w_freq*t)
        y_f = radius*math.sin(w_freq*t)
        z_f = zd#+1/4.*radius*math.sin(w_freq/2*t)
        # print("x_f:",x_f)
        # print("y_f:",y_f)
        dx_f = -radius*math.pow(w_freq,1)*math.sin(w_freq*t)
        dy_f = radius*math.pow(w_freq,1)*math.cos(w_freq*t)

        dz_f = 1/8.*math.pow(w_freq,1)*radius*math.cos(w_freq/2*t)
        ddx_f = -radius*math.pow(w_freq,2)*math.cos(w_freq*t)
        ddy_f = -radius*math.pow(w_freq,2)*math.sin(w_freq*t)
        ddz_f = -1/16.*radius*math.pow(w_freq,2)*math.sin(w_freq/2*t)
        dddx_f = radius*math.pow(w_freq,3)*math.sin(w_freq*t)
        dddy_f = -radius*math.pow(w_freq,3)*math.cos(w_freq*t)
        ddddx_f = radius*math.pow(w_freq,4)*math.cos(w_freq*t)
        ddddy_f = radius*math.pow(w_freq,4)*math.sin(w_freq*t)
    elif circle_flag==1:
        # Example 2 : Lissajous curve a=1 b=2
        a=1;
        b=ratio_ab*a;
        
        x_f = radius*math.sin(a*w_freq*t+delta_lissajous)
        y_f = radius*math.sin(b*w_freq*t)
        
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


    zd=xf[2]

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
    q=x[0:7]
    qd=x[7:13]


    phi_l=x[13]
    phi_r=x[14]
    psi_l=x[15]
    psi_r=x[16]

    # print("qnorm:", np.dot(q[3:7],q[3:7]))

    q0=q[3]
    q1=q[4]
    q2=q[5]
    q3=q[6]


    r=np.zeros((3,1))
    v=np.zeros((3,1))
    w=np.zeros((3,1))
    r=q[0:3]
    
    v=qd[0:3]
    v_vel=v; 

    w=qd[3:6]
    q_quat=q[3:7]
    
    dphi_l=x[17]
    dphi_r=x[18]

    # dpsi_l=x[19]
    # dpsi_r=x[20]

    # Filtered signal

    num_iter=math.ceil(t/time_step_cycle);
    M_mat_quat=np.zeros((4,4))
    # print("num_iter:", num_iter)
    # print("r_shape:", r.shape)
    if num_iter<filter_window_size:
        r_filtered=np.matmul(r.reshape(3,1),np.ones((1,filter_window_size)))
        q_filtered=np.dot(q_quat.reshape(4,1),np.ones((1,filter_window_size)))
        v_filtered=np.dot(v.reshape(3,1),np.ones((1,filter_window_size)))
        w_filtered=np.dot(w.reshape(3,1),np.ones((1,filter_window_size)))

        q_filtered_avg = q_quat
    else:
        r_filtered[:,0:filter_window_size-1]=r_filtered[:,1:filter_window_size]
        q_filtered[:,0:filter_window_size-1]=q_filtered[:,1:filter_window_size]
        v_filtered[:,0:filter_window_size-1]=v_filtered[:,1:filter_window_size]
        w_filtered[:,0:filter_window_size-1]=w_filtered[:,1:filter_window_size]

        r_filtered[:,filter_window_size-1]=r
        q_filtered[:,filter_window_size-1]=q_quat
        v_filtered[:,filter_window_size-1]=v
        w_filtered[:,filter_window_size-1]=w

        for j_iter in range(0,filter_window_size):
            M_mat_quat = M_mat_quat+np.matmul(q_filtered[:,j_iter].reshape(4,1),q_filtered[:,j_iter].reshape(1,4))
            # print("M_mat_quat:", M_mat_quat)
        # print("M_mat_quat:", M_mat_quat)
        w_eig, vr_eig = np.linalg.eig(M_mat_quat)
        # print("vr_eig:", vr_eig)
        # print("w_eig:", w_eig)
        
        q_filtered_avg = vr_eig[:, np.argmax(w_eig)]
    # Averaging filter
    r_filtered_avg= np.sum(r_filtered,1)/filter_window_size
    v_filtered_avg= np.sum(v_filtered,1)/filter_window_size
    w_filtered_avg= np.sum(w_filtered,1)/filter_window_size

    # Current wing velocity
    # Wing trajectory
    
    decay_rate_wing = -1e2;
    
    # Slow varing mean stroke angle
    mean_stroke_l = 0 #-math.pi/10*math.sin(w_resonance*0.05*t);
    mean_stroke_r = 0 #+math.pi/10*math.sin(w_resonance*0.05*t);

    # # # Lower frequency mode
    # mean_stroke_r = 0;
    # mean_stroke_l = 0;

    torque_flag =0;
    if torque_flag==1:
        mean_stroke_l = 0 #-math.pi/8 #-math.pi/10*math.sin(w_resonance*0.05*t);
        mean_stroke_r = 0 # math.pi/8 #+math.pi/10*math.sin(w_resonance*0.05*t);

        half_wave_weight = 0.0;
        wave_weight = 0.934;
        twice_wave_weight = 0.216;
        third_wave_weight = 0.047
        half_wave_freq = w_resonance/2.;
        twice_wave_freq = 2.*w_resonance;
        third_wave_freq = 3.*w_resonance;
        torque_async =math.pi*(1.0);
    else:
        # Slow varing mean stroke angle
        mean_stroke_l = -math.pi/15*(w_filtered_avg[1])*0.1 #-math.pi/15*math.sin(w_resonance*0.1*t);
        mean_stroke_r = math.pi/15*(w_filtered_avg[1])*0.1 #math.pi/15*math.sin(w_resonance*0.1*t);
        half_wave_weight = 0.0;
        wave_weight = 1.;
        twice_wave_weight = 0.0;
        third_wave_weight = 0.;
        half_wave_freq = 0.5*w_resonance;
        twice_wave_freq = 2.*w_resonance;
        third_wave_freq = 3.*w_resonance;
        torque_async =0.0;

    Phi_off_l = mean_stroke_l
    Phi_off_r = mean_stroke_r
    
    Phi_amp_l = -math.pi/8.5#-math.pi/8.5*(zd-r_filtered_avg[2])*0.+math.pi/8.5*(v_filtered_avg[2])*0+math.pi/8.5*(w_filtered_avg[0])*0.05;
    Phi_amp_r = math.pi/8.5#+math.pi/8.5*(zd-r_filtered_avg[2])*0.-math.pi/8.5*(v_filtered_avg[2])*0+math.pi/8.5*(w_filtered_avg[0])*0.05;
    
    if Phi_amp_l>=0:
        phi_amp_l=0;
    if Phi_amp_r<=0:
        phi_amp_r=0;
    
    V_l = 1/Transfer_mag_zero*Phi_off_l + Phi_amp_l*(wave_weight/Transfer_mag_reso*math.sin(w_resonance*t)             
            +half_wave_weight/Transfer_mag_reso_half*math.sin(half_wave_freq*t)              
            +twice_wave_weight/Transfer_mag_reso_twice*math.sin(twice_wave_freq*t)
            +third_wave_weight/Transfer_mag_reso_third*math.sin(third_wave_freq*t))

    V_r = 1/Transfer_mag_zero*Phi_off_r + Phi_amp_r*(wave_weight/Transfer_mag_reso*math.sin(w_resonance*t+torque_async)
            +half_wave_weight/Transfer_mag_reso_half*math.sin(half_wave_freq*t+torque_async) 
            +twice_wave_weight/Transfer_mag_reso_twice*math.sin(twice_wave_freq*t+torque_async)
            +third_wave_weight/Transfer_mag_reso_third*math.sin(third_wave_freq*t+torque_async))

    # print("V_l:", V_l)
    # print("phi_l:", phi_l)
    u[6] = V_l
    u[7] = V_r



    # dpsi_l = math.pi/4.0*math.pow(w_resonance/1.,1)*math.cos(w_resonance/1.*t)
    # dpsi_r = math.pi/4.0*math.pow(w_resonance/1.,1)*math.cos(w_resonance/1.*t+torque_async)
    # u[8] = k*dpsi_l
    # u[9] = dpsi_r
    # dpsi_l = math.pi/4.0*math.pow(w_resonance/1.,1)*math.cos(w_resonance/1.*t)
    # dpsi_r = math.pi/4.0*math.pow(w_resonance/1.,1)*math.cos(w_resonance/1.*t+torque_async)
    # u[8] = dpsi_l
    # u[9] = dpsi_r
    
    zd=xf[2]
    # xd=xf[0]
    # yd=xf[1]
    # wd=xf[11:14]

   
    # print("r_filtered_avg", r_filtered_avg)
        
    # print("r_filtered", r_filtered)
    # Useful vectors and matrices
    
    (Rq, Eq, wIw, I_inv, fdw, taudw, vd_aero, wd_aero)=robobee_plant_aero.GetManipulatorDynamics(q, qd)
    


    

    w_hat = np.zeros((3,3))
    w_hat[0,:] = np.array([     0,   -w[2],     w[1] ])
    w_hat[1,:] = np.array([  w[2],       0,    -w[0] ])
    w_hat[2,:] = np.array([ -w[1],    w[0],        0 ])

    Rqe3 = np.dot(Rq,e3)
    Rqe3_hat = np.zeros((3,3))
    Rqe3_hat[0,:] = np.array([        0,   -Rqe3[2],     Rqe3[1] ])
    Rqe3_hat[1,:] = np.array([  Rqe3[2],          0,    -Rqe3[0] ])
    Rqe3_hat[2,:] = np.array([ -Rqe3[1],    Rqe3[0],           0 ])
    
    Rq_l = np.zeros((3,3))
    Rq_l[0,:] = np.array([ math.cos(phi_l),   -math.sin(phi_l),     0 ])
    Rq_l[1,:] = np.array([ math.sin(phi_l),    math.cos(phi_l),     0 ])
    Rq_l[2,:] = np.array([               0,                  0,     1 ])
    Rq_le2 = np.dot(Rq_l,e2)
    RqRq_le2 = np.dot(Rq,Rq_le2)
    w_hatRq_le2 = np.dot(w_hat, Rq_le2)
    Rqw_hatRq_le2 = np.dot(Rq, w_hatRq_le2)

    # print("x:", x)
    # print("w_hatRq_le2:", w_hatRq_le2)

    Rq_r = np.zeros((3,3))
    Rq_r[0,:] = np.array([ math.cos(phi_r),   -math.sin(phi_r),     0 ])
    Rq_r[1,:] = np.array([ math.sin(phi_r),    math.cos(phi_r),     0 ])
    Rq_r[2,:] = np.array([               0,                  0,     1 ])
    Rq_re2 = np.dot(Rq_r,e2)
    RqRq_re2 = np.dot(Rq,Rq_re2)
    w_hatRq_re2 = np.dot(w_hat, Rq_re2)
    Rqw_hatRq_re2 = np.dot(Rq, w_hatRq_re2)

    dphi_l_hat = np.zeros((3,3))
    dphi_l_hat[0,:] = np.array([      0,   -dphi_l,     0 ])
    dphi_l_hat[1,:] = np.array([ dphi_l,         0,     0 ])
    dphi_l_hat[2,:] = np.array([      0,         0,     0 ])

    dphi_l_hate2 = np.dot(dphi_l_hat, e2)
    Rq_ldphi_l_hate2 = np.dot(Rq_l, dphi_l_hate2)
    RqRq_ldphi_l_hate2 = np.dot(Rq, Rq_ldphi_l_hate2)

    dphi_r_hat = np.zeros((3,3))
    dphi_r_hat[0,:] = np.array([      0,   -dphi_r,     0 ])
    dphi_r_hat[1,:] = np.array([ dphi_r,         0,     0 ])
    dphi_r_hat[2,:] = np.array([      0,         0,     0 ])
    
    dphi_r_hate2 = np.dot(dphi_r_hat, e2)
    Rq_rdphi_r_hate2 = np.dot(Rq_r, dphi_r_hate2)
    RqRq_rdphi_r_hate2 = np.dot(Rq, Rq_rdphi_r_hate2)

    # print("dphi_r_hat:", dphi_r_hat)

    Rq_psi_l = np.zeros((3,3))
    Rq_psi_l[0,:] = np.array([  math.cos(psi_l),   0, math.sin(psi_l)])
    Rq_psi_l[1,:] = np.array([                0,   0,  0             ])
    Rq_psi_l[2,:] = np.array([ -math.sin(psi_l),   0, math.cos(psi_l)])

    Rq_psi_r = np.zeros((3,3))
    Rq_psi_r[0,:] = np.array([  math.cos(psi_r),   0, math.sin(psi_r)])
    Rq_psi_r[1,:] = np.array([                0,   0,  0             ])
    Rq_psi_r[2,:] = np.array([ -math.sin(psi_r),   0, math.cos(psi_r)])

    Iw = np.dot(I,w)
    r_w = rw_l*e3; # Length to the wing from CoM
    wr_w = np.cross(w,r_w) # w x r
    wr_w_hat = np.dot(w_hat,r_w)
    # print("wr_w-wr_w_hat", wr_w-wr_w_hat)

    wwr_w = np.cross(w,wr_w) # w x (w x r)

    # Wing angle

    eta = np.zeros(3)
    eta_l = np.zeros(3)
    eta_r = np.zeros(3)    

    eta = r+Rqe3*rw_l;
    eta_l = eta+np.dot(Rq,Rq_le2)*r_cp  # phi_l positive : clockwise
    eta_r = eta-np.dot(Rq,Rq_re2)*r_cp # phi_r positive : clockwise

    # Wing velocity

    deta = np.zeros(3)
    deta_l = np.zeros(3)
    deta_r = np.zeros(3)    

    deta = v + np.dot(Rq,wr_w)
    deta_l = deta + Rqw_hatRq_le2*r_cp + RqRq_ldphi_l_hate2*r_cp 
    deta_r = deta - Rqw_hatRq_re2*r_cp - RqRq_rdphi_r_hate2*r_cp 

    
    # Projection to wing stroke plane
    P_proj_wing_l = np.zeros((3,3))
    P_proj_wing_l = np.eye(3)-np.outer(RqRq_le2, RqRq_le2)
    # print("outer_before:", RqRq_le2)
    # print("outer:", P_proj_wing_l)
    P_proj_wing_r = np.zeros((3,3))
    P_proj_wing_r = np.eye(3)-np.outer(RqRq_re2, RqRq_re2)

    ## Decomposing the wing velocity in the projected domain
    
    proj_deta_l_body                = np.dot(P_proj_wing_l,deta)
    proj_deta_l_body_wing_couple    = np.dot(P_proj_wing_l,Rqw_hatRq_le2)
    proj_deta_l_wing                = np.dot(P_proj_wing_l,RqRq_ldphi_l_hate2)

    proj_deta_r_body                =  np.dot(P_proj_wing_r,deta)
    proj_deta_r_body_wing_couple    = -np.dot(P_proj_wing_r,Rqw_hatRq_re2)
    proj_deta_r_wing                = -np.dot(P_proj_wing_r,RqRq_rdphi_r_hate2)

    # nominal velocity in wing stroke plane for CoP
    # print("deta :", deta)
    # print("deta_l :", deta_l)
    # print("deta_r :", deta_r)
    v_ortho_l = np.dot(P_proj_wing_l, deta_l)
    v_ortho_r = np.dot(P_proj_wing_r, deta_r) 
    
    v_ortho_l_norm_sq = np.dot(v_ortho_l,v_ortho_l)
    v_ortho_r_norm_sq = np.dot(v_ortho_r,v_ortho_r)

    v_ortho_l_norm = math.sqrt(v_ortho_l_norm_sq)
    v_ortho_r_norm = math.sqrt(v_ortho_r_norm_sq)

    if v_ortho_l_norm==0 or v_ortho_r_norm==0 :
        v_ortho_l_normalized = np.zeros(3)
        v_ortho_r_normalized = np.zeros(3)
    else:
        v_ortho_l_normalized = v_ortho_l/v_ortho_l_norm
        v_ortho_r_normalized = v_ortho_r/v_ortho_r_norm 

    # print("v_ortho_l_normalized :", v_ortho_l_normalized)
    # print("v_ortho_r_normalized :", v_ortho_r_normalized)   
    # print("v_ortho_l_norm_sq :", v_ortho_l_norm_sq)
    # print("v_ortho_r_norm_sq :", v_ortho_r_norm_sq)   

    # Hinge angle in the global frame

    psi_l_wing = np.dot(Rq_psi_l,-e3) 
    psi_l_body = np.dot(Rq_l,psi_l_wing) 
    psi_l_world = np.dot(Rq,psi_l_body) 

    psi_r_wing = np.dot(Rq_psi_r,-e3) 
    psi_r_body = np.dot(Rq_r,psi_r_wing) 
    psi_r_world = np.dot(Rq,psi_r_body) 


    AoA_inner_product_l = np.dot(v_ortho_l_normalized, -psi_l_world) # Based on Kevin Science Robobtics supplement paper
    AoA_inner_product_r = np.dot(v_ortho_r_normalized, -psi_r_world)
    
    AoA_l = math.acos(AoA_inner_product_l)
    AoA_r = math.acos(AoA_inner_product_r)
    # print("AoA_l:", AoA_l)
    # print("AoA_r:", AoA_r)
    # nominal velocity in wing stroke plane for each blade

    FL_l_r=np.zeros(num_blade)
    FL_r_r=np.zeros(num_blade)
    FD_l_r = np.zeros((3,num_blade))
    FD_r_r = np.zeros((3,num_blade))
    FL_l_vec=np.zeros((3,num_blade))
    FL_r_vec=np.zeros((3,num_blade))
    FD_l_vec=np.zeros((3,num_blade))
    FD_r_vec=np.zeros((3,num_blade))
    F_l_vec=np.zeros((3,num_blade))
    F_r_vec=np.zeros((3,num_blade))
    tau_aero_l =np.zeros((3,num_blade))
    tau_aero_r =np.zeros((3,num_blade))

    chord_r_CoP_z_l=np.zeros(num_blade)
    chord_r_CoP_z_r=np.zeros(num_blade)
    
    chord_r_CoP_world_l=np.zeros((3,num_blade))
    chord_r_CoP_world_r=np.zeros((3,num_blade))



    for i_iter in range(0,num_blade):
        chord_r_from_root = wing_xr + chordlength_r[i_iter]

        chord_r_from_root_world_l=Rqe3*rw_l+np.dot(Rq,Rq_le2)*chord_r_from_root
        chord_r_from_root_world_r=Rqe3*rw_l-np.dot(Rq,Rq_re2)*chord_r_from_root

        v_ortho_l_r =  proj_deta_l_body + (proj_deta_l_body_wing_couple + proj_deta_l_wing)*chord_r_from_root
        v_ortho_r_r =  proj_deta_r_body + (proj_deta_r_body_wing_couple + proj_deta_r_wing)*chord_r_from_root

        v_ortho_l_r_norm_sq = np.dot(v_ortho_l_r, v_ortho_l_r)
        v_ortho_r_r_norm_sq = np.dot(v_ortho_r_r, v_ortho_r_r)

        # print("v_ortho_r_r_norm_sq", v_ortho_r_r_norm_sq )
        # print("v_ortho_l_r_norm_sq", v_ortho_l_r_norm_sq )

        v_ortho_l_r_norm = math.sqrt(v_ortho_l_r_norm_sq)
        v_ortho_r_r_norm = math.sqrt(v_ortho_r_r_norm_sq)

        if v_ortho_l_r_norm==0 or v_ortho_r_r_norm==0 :
            v_ortho_l_r_normalized = np.zeros(3)
            v_ortho_r_r_normalized = np.zeros(3)
        else:
            v_ortho_l_r_normalized = v_ortho_l_r/v_ortho_l_r_norm
            v_ortho_r_r_normalized = v_ortho_r_r/v_ortho_r_r_norm 

        AoA_inner_product_l_r = np.dot( v_ortho_l_r_normalized, -psi_l_world) # pointing downward : c vector
        AoA_inner_product_r_r = np.dot( v_ortho_r_r_normalized, -psi_r_world) # pointing downward : c vector

        AoA_l_r = math.acos(AoA_inner_product_l_r)
        AoA_r_r = math.acos(AoA_inner_product_r_r)

        # Center of Pressure for each blade : Kevin IROS 2016
        chord_r_CoP_z_l[i_iter] = chordlength_yLE[i_iter]-(0.25 + 0.25/(1+math.exp(5.*(1.-4.*AoA_l_r/math.pi))))*chordlength_cr[i_iter]
        chord_r_CoP_z_r[i_iter] = chordlength_yLE[i_iter]-(0.25 + 0.25/(1+math.exp(5.*(1.-4.*AoA_r_r/math.pi))))*chordlength_cr[i_iter]

        chord_r_CoP_world_l[0:3,i_iter] = chord_r_from_root_world_l+psi_l_world*chord_r_CoP_z_l[i_iter]
        chord_r_CoP_world_r[0:3,i_iter] = chord_r_from_root_world_r+psi_r_world*chord_r_CoP_z_r[i_iter]

        # print("Psi ", psi_l)
        # print("AoA_l_r", AoA_l_r)
        CL_l = CL0*math.sin(2.*AoA_l_r)
        CL_r = CL0*math.sin(2.*AoA_r_r)

        CD_l = (CD0+CDmax)/2.- (CDmax-CD0)/2.*math.cos(2.*AoA_l_r)
        CD_r = (CD0+CDmax)/2.- (CDmax-CD0)/2.*math.cos(2.*AoA_r_r)

        # print("CL_l:", CL_l)

        # print("CD_l:", CD_l)

        FL_l_r[i_iter] = 1./2.*rho*CL_l*v_ortho_l_r_norm_sq*chordlength_cr[i_iter]*blade_width
        FL_r_r[i_iter] = 1./2.*rho*CL_r*v_ortho_r_r_norm_sq*chordlength_cr[i_iter]*blade_width

        # print("FL_l_r", FL_l_r[i_iter])

        # print("FL_r_r", FL_r_r[i_iter])

        FD_l_r[0:3,i_iter] = -1./2.*rho*CD_l*v_ortho_l_r_norm_sq*v_ortho_l_r_normalized*chordlength_cr[i_iter]*blade_width
        FD_r_r[0:3,i_iter] = -1./2.*rho*CD_r*v_ortho_r_r_norm_sq*v_ortho_r_r_normalized*chordlength_cr[i_iter]*blade_width
        
        FL_l_vec[0:3,i_iter] = FL_l_r[i_iter]*np.dot(Rq,e3)/Force_scale
        FL_r_vec[0:3,i_iter] = FL_r_r[i_iter]*np.dot(Rq,e3)/Force_scale
        FD_l_vec[0:3,i_iter] = FD_l_r[0:3,i_iter]/Force_scale
        FD_r_vec[0:3,i_iter] = FD_r_r[0:3,i_iter]/Force_scale
        
        # print("FL_l_vec in body:", np.dot(Rq.T,FL_l_vec[0:3,i_iter]))
        # print("FD_l_vec in body:", np.dot(Rq.T,FD_l_vec[0:3,i_iter]))


        F_l_vec[0:3,i_iter] = FL_l_vec[0:3,i_iter] + FD_l_vec[0:3,i_iter]
        F_r_vec[0:3,i_iter] = FL_r_vec[0:3,i_iter] + FD_r_vec[0:3,i_iter]

        tau_aero_l[0:3,i_iter] = np.cross(chord_r_CoP_world_l[0:3,i_iter],F_l_vec[0:3,i_iter])
        tau_aero_l[0:3,i_iter] = np.dot(Rq.T,tau_aero_l[0:3,i_iter])

        tau_aero_r[0:3,i_iter] = np.cross(chord_r_CoP_world_r[0:3,i_iter],F_r_vec[0:3,i_iter])
        tau_aero_r[0:3,i_iter] = np.dot(Rq.T,tau_aero_r[0:3,i_iter])

    # print("CL_l:", CL_l)  

    # print("CD_l:", CD_l)    

    # FL_l = 1./2.*rho*beta_geometry/(r_cp*r_cp)*CL*v_ortho_l_norm_sq
    # FL_r = 1./2.*rho*beta_geometry/(r_cp*r_cp)*CL*v_ortho_r_norm_sq
    
    FL_l = np.sum(FL_l_r)
    FL_r = np.sum(FL_r_r)
    FD_l = np.sum(FD_l_r)
    FD_r = np.sum(FD_r_r)


    # print("FL_l:", FL_l)
    # print("FD_l:", FD_l)
    
    # Thrust_l = FL_l/(m*Force_scale)
    # Thrust_r = FL_r/(m*Force_scale)
    # Thrust = Thrust_l+Thrust_r

    # print("Thrust left:", Thrust_l)
    # print("Thrust right:", Thrust_r)
    # print("Thrust:", Thrust)

    # FD_l = np.zeros(3)
    # FD_r = np.zeros(3)

    # # Need to find correct c(r) to have better estimate for beta
    # FD_l_vec = 1./2.*rho*beta_geometry/(r_cp*r_cp)*CL*v_ortho_l_norm_sq*v_ortho_l_normalized/Force_scale
    # FD_r_vec = 1./2.*rho*beta_geometry/(r_cp*r_cp)*CL*v_ortho_r_norm_sq*v_ortho_r_normalized/Force_scale

    # # Exact Drag Force 

    # FD_l_vec = np.sum(FD_l_r,axis=1)/Force_scale
    # FD_r_vec = np.sum(FD_r_r,axis=1)/Force_scale

    # print("FD_l_vec", FD_l_vec)
    # FL_l_vec = FL_l*np.dot(Rq,e3)/Force_scale
    # FL_r_vec = FL_r*np.dot(Rq,e3)/Force_scale

    # F_l_vec = FL_l_vec + FD_l_vec
    # F_r_vec = FL_r_vec + FD_r_vec

    # tau_aero_l = np.cross(eta_l,F_l_vec)
    # tau_aero_l = np.dot(Rq.T,tau_aero_l)

    # tau_aero_r = np.cross(eta_r,F_r_vec)
    # tau_aero_r = np.dot(Rq.T,tau_aero_r)


    F_aero_sum = (np.sum(F_l_vec,1)+np.sum(F_r_vec,1))/m

    # print("F_aero_sum", F_aero_sum)

    u[0:3]=F_aero_sum

    tau_aero_l_sum = np.sum(tau_aero_l, axis=1)
    tau_aero_r_sum = np.sum(tau_aero_r, axis=1)
    tau_aero = tau_aero_l_sum+tau_aero_r_sum
    # print("tau_aero_l: ", tau_aero_l_sum)
    # print("tau_aero_r: ", tau_aero_r_sum)
    # print("tau_aero: ", tau_aero)

    u[3:6] = tau_aero 
    # u[4] = dphi_l
    # u[5] = dphi_r
    
    # u[6] = dpsi_l
    # u[7] = dpsi_r

    # u[0] = Thrust #1.5*g;
    Thrust = np.dot(np.dot(Rq,e3),F_aero_sum)
    # print("Thrust", Thrust)

    # chordlength_r = np.loadtxt('chordlength_list_r.txt')
    # print("chordlength r:", chordlength_r[1])


    return u



def test_Feedback_Linearization_controller_intermediate_pt(x,t):
    # Output feedback linearization 2
    #
    # y1= ||r-rf||^2/2
    # y2= wx
    # y3= wy
    # y4= wz
    #
    # Analysis: The zero dynamics is unstable.
    u_ForceTorque = np.zeros(4)
    u = np.zeros(8)
    global g, m, I, r_w, rw_l, bw, xf, Aout, Bout, Mout, T_period, w_freq, radius, ratio_ab, delta_lissajous, ratio_rw, circle_flag, w_resonance, u_sampled, rho, beta_geometry, CL, chordlength_r, chordlength_cr, blade_width, torque_flag, r_filtered, q_filtered, v_filtered, w_filtered, Thrust_filtered, tau_aero_filtered, sample_flag, Thrust_filtered_avg, tau_aero_filtered_avg, q_filtered_avg_prev

    ratio_rw_timevarying =base_ratio+ratio_rw*(math.exp(-decaying_rate*t))
    # print("%%%%%%%%%%%%%%%%%%%%%")
    # print("%%CLF-QP  %%%%%%%%%%%")
    # print("%%%%%%%%%%%%%%%%%%%%%")


    print("t:", t)
    prog = MathematicalProgram()
    u_var = prog.NewContinuousVariables(3, "u_var")
    c_var = prog.NewContinuousVariables(1, "c_var")
    
    zd = xf[2]
    # # # # Example 1 : Circle
    if circle_flag==0:
        x_f = xf[0]+radius*math.cos(w_freq*t)
        y_f = xf[1]+radius*math.sin(w_freq*t)
        z_f = xf[2]#+1/4.*radius*math.sin(w_freq/2*t)
        # print("x_f:",x_f)
        # print("y_f:",y_f)
        dx_f = -radius*math.pow(w_freq,1)*math.sin(w_freq*t)
        dy_f = radius*math.pow(w_freq,1)*math.cos(w_freq*t)

        dz_f = 1/8.*math.pow(w_freq,1)*radius*math.cos(w_freq/2*t)
        ddx_f = -radius*math.pow(w_freq,2)*math.cos(w_freq*t)
        ddy_f = -radius*math.pow(w_freq,2)*math.sin(w_freq*t)
        ddz_f = -1/16.*radius*math.pow(w_freq,2)*math.sin(w_freq/2*t)
        dddx_f = radius*math.pow(w_freq,3)*math.sin(w_freq*t)
        dddy_f = -radius*math.pow(w_freq,3)*math.cos(w_freq*t)
        ddddx_f = radius*math.pow(w_freq,4)*math.cos(w_freq*t)
        ddddy_f = radius*math.pow(w_freq,4)*math.sin(w_freq*t)
    elif circle_flag==1:
        # Example 2 : Lissajous curve a=1 b=2
        a=1;
        b=ratio_ab*a;
        
        x_f = radius*math.sin(a*w_freq*t+delta_lissajous)
        y_f = radius*math.sin(b*w_freq*t)
        
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
    q=x[0:7]
    qd=x[7:13]


    phi_l=x[13]
    phi_r=x[14]
    psi_l=x[15]
    psi_r=x[16]

    # print("qnorm:", np.dot(q[3:7],q[3:7]))

    q0=q[3]
    q1=q[4]
    q2=q[5]
    q3=q[6]


    r=np.zeros((3,1))
    v=np.zeros((3,1))
    w=np.zeros((3,1))
    r=q[0:3]
    
    v=qd[0:3]
    v_vel=v; 

    w=qd[3:6]
    q_quat=q[3:7]
    
    dphi_l=x[17]
    dphi_r=x[18]

    # dpsi_l=x[19]
    # dpsi_r=x[20]

    # Filtered signal

    num_iter=math.ceil(t/time_step_cycle);
    M_mat_quat=np.zeros((4,4))
    # print("num_iter:", num_iter)
    # print("r_shape:", r.shape)
    if num_iter<filter_window_size:
        r_filtered=np.matmul(r.reshape(3,1),np.ones((1,filter_window_size)))
        q_filtered=np.dot(q_quat.reshape(4,1),np.ones((1,filter_window_size)))
        v_filtered=np.dot(v.reshape(3,1),np.ones((1,filter_window_size)))
        w_filtered=np.dot(w.reshape(3,1),np.ones((1,filter_window_size)))

        q_filtered_avg = q_quat
    else:
        r_filtered[:,0:filter_window_size-1]=r_filtered[:,1:filter_window_size]
        q_filtered[:,0:filter_window_size-1]=q_filtered[:,1:filter_window_size]
        v_filtered[:,0:filter_window_size-1]=v_filtered[:,1:filter_window_size]
        w_filtered[:,0:filter_window_size-1]=w_filtered[:,1:filter_window_size]

        r_filtered[:,filter_window_size-1]=r
        q_filtered[:,filter_window_size-1]=q_quat
        v_filtered[:,filter_window_size-1]=v
        w_filtered[:,filter_window_size-1]=w

        for j_iter in range(0,filter_window_size):
            M_mat_quat = M_mat_quat+np.matmul(q_filtered[:,j_iter].reshape(4,1),q_filtered[:,j_iter].reshape(1,4))
            # print("M_mat_quat:", M_mat_quat)
        # print("M_mat_quat:", M_mat_quat)
        w_eig, vr_eig = np.linalg.eig(M_mat_quat)
        # print("vr_eig:", vr_eig)
        # print("vr_eig:", vr_eig)
        # print("w_eig:", w_eig)
        # print("argmax w_eig", np.argmax(w_eig))

        q_filtered_avg_undecided = vr_eig[:, np.argmax(w_eig)]
        error_q_avg_plus = q_filtered_avg_undecided-q_filtered_avg_prev
        error_q_avg_minus = -q_filtered_avg_undecided-q_filtered_avg_prev

        if np.dot(error_q_avg_plus,error_q_avg_plus)>np.dot(error_q_avg_minus,error_q_avg_minus):
            q_filtered_avg=-q_filtered_avg_undecided
            # print("q_filtered_avg",q_filtered_avg)
            # print("q_filtered_avg_prev",q_filtered_avg_prev)
        else:
            q_filtered_avg=q_filtered_avg_undecided
            # print("q_filtered_avg",q_filtered_avg)
            # print("q_filtered_avg_prev",q_filtered_avg_prev)

    q_filtered_avg_prev=q_filtered_avg
        # print("norm q", np.dot(q_filtered_avg,q_filtered_avg))
    # Averaging filter
    r_filtered_avg= np.sum(r_filtered,1)/filter_window_size
    v_filtered_avg= np.sum(v_filtered,1)/filter_window_size
    w_filtered_avg= np.sum(w_filtered,1)/filter_window_size

    # Useful vectors and matrices
    
    (Rq, Eq, wIw, I_inv, fdw, taudw, vd_aero, wd_aero)=robobee_plant_aero.GetManipulatorDynamics(np.hstack([r_filtered_avg, q_filtered_avg]), np.hstack([v_filtered_avg, w_filtered_avg]))
    
    r=r_filtered_avg
    v=v_filtered_avg
    w=w_filtered_avg
    q0=q_filtered_avg[0]
    q1=q_filtered_avg[1]
    q2=q_filtered_avg[2]
    q3=q_filtered_avg[3]


    

    w_hat = np.zeros((3,3))
    w_hat[0,:] = np.array([     0,   -w[2],     w[1] ])
    w_hat[1,:] = np.array([  w[2],       0,    -w[0] ])
    w_hat[2,:] = np.array([ -w[1],    w[0],        0 ])

    Iw = np.dot(I,w)
    r_w = rw_l*e3; # Length to the wing from CoM
    wr_w = np.cross(w,r_w) # w x r
    wr_w_hat = np.dot(w_hat,r_w)
    # print("wr_w-wr_w_hat", wr_w-wr_w_hat)

    wwr_w = np.cross(w,wr_w) # w x (w x r)

    # fdw taudw vd_aero wd_aero baed on the model
    fdw = -bw*(v+np.dot(Rq,wr_w))
    RqTfdw=np.dot(Rq.T,fdw)
    taudw = np.cross(r_w,RqTfdw)
        
    vd_aero =  np.dot((-g*np.eye(3)),e3) + fdw/m # \dot{v} = -ge3 +R(q)e3 u[0] : u[0] Thrust is a unit of acceleration
    wd_aero = -np.dot(I_inv,wIw)+ np.dot(I_inv, taudw)

    we3 = np.cross(w,e3)
    wwe3 = np.cross(w,we3)
    we1 = np.cross(w,e1)
    wwe1 = np.cross(w,we1)


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

    r_w_hat = np.zeros((3,3))
    r_w_hat[0,:] = np.array([        0,   -r_w[2],     r_w[1] ])
    r_w_hat[1,:] = np.array([   r_w[2],         0,    -r_w[0] ])
    r_w_hat[2,:] = np.array([  -r_w[1],    r_w[0],          0 ])
    
    Rq_r_w_hat = np.zeros((3,3))
    Rq_r_w_hat=np.dot(Rq,r_w_hat);
    # print("Rq_r_w_hat: ",r_w_hat )
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


    e3_hat = np.zeros((3,3))
    e3_hat[0,:] = np.array([        0,   -e3[2],     e3[1] ])
    e3_hat[1,:] = np.array([  e3[2],          0,    -e3[0] ])
    e3_hat[2,:] = np.array([ -e3[1],    e3[0],           0 ])

    # Precalculation

    
    #- Checking the derivation

    # print("F1qEqT-(-Rqe3_hat)",np.dot(F1q,Eq.T)-(-Rqe3_hat))
    # Rqe3_cal = np.zeros(3)
    # Rqe3_cal[0] = 2*(q3*q1+q0*q2)
    # Rqe3_cal[1] = 2*(q3*q2-q0*q1)
    # Rqe3_cal[2] = (q0*q0+q3*q3-q1*q1-q2*q2)

    # print("Rqe3 - Rqe3_cal", Rqe3-Rqe3_cal)

    # Four output
    y1 = r[0]+ratio_rw_timevarying*np.dot(e1.T,np.dot(Rq,r_w))-x_f
    y2 = r[1]+ratio_rw_timevarying*np.dot(e2.T,np.dot(Rq,r_w))-y_f
    y3 = r[2]+ratio_rw_timevarying*np.dot(e3.T,np.dot(Rq,r_w))-z_f
    y4 = math.atan2(Rqe1_x,Rqe1_y)-math.pi/8
    # print("Rqe1_x:", Rqe1_x)

    eta1 = np.zeros(3)
    eta1 = np.array([y1,y2,y3])
    eta5 = y4
    # # print("y4", y4)
   
    # First derivative of first three output and yaw output
    eta2 = np.zeros(3)
    eta2 = v + ratio_rw_timevarying*np.dot(Rq,wr_w) - np.array([dx_f,dy_f,0])
    
    x2y2 = (math.pow(Rqe1_x,2)+math.pow(Rqe1_y,2)) # x^2+y^2
    
    eta6_temp = np.zeros(3)     #eta6_temp = (ye2T-xe1T)/(x^2+y^2)
    eta6_temp = (Rqe1_y*e2.T-Rqe1_x*e1.T)/x2y2    
    # print("eta6_temp:", eta6_temp)
    # Body frame w  ( multiply R)
    eta6 = np.dot(eta6_temp,np.dot(-Rqe1_hat,np.dot(Rq,w)))


    # Second derivative of first three output
    eta3_temp = np.zeros(3)
    kd = 2*1e-1

    eta3_damping = -kd*np.dot(I_inv, w)
    eta3_temp = vd_aero+ratio_rw_timevarying*np.dot(Rdot,wr_w)-ratio_rw_timevarying*np.dot(Rq_r_w_hat,wd_aero)- np.array([ddx_f,ddy_f,0])#-np.dot(Rq_r_w_hat,eta3_damping)
    
    # Body frame w
    dRqe1_x = np.dot(e2,np.dot(-Rqe1_hat,Rw)) # \dot{x}
    dRqe1_y = np.dot(e1,np.dot(-Rqe1_hat,Rw)) # \dot{y}
    alpha1 = 2*(Rqe1_x*dRqe1_x+Rqe1_y*dRqe1_y)/x2y2 # (2xdx +2ydy)/(x^2+y^2)

    B_yaw_temp3 =np.zeros(3)
    B_yaw_temp3 = alpha1*np.dot(Rqe1_hat,Rw)+np.dot(Rqe1_hat,np.dot(Rq,np.dot(I_inv,wIw-taudw)))-np.dot(Rw_hat,np.dot(Rqe1_hat,Rw))

    B_yaw = np.dot(eta6_temp,B_yaw_temp3) # +alpha2 :Could be an error in math.
    g_yaw = np.zeros(3)
    g_yaw = -np.dot(eta6_temp,np.dot(Rqe1_hat,np.dot(Rq,I_inv)))


    # (3x4) only position
    A_fl = np.zeros((3,4))
    A_fl[0:3,0] = Rqe3
    A_fl[0:3,1:4] = -ratio_rw_timevarying*np.dot(Rq_r_w_hat,I_inv)
    
    # print("A_fl:", A_fl)
    A_fl_pseudo = np.zeros((3,3))
    A_fl_pseudo = np.dot(A_fl,A_fl.T)

    # print("A_fl pseudo:", A_fl_pseudo)
    A_fl_pseudo_inv = np.linalg.inv(A_fl_pseudo);
    A_fl_inv = np.dot(A_fl.T,A_fl_pseudo_inv)
    # A_fl_pseudo_check = np.zeros((4,3))
    # A_fl_pseudo_check[0,:] = e3.T
    # A_fl_pseudo_check[1:4,:] = np.dot(I,e3_hat)/rw_l
    # A_fl_pseudo_check = np.dot(A_fl_pseudo_check,Rq.T)
    # print("A_fl pseudo_inv:", A_fl_inv)
    # print("A_fl pseudo_inv_check:", A_fl_pseudo_check)

    # A_fl_det = np.linalg.det(A_fl_pseudo)
     # Testing the feedback 
    mu = np.zeros(3)
    # epsilonnn=3e-1
    # mu[0:3] = -1/math.pow(epsilonnn,2)*eta1 -2/math.pow(epsilonnn,1)*eta2
    eta = np.zeros(6)
    eta = np.hstack([eta1, eta2])

    # FL controller
    # R_inv = np.linalg.inv(Rout);
    k = np.matmul(R_inv,np.matmul(Bout.T,Mout))
    mu = -1./1.*np.matmul(k,eta)
    v_aero = -eta3_temp + mu;

    u_ForceTorque[0:4] = np.dot(A_fl_inv,v_aero)

    # deta =np.hstack([eta2.T, eta3.T, eta4.T, eta6.T, eta7.T, 0,0,0,0])
    # print("deta- d_eta", np.dot(Aout,eta)-deta)
    eta_norm = np.dot(eta,eta)
    # print("fdw: ", fdw/m)
    # print("taudw:", np.dot(I_inv, taudw))
    # print("velocity: ", v)
    # print("angular vel :", w )
    # print("vTRe3: ",  np.dot(v_vel.T,Rqe3))
    # u[0] = m*g/np.dot(v_vel.T,Rqe3)*(np.dot(v,e3)+np.dot(e3, np.dot(Rq,wr_w)))
    # u[1:4] = -0.1*taudw-kd*e3*w[2]
    # u[1:4] = -1*w
    Ixx=I[0,0]
    Iyy=I[1,1]
    Izz=I[2,2]
    # u_ForceTorque[1:4] = u[1:4] #-k_yaw*w[2]*e3#-(Ixx-Iyy)*e3*w[0]*w[1]-Izz*w[2]*e3#-(Ixx-Iyy)*e3*w[0]*w[1]# -0.05*w


    ### CLF-QP controller
    # v-CLF QP controller

    FP_PF = np.dot(Aout.T,Mout)+np.dot(Mout,Aout)
    PG = np.dot(Mout, Bout)
    
    L_FVx = np.dot(eta,np.dot(FP_PF,eta))
    L_GVx = 2*np.dot(eta.T,PG) # row vector

    # Decompose with bw term in eta3_tmep

    fdw = -(v+np.dot(Rq,wr_w))
    RqTfdw=np.dot(Rq.T,fdw)
    taudw = np.cross(r_w,RqTfdw)
    vd_aero_adaptive = fdw/m
    vd_aero_nonadaptive = -g*e3
    wd_aero_adaptive = np.dot(I_inv, taudw)
    wd_aero_nonadaptive = -np.dot(I_inv,wIw)

    eta3_temp_adaptive = vd_aero_adaptive-ratio_rw_timevarying*np.dot(Rq_r_w_hat,wd_aero_adaptive)
    bw_adaptive_flag = np.dot(L_GVx,eta3_temp_adaptive.T)
    eta3_temp_nonadaptive = vd_aero_nonadaptive +ratio_rw_timevarying*np.dot(Rdot,wr_w)-ratio_rw_timevarying*np.dot(Rq_r_w_hat,wd_aero_nonadaptive)- np.array([ddx_f,ddy_f,0])#-np.dot(Rq_r_w_hat,eta3_damping)

    if bw_adaptive_flag>0:
        L_fhx_star= bw_max*eta3_temp_adaptive+eta3_temp_nonadaptive
    
    else:
        L_fhx_star= bw_min*eta3_temp_adaptive+eta3_temp_nonadaptive
    # print("L_fhx_star", L_fhx_star)
    # print("eta3_temp", eta3_temp)
    # L_fhx_star = eta3_temp.T;
    # print("size:", np.size(u_var))
    Vx = np.dot(eta, np.dot(Mout,eta) )
    # phi0_exp = L_FVx+np.dot(L_GVx,L_fhx_star)+(min_e_Q/max_e_P)*Vx*1    # exponentially stabilizing
    # phi0_exp = L_FVx+np.dot(L_GVx,L_fhx_star)+min_e_Q*eta_norm      # more exact bound - exponentially stabilizing
    phi0_exp = L_FVx+np.dot(L_GVx,L_fhx_star)      # more exact bound - exponentially stabilizing
    
    A_fl_new = np.zeros((3,3))
    A_fl_new= A_fl[0:3,0:3]
    
    # print("A_fl_new:", A_fl_new)

    phi1_decouple = np.dot(L_GVx,A_fl_new)
    
    # # Solve QP
    v_var = np.dot(A_fl_new,u_var)  + L_fhx_star
    Quadratic_Positive_def = np.matmul(A_fl_new.T,A_fl_new)
    QP_det = np.linalg.det(Quadratic_Positive_def)
    c_QP = 2*np.dot(L_fhx_star.T,A_fl_new)
    
    
    # CLF_QP_cost_v = np.dot(v_var,v_var) // Exact quadratic cost
    CLF_QP_cost_v_effective = np.dot(u_var, np.dot(Quadratic_Positive_def,u_var))+np.dot(c_QP,u_var)-c_var[0] # Quadratic cost without constant term
    # CLF_QP_cost_v_effective = np.dot(u_var, np.dot(Quadratic_Positive_def,u_var))+np.dot(c_QP,u_var) # Quadratic cost without constant term
    
    # CLF_QP_cost_u = np.dot(u_var,u_var)
    
    # phi1 = np.dot(phi1_decouple,u_var)
    phi1 = np.dot(phi1_decouple,u_var)+c_var[0]*eta_norm

    #----Printing intermediate states

    # print("L_fhx_star: ",L_fhx_star)
    # print("c_QP:", c_QP)
    # print("Qp : ",Quadratic_Positive_def)
    # print("Qp det: ", QP_det)
    # print("c_QP", c_QP)

    # print("phi0_exp: ", phi0_exp)
    # print("PG:", PG)
    # print("L_GVx:", L_GVx)
    # print("eta6", eta6)
    # print("d : ", phi1_decouple)
    # print("Cost expression:", CLF_QP_cost_v)
    # print("Const expression:", phi0_exp+phi1)

    #----Different solver option // Gurobi did not work with python at this point (some binding issue 8/8/2018)
    # solver = IpoptSolver()
    # solver = GurobiSolver()
    # print solver.available()
    # assert(solver.available()==True)
    # assertEqual(solver.solver_type(), mp.SolverType.kGurobi)
    # solver.Solve(prog)
    # assertEqual(result, mp.SolutionResult.kSolutionFound)
    
    # mp.AddLinearConstraint()
    # print("x:", x)
    # print("phi_0_exp:", phi0_exp)
    # print("phi1_decouple:", phi1_decouple)
    
    # print("eta1:", eta1)
    # print("eta2:", eta2)
    # print("eta3:", eta3)
    # print("eta4:", eta4)
    # print("eta5:", eta5)
    # print("eta6:", eta6)

    # Set up the QP problem
    prog.AddQuadraticCost(CLF_QP_cost_v_effective)
    prog.AddConstraint(phi0_exp+phi1<=0)
    prog.AddConstraint(c_var[0]>=0)
    prog.AddConstraint(c_var[0]<=500)
    prog.AddConstraint(u_var[0]<=27)
    prog.AddConstraint(u_var[0]>=0)
    prog.AddConstraint(u_var[1]<=5)
    prog.AddConstraint(u_var[2]<=5)
    prog.AddConstraint(u_var[1]>=-5)
    prog.AddConstraint(u_var[2]>=-5)


    # solver = IpoptSolver()
    solver = GurobiSolver()
    
    # print(solver.available())
    prog.SetSolverOption(mp.SolverType.kIpopt, "print_level", 1) # CAUTION: Assuming that solver used Ipopt

    # print("CLF value:", Vx) # Current CLF value

    prog.SetInitialGuess(u_var, u_ForceTorque[0:3])
    # prog.Solve() # Solve with default osqp
    
    solver.Solve(prog)
    # print("Optimal u : ", prog.GetSolution(u_var))
    U_CLF_QP = prog.GetSolution(u_var)
    # C_CLF_QP = prog.GetSolution(c_var)
    # # #---- Printing for debugging
    # # # dx = robobee_plantBS.evaluate_f(U_fl,x)    
    # # # print("dx", dx)
    # # # print("\n######################")
    # # # # print("qe3:", A_fl[0,0])
    # # # print("u:", u)
    # # # print("\n####################33")
    
    # # # deta4 = B_qw+Rqe3*U_fl_zero[0]+np.matmul(-np.matmul(Rqe3_hat,I_inv),U_fl_zero[1:4])*xi1
    # # # deta6 = B_yaw+np.dot(g_yaw,U_fl_zero[1:4])
    # # # print("deta4:",deta4)
    # # # print("deta6:",deta6)
    # print(C_CLF_QP)

    # phi1_opt = np.dot(phi1_decouple, U_CLF_QP)
    # phi1_opt_FL = np.dot(phi1_decouple, U_fl[0:3])

    # print("FL u: ", U_fl)
    # print("CLF u:", U_CLF_QP)
    # print("Cost FL: ", np.dot(mu,mu))

    # v_CLF = np.dot(A_fl_new,U_CLF_QP)+L_fhx_star
    # # # print("Cost CLF: ", np.dot(v_CLF,v_CLF)-np.dot(L_fhx_star,L_fhx_star))

    # # print("Cost CLF: ", np.dot(v_CLF,v_CLF))
    # # print("Constraint FL : ", phi0_exp+phi1_opt_FL)
    # # print("Constraint CLF : ", phi0_exp+phi1_opt)
    # # u=np.zeros(4)
    u_ForceTorque[0:3] = U_CLF_QP


    # Input function shape generator
    torque_flag =0;
    if torque_flag==1:
        mean_stroke_l = -math.pi/8 #-math.pi/10*math.sin(w_resonance*0.05*t);
        mean_stroke_r = math.pi/8 #+math.pi/10*math.sin(w_resonance*0.05*t);

        half_wave_weight = 0.3;
        wave_weight = 0.934;
        twice_wave_weight = 0. #0.216;
        third_wave_weight = 0. #0.047
        half_wave_freq = w_resonance/2.;
        twice_wave_freq = 2.*w_resonance;
        third_wave_freq = 3.*w_resonance;
        torque_async =math.pi*(1.0);
    else:
        # Slow varing mean stroke angle
        # mean_stroke_l =-math.pi/15*(w_filtered_avg[1])*0.1 #-math.pi/15*math.sin(w_resonance*0.1*t);
        # mean_stroke_r =math.pi/15*(w_filtered_avg[1])*0.1 #math.pi/15*math.sin(w_resonance*0.1*t);
        half_wave_weight = 0.0;
        wave_weight = 1.;
        twice_wave_weight = 0.0;
        third_wave_weight = 0.;
        half_wave_freq = 0.5*w_resonance;
        twice_wave_freq = 2.*w_resonance;
        third_wave_freq = 3.*w_resonance;
        torque_async =0.0;


    # Invert the mapping

    # 500Hz control updates
    if flag_discrete_control_500Hz==1:
        vicon_sampled_time =truncate(t*1000,2)%2
        # print("sampled t: ", vicon_sampled_time)
        
        if vicon_sampled_time==0 and sample_flag==0:
            u_sampled = u_ForceTorque
            sample_flag=1
        if vicon_sampled_time==1 and sample_flag==1:
            sample_flag=0
        
        u_ForceTorque=u_sampled

    F_desired_thrust = u_ForceTorque[0]*m
    torque_desired = u_ForceTorque[1:4]#-np.array([Ixx*w[1],Iyy*w[1],0])

    F_l_avg_coeff=1/2.*rho*beta_geometry*CL*(math.pow(wave_weight*w_resonance,2)
        +math.pow(half_wave_weight*0.5*w_resonance,2)
        +math.pow(twice_wave_weight*2*w_resonance,2)
        +math.pow(third_wave_weight*3*w_resonance,2))/Force_scale

    F_r_avg_coeff=1/2.*rho*beta_geometry*CL*(math.pow(wave_weight*w_resonance,2)
        +math.pow(half_wave_weight*0.5*w_resonance,2)
        +math.pow(twice_wave_weight*2*w_resonance,2)
        +math.pow(third_wave_weight*3*w_resonance,2))/Force_scale

    if F_desired_thrust/m<=0.1:
        print("F_desired_thrust", F_desired_thrust/m)
        F_desired_thrust=0;
        Phi_off_l=0;
        Phi_off_r=0;
        Phi_amp_l=0;
        Phi_amp_r=0;
    else:    
        if math.fabs((torque_desired[1])/F_desired_thrust/r_cp)>0.707:
            print("F_desired_thrust", F_desired_thrust/m)
            print("sin(phi_off)", (torque_desired[1])/F_desired_thrust/r_cp)
            print("Phi_off_l_estimate", (math.copysign(1,torque_desired[1])*math.asin(0.707)))
            # Phi_off_l_estimate = (math.copysign(1,torque_desired[1])*math.asin(0.9))
        else:
            Phi_off_l_estimate = math.asin((torque_desired[1])/F_desired_thrust/r_cp)
        Phi_amp_sq = (F_desired_thrust*r_cp*math.cos(Phi_off_l_estimate)+torque_desired[0])/(r_cp*math.cos(Phi_off_l_estimate)*F_l_avg_coeff)/2.
        Phi_amp_sq_cross = (F_desired_thrust*r_cp*math.cos(Phi_off_l_estimate)-torque_desired[0])/(r_cp*math.cos(Phi_off_l_estimate)*F_l_avg_coeff)/2.
        
        if Phi_amp_sq<=0: 
            print("Phi_amp_sq", Phi_amp_sq)
            Phi_amp_sq=0;
        if Phi_amp_sq_cross<=0:
            print("Phi_amp_sq_cross", Phi_amp_sq_cross)
            Phi_amp_sq_cross=0;

        Phi_amp_avg = (math.sqrt(Phi_amp_sq)+math.sqrt(Phi_amp_sq_cross))/2.
        Phi_amp_diff = (math.sqrt(Phi_amp_sq)-math.sqrt(Phi_amp_sq_cross))/2.


        Phi_off_l = Phi_off_l_estimate*gain_offset
        Phi_off_r = -Phi_off_l_estimate*gain_offset
        # print("roll torque_desired", torque_desired[0]);
        # print("Phi_amp_diff", Phi_amp_diff)
        Phi_amp_l = -(Phi_amp_avg+Phi_amp_diff*gain_diff)*gain_amp#-math.pi/8.5-math.pi/8.5*(zd-r_filtered_avg[2])*0.+math.pi/8.5*(v_filtered_avg[2])*0+math.pi/8.5*(w_filtered_avg[0])*0.05;
        Phi_amp_r = (Phi_amp_avg-Phi_amp_diff*gain_diff)*gain_amp#math.pi/8.5+math.pi/8.5*(zd-r_filtered_avg[2])*0.-math.pi/8.5*(v_filtered_avg[2])*0+math.pi/8.5*(w_filtered_avg[0])*0.05;
        
        # print(Phi_amp_l)
    # if Phi_amp_l>=0:
    #     phi_amp_l=0;
    # if Phi_amp_r<=0:
    #     phi_amp_r=0;
    
    V_l = 1/Transfer_mag_zero*Phi_off_l + Phi_amp_l*(wave_weight/Transfer_mag_reso*math.sin(w_resonance*t)             
            +half_wave_weight/Transfer_mag_reso_half*math.sin(half_wave_freq*t)              
            +twice_wave_weight/Transfer_mag_reso_twice*math.sin(twice_wave_freq*t)
            +third_wave_weight/Transfer_mag_reso_third*math.sin(third_wave_freq*t))

    V_r = 1/Transfer_mag_zero*Phi_off_r + Phi_amp_r*(wave_weight/Transfer_mag_reso*math.sin(w_resonance*t+torque_async)
            +half_wave_weight/Transfer_mag_reso_half*math.sin(half_wave_freq*t+torque_async) 
            +twice_wave_weight/Transfer_mag_reso_twice*math.sin(twice_wave_freq*t+torque_async)
            +third_wave_weight/Transfer_mag_reso_third*math.sin(third_wave_freq*t+torque_async))

    # print("V_l:", V_l)
    # print("phi_l:", phi_l)
    u[6] = V_l
    u[7] = V_r

    Rq_l = np.zeros((3,3))
    Rq_l[0,:] = np.array([ math.cos(phi_l),   -math.sin(phi_l),     0 ])
    Rq_l[1,:] = np.array([ math.sin(phi_l),    math.cos(phi_l),     0 ])
    Rq_l[2,:] = np.array([               0,                  0,     1 ])
    Rq_le2 = np.dot(Rq_l,e2)
    RqRq_le2 = np.dot(Rq,Rq_le2)
    w_hatRq_le2 = np.dot(w_hat, Rq_le2)
    Rqw_hatRq_le2 = np.dot(Rq, w_hatRq_le2)

    # print("x:", x)
    # print("w_hatRq_le2:", w_hatRq_le2)

    Rq_r = np.zeros((3,3))
    Rq_r[0,:] = np.array([ math.cos(phi_r),   -math.sin(phi_r),     0 ])
    Rq_r[1,:] = np.array([ math.sin(phi_r),    math.cos(phi_r),     0 ])
    Rq_r[2,:] = np.array([               0,                  0,     1 ])
    Rq_re2 = np.dot(Rq_r,e2)
    RqRq_re2 = np.dot(Rq,Rq_re2)
    w_hatRq_re2 = np.dot(w_hat, Rq_re2)
    Rqw_hatRq_re2 = np.dot(Rq, w_hatRq_re2)

    dphi_l_hat = np.zeros((3,3))
    dphi_l_hat[0,:] = np.array([      0,   -dphi_l,     0 ])
    dphi_l_hat[1,:] = np.array([ dphi_l,         0,     0 ])
    dphi_l_hat[2,:] = np.array([      0,         0,     0 ])

    dphi_l_hate2 = np.dot(dphi_l_hat, e2)
    Rq_ldphi_l_hate2 = np.dot(Rq_l, dphi_l_hate2)
    RqRq_ldphi_l_hate2 = np.dot(Rq, Rq_ldphi_l_hate2)

    dphi_r_hat = np.zeros((3,3))
    dphi_r_hat[0,:] = np.array([      0,   -dphi_r,     0 ])
    dphi_r_hat[1,:] = np.array([ dphi_r,         0,     0 ])
    dphi_r_hat[2,:] = np.array([      0,         0,     0 ])
    
    dphi_r_hate2 = np.dot(dphi_r_hat, e2)
    Rq_rdphi_r_hate2 = np.dot(Rq_r, dphi_r_hate2)
    RqRq_rdphi_r_hate2 = np.dot(Rq, Rq_rdphi_r_hate2)

    # print("dphi_r_hat:", dphi_r_hat)

    Rq_psi_l = np.zeros((3,3))
    Rq_psi_l[0,:] = np.array([  math.cos(psi_l),   0, math.sin(psi_l)])
    Rq_psi_l[1,:] = np.array([                0,   0,  0             ])
    Rq_psi_l[2,:] = np.array([ -math.sin(psi_l),   0, math.cos(psi_l)])

    Rq_psi_r = np.zeros((3,3))
    Rq_psi_r[0,:] = np.array([  math.cos(psi_r),   0, math.sin(psi_r)])
    Rq_psi_r[1,:] = np.array([                0,   0,  0             ])
    Rq_psi_r[2,:] = np.array([ -math.sin(psi_r),   0, math.cos(psi_r)])

    Iw = np.dot(I,w)
    r_w = rw_l*e3; # Length to the wing from CoM
    wr_w = np.cross(w,r_w) # w x r
    wr_w_hat = np.dot(w_hat,r_w)
    # print("wr_w-wr_w_hat", wr_w-wr_w_hat)

    wwr_w = np.cross(w,wr_w) # w x (w x r)

    # Wing angle

    eta = np.zeros(3)
    eta_l = np.zeros(3)
    eta_r = np.zeros(3)    

    eta = r+Rqe3*rw_l;
    eta_l = eta+np.dot(Rq,Rq_le2)*r_cp  # phi_l positive : clockwise
    eta_r = eta-np.dot(Rq,Rq_re2)*r_cp # phi_r positive : clockwise

    # Wing velocity

    deta = np.zeros(3)
    deta_l = np.zeros(3)
    deta_r = np.zeros(3)    

    deta = v + np.dot(Rq,wr_w)
    deta_l = deta + Rqw_hatRq_le2*r_cp + RqRq_ldphi_l_hate2*r_cp 
    deta_r = deta - Rqw_hatRq_re2*r_cp - RqRq_rdphi_r_hate2*r_cp 

    
    # Projection to wing stroke plane
    P_proj_wing_l = np.zeros((3,3))
    P_proj_wing_l = np.eye(3)-np.outer(RqRq_le2, RqRq_le2)
    # print("outer_before:", RqRq_le2)
    # print("outer:", P_proj_wing_l)
    P_proj_wing_r = np.zeros((3,3))
    P_proj_wing_r = np.eye(3)-np.outer(RqRq_re2, RqRq_re2)

    ## Decomposing the wing velocity in the projected domain
    
    proj_deta_l_body                = np.dot(P_proj_wing_l,deta)
    proj_deta_l_body_wing_couple    = np.dot(P_proj_wing_l,Rqw_hatRq_le2)
    proj_deta_l_wing                = np.dot(P_proj_wing_l,RqRq_ldphi_l_hate2)

    proj_deta_r_body                =  np.dot(P_proj_wing_r,deta)
    proj_deta_r_body_wing_couple    = -np.dot(P_proj_wing_r,Rqw_hatRq_re2)
    proj_deta_r_wing                = -np.dot(P_proj_wing_r,RqRq_rdphi_r_hate2)

    # nominal velocity in wing stroke plane for CoP
    # print("deta :", deta)
    # print("deta_l :", deta_l)
    # print("deta_r :", deta_r)
    v_ortho_l = np.dot(P_proj_wing_l, deta_l)
    v_ortho_r = np.dot(P_proj_wing_r, deta_r) 
    
    v_ortho_l_norm_sq = np.dot(v_ortho_l,v_ortho_l)
    v_ortho_r_norm_sq = np.dot(v_ortho_r,v_ortho_r)

    v_ortho_l_norm = math.sqrt(v_ortho_l_norm_sq)
    v_ortho_r_norm = math.sqrt(v_ortho_r_norm_sq)

    if v_ortho_l_norm==0 or v_ortho_r_norm==0 :
        v_ortho_l_normalized = np.zeros(3)
        v_ortho_r_normalized = np.zeros(3)
    else:
        v_ortho_l_normalized = v_ortho_l/v_ortho_l_norm
        v_ortho_r_normalized = v_ortho_r/v_ortho_r_norm 

    # print("v_ortho_l_normalized :", v_ortho_l_normalized)
    # print("v_ortho_r_normalized :", v_ortho_r_normalized)   
    # print("v_ortho_l_norm_sq :", v_ortho_l_norm_sq)
    # print("v_ortho_r_norm_sq :", v_ortho_r_norm_sq)   

    # Hinge angle in the global frame

    psi_l_wing = np.dot(Rq_psi_l,-e3) 
    psi_l_body = np.dot(Rq_l,psi_l_wing) 
    psi_l_world = np.dot(Rq,psi_l_body) 

    psi_r_wing = np.dot(Rq_psi_r,-e3) 
    psi_r_body = np.dot(Rq_r,psi_r_wing) 
    psi_r_world = np.dot(Rq,psi_r_body) 


    AoA_inner_product_l = np.dot(v_ortho_l_normalized, -psi_l_world) # Based on Kevin Science Robobtics supplement paper
    AoA_inner_product_r = np.dot(v_ortho_r_normalized, -psi_r_world)
    
    AoA_l = math.acos(AoA_inner_product_l)
    AoA_r = math.acos(AoA_inner_product_r)
    # print("AoA_l:", AoA_l)
    # print("AoA_r:", AoA_r)
    # nominal velocity in wing stroke plane for each blade

    FL_l_r=np.zeros(num_blade)
    FL_r_r=np.zeros(num_blade)
    FD_l_r = np.zeros((3,num_blade))
    FD_r_r = np.zeros((3,num_blade))
    FL_l_vec=np.zeros((3,num_blade))
    FL_r_vec=np.zeros((3,num_blade))
    FD_l_vec=np.zeros((3,num_blade))
    FD_r_vec=np.zeros((3,num_blade))
    F_l_vec=np.zeros((3,num_blade))
    F_r_vec=np.zeros((3,num_blade))
    tau_aero_l =np.zeros((3,num_blade))
    tau_aero_r =np.zeros((3,num_blade))

    chord_r_CoP_z_l=np.zeros(num_blade)
    chord_r_CoP_z_r=np.zeros(num_blade)
    
    chord_r_CoP_world_l=np.zeros((3,num_blade))
    chord_r_CoP_world_r=np.zeros((3,num_blade))



    for i_iter in range(0,num_blade):
        chord_r_from_root = wing_xr + chordlength_r[i_iter]

        chord_r_from_root_world_l=Rqe3*rw_l+np.dot(Rq,Rq_le2)*chord_r_from_root
        chord_r_from_root_world_r=Rqe3*rw_l-np.dot(Rq,Rq_re2)*chord_r_from_root

        v_ortho_l_r =  proj_deta_l_body + (proj_deta_l_body_wing_couple + proj_deta_l_wing)*chord_r_from_root
        v_ortho_r_r =  proj_deta_r_body + (proj_deta_r_body_wing_couple + proj_deta_r_wing)*chord_r_from_root

        v_ortho_l_r_norm_sq = np.dot(v_ortho_l_r, v_ortho_l_r)
        v_ortho_r_r_norm_sq = np.dot(v_ortho_r_r, v_ortho_r_r)

        # print("v_ortho_r_r_norm_sq", v_ortho_r_r_norm_sq )
        # print("v_ortho_l_r_norm_sq", v_ortho_l_r_norm_sq )

        v_ortho_l_r_norm = math.sqrt(v_ortho_l_r_norm_sq)
        v_ortho_r_r_norm = math.sqrt(v_ortho_r_r_norm_sq)

        if v_ortho_l_r_norm==0 or v_ortho_r_r_norm==0 :
            v_ortho_l_r_normalized = np.zeros(3)
            v_ortho_r_r_normalized = np.zeros(3)
        else:
            v_ortho_l_r_normalized = v_ortho_l_r/v_ortho_l_r_norm
            v_ortho_r_r_normalized = v_ortho_r_r/v_ortho_r_r_norm 

        AoA_inner_product_l_r = np.dot( v_ortho_l_r_normalized, -psi_l_world) # pointing downward : c vector
        AoA_inner_product_r_r = np.dot( v_ortho_r_r_normalized, -psi_r_world) # pointing downward : c vector

        AoA_l_r = math.acos(AoA_inner_product_l_r)
        AoA_r_r = math.acos(AoA_inner_product_r_r)

        # Center of Pressure for each blade : Kevin IROS 2016
        chord_r_CoP_z_l[i_iter] = chordlength_yLE[i_iter]-(0.25 + 0.25/(1+math.exp(5.*(1.-4.*AoA_l_r/math.pi))))*chordlength_cr[i_iter]
        chord_r_CoP_z_r[i_iter] = chordlength_yLE[i_iter]-(0.25 + 0.25/(1+math.exp(5.*(1.-4.*AoA_r_r/math.pi))))*chordlength_cr[i_iter]

        chord_r_CoP_world_l[0:3,i_iter] = chord_r_from_root_world_l+psi_l_world*chord_r_CoP_z_l[i_iter]
        chord_r_CoP_world_r[0:3,i_iter] = chord_r_from_root_world_r+psi_r_world*chord_r_CoP_z_r[i_iter]

        # print("Psi ", psi_l)
        # print("AoA_l_r", AoA_l_r)
        CL_l = CL0*math.sin(2.*AoA_l_r)
        CL_r = CL0*math.sin(2.*AoA_r_r)

        CD_l = (CD0+CDmax)/2.- (CDmax-CD0)/2.*math.cos(2.*AoA_l_r)
        CD_r = (CD0+CDmax)/2.- (CDmax-CD0)/2.*math.cos(2.*AoA_r_r)

        # print("CL_l:", CL_l)

        # print("CD_l:", CD_l)

        FL_l_r[i_iter] = 1./2.*rho*CL_l*v_ortho_l_r_norm_sq*chordlength_cr[i_iter]*blade_width
        FL_r_r[i_iter] = 1./2.*rho*CL_r*v_ortho_r_r_norm_sq*chordlength_cr[i_iter]*blade_width

        # print("FL_l_r", FL_l_r[i_iter])

        # print("FL_r_r", FL_r_r[i_iter])

        FD_l_r[0:3,i_iter] = -1./2.*rho*CD_l*v_ortho_l_r_norm_sq*v_ortho_l_r_normalized*chordlength_cr[i_iter]*blade_width
        FD_r_r[0:3,i_iter] = -1./2.*rho*CD_r*v_ortho_r_r_norm_sq*v_ortho_r_r_normalized*chordlength_cr[i_iter]*blade_width
        
        FL_l_vec[0:3,i_iter] = FL_l_r[i_iter]*np.dot(Rq,e3)/Force_scale
        FL_r_vec[0:3,i_iter] = FL_r_r[i_iter]*np.dot(Rq,e3)/Force_scale
        FD_l_vec[0:3,i_iter] = FD_l_r[0:3,i_iter]/Force_scale
        FD_r_vec[0:3,i_iter] = FD_r_r[0:3,i_iter]/Force_scale
        
        # print("FL_l_vec in body:", np.dot(Rq.T,FL_l_vec[0:3,i_iter]))
        # print("FD_l_vec in body:", np.dot(Rq.T,FD_l_vec[0:3,i_iter]))


        F_l_vec[0:3,i_iter] = FL_l_vec[0:3,i_iter] + FD_l_vec[0:3,i_iter]
        F_r_vec[0:3,i_iter] = FL_r_vec[0:3,i_iter] + FD_r_vec[0:3,i_iter]

        tau_aero_l[0:3,i_iter] = np.cross(chord_r_CoP_world_l[0:3,i_iter],F_l_vec[0:3,i_iter])
        tau_aero_l[0:3,i_iter] = np.dot(Rq.T,tau_aero_l[0:3,i_iter])

        tau_aero_r[0:3,i_iter] = np.cross(chord_r_CoP_world_r[0:3,i_iter],F_r_vec[0:3,i_iter])
        tau_aero_r[0:3,i_iter] = np.dot(Rq.T,tau_aero_r[0:3,i_iter])


    FL_l = np.sum(FL_l_r)
    FL_r = np.sum(FL_r_r)
    FD_l = np.sum(FD_l_r)
    FD_r = np.sum(FD_r_r)


    F_aero_sum = (np.sum(F_l_vec,1)+np.sum(F_r_vec,1))/m

    # print("F_aero_sum", F_aero_sum)

    u[0:3]=F_aero_sum

    tau_aero_l_sum = np.sum(tau_aero_l, axis=1)
    tau_aero_r_sum = np.sum(tau_aero_r, axis=1)
    tau_aero = tau_aero_l_sum+tau_aero_r_sum
    # print("tau_aero_l: ", tau_aero_l_sum)
    # print("tau_aero_r: ", tau_aero_r_sum)
    
    u[3:6] = tau_aero 
    # print("u_ForceTorque",u_ForceTorque)
    # u[0:3]=u_ForceTorque[0]*np.dot(Rq,e3)
    # u[3:6]=u_ForceTorque[1:4]
    # u[4] = dphi_l
    # u[5] = dphi_r
    
    # u[6] = dpsi_l
    # u[7] = dpsi_r

    # u[0] = Thrust #1.5*g;

    # # Status check : Avg vs Estimation
    Thrust = np.dot(np.dot(Rq,e3),F_aero_sum)
    print("Thrust", Thrust)
    print("Thrust_desired", F_desired_thrust/m)
    # # print("Thrust_filtered", Thrust_filtered)
    if num_iter<filter_window_size:
        Thrust_filtered=Thrust*np.ones(filter_window_size)
        tau_aero_filtered=np.dot(tau_aero.reshape(3,1),np.ones((1,filter_window_size)))

    else:
        Thrust_filtered[0:filter_window_size-1]=Thrust_filtered[1:filter_window_size]
        tau_aero_filtered[:,0:filter_window_size-1]=tau_aero_filtered[:,1:filter_window_size]

        Thrust_filtered[filter_window_size-1]=Thrust
        tau_aero_filtered[:,filter_window_size-1]=tau_aero

    Thrust_filtered_avg= np.sum(Thrust_filtered)/filter_window_size
    tau_aero_filtered_avg= np.sum(tau_aero_filtered,1)/filter_window_size
    print("Thrust_filtered", Thrust_filtered_avg)

    print("tau_aero: ", tau_aero)
    print("tau_desired:", torque_desired)
    print("tau_filtered", tau_aero_filtered_avg)


    # chordlength_r = np.loadtxt('chordlength_list_r.txt')
    # print("chordlength r:", chordlength_r[1])


    # # dpsi_l = math.pi/4.0*math.pow(w_resonance/1.,1)*math.cos(w_resonance/1.*t)
    # # dpsi_r = math.pi/4.0*math.pow(w_resonance/1.,1)*math.cos(w_resonance/1.*t+torque_async)
    # # u[8] = k*dpsi_l
    # # u[9] = dpsi_r
    # # dpsi_l = math.pi/4.0*math.pow(w_resonance/1.,1)*math.cos(w_resonance/1.*t)
    # # dpsi_r = math.pi/4.0*math.pow(w_resonance/1.,1)*math.cos(w_resonance/1.*t+torque_async)
    # # u[8] = dpsi_l
    # # u[9] = dpsi_r
    
    # zd=xf[2]





    # # Wing trajectory
    # # u[4] = -math.pi/4*math.pow(w_resonance,1)*math.sin(w_resonance*t)
    # # u[5] = math.pi/4*math.pow(w_resonance,1)*math.sin(w_resonance*t)
    # decay_rate_wing = -1e2;
    
    # # u[0] = g
    # # if u[0]<0:
    # #     u =np.zeros(4)
    # U_fl = u;
    # print("u :", u)

    # print("A_fl:", A_fl)
    # # print("A_fl_det:", A_fl_det)
    # print("eta_norm:", eta_norm)

    # print("min_e_Q_BAB:", min_e_Q_BAB)
    # print("eval_p_ISS:", eval_P_ISS)
    # print("kq", kq)
    # mu_M = (bw+m)/(m*rw_l*ratio_rw_timevarying*min_eval_P_ISS)*(radius*w_freq*w_freq)
    
    # # Ub_w = (1.+1./2.)*mu_M*mu_M+ 4*math.sqrt(g/ratio_rw_timevarying)
    # Ub_w = (1.+1./2.)*mu_M*mu_M+ 4*(g/ratio_rw_timevarying)
    
    # # Ub_w = 1/(ratio_rw_timevarying*min_eval_P_ISS)*(bw/m*radius*w_freq+radius*1*w_freq*w_freq)+ 2*math.sqrt(g/ratio_rw_timevarying)
    # print("bounds on w1 w2 : ", Ub_w)
    # print("evalue :", bw*ratio_rw/((1+ratio_rw)*m) )
    # print("mu_M", mu_M)

    return u 
# Test LQR


if flag_FL_controller==1:
    input_log, state_log = \
        RunSimulation(robobee_plant_aero,
              test_Feedback_Linearization_controller,
              x0=x0,
              duration=duration)
elif flag_FL_controller==0:
    input_log, state_log = \
        RunSimulation(robobee_plant_aero,
              test_controller,
              x0=x0,
              duration=duration)
elif flag_FL_controller==2:
    input_log, state_log = \
        RunSimulation(robobee_plant_aero,
              test_controller_thrust,
              x0=x0,
              duration=duration)
elif flag_FL_controller==3:
    input_log, state_log = \
        RunSimulation(robobee_plant_aero,
              test_Feedback_Linearization_controller_intermediate_pt,
              x0=x0,
              duration=duration)
elif flag_FL_controller==4:
    input_log, state_log = \
        RunSimulation(robobee_plant_aero,
              test_harmonic_controller,
              x0=x0,
              duration=duration)


sample_flag=0;
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
ubar = np.zeros((8,num_iteration)) # Convert to Euler Angle
u = np.zeros((8,num_iteration)) # Convert to Euler Angle
w_compare= np.zeros((3,num_iteration))
fdw_all = np.zeros((3,num_iteration))
fdw_all_output = np.zeros((3,num_iteration))
d_fdw_all = np.zeros((3,num_iteration))
dfdw_all = np.zeros((3,num_iteration))
vdd_all = np.zeros((3,num_iteration))
wdd_all = np.zeros((3,num_iteration))
d_wd_all = np.zeros((3,num_iteration))
vd_all = np.zeros((3,num_iteration))
d_vd_all = np.zeros((3,num_iteration))
wd_all = np.zeros((3,num_iteration))
d_w_all = np.zeros((3,num_iteration))
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
u_all = np.zeros((8,num_iteration))
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
vd_Kinetic_upper_all=np.zeros(num_iteration)
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
d_V_all  = np.zeros(num_iteration)

Vx_all = np.zeros(num_iteration)
eta_norm_all = np.zeros(num_iteration)
mu_1_norm =np.zeros(num_iteration)

x_f = np.zeros(num_iteration)
y_f = np.zeros(num_iteration)
z_f= np.zeros(num_iteration)
dx_f = np.zeros(num_iteration)
dy_f = np.zeros(num_iteration)
dz_f = np.zeros(num_iteration)
ddx_f = np.zeros(num_iteration)
ddy_f = np.zeros(num_iteration)
ddz_f = np.zeros(num_iteration)
dddx_f = np.zeros(num_iteration)
dddy_f = np.zeros(num_iteration)
dddz_f = np.zeros(num_iteration)
ddddx_f = np.zeros(num_iteration)
ddddy_f = np.zeros(num_iteration)
ddddz_f = np.zeros(num_iteration)
Ub_w_all = np.ones(num_iteration)
Ub_w_all_new = np.ones(num_iteration)
perturb = np.zeros((3,num_iteration))
dist_CoM_ref= np.ones(num_iteration)
wing_l = np.zeros((3,num_iteration))
wing_r = np.zeros((3,num_iteration))
vehicle_bottom = np.zeros((3,num_iteration))
vehicle_top = np.zeros((3,num_iteration))

# harmonic output
phi_l = np.zeros(num_iteration)
phi_r = np.zeros(num_iteration)
psi_l = np.zeros(num_iteration)
psi_r = np.zeros(num_iteration)
dphi_l = np.zeros(num_iteration)
dphi_r = np.zeros(num_iteration)

V_l = np.zeros(num_iteration)
V_r = np.zeros(num_iteration)
Thrust_l = np.zeros(num_iteration)
Thrust_r = np.zeros(num_iteration)
Thrust_l_avg = np.zeros(num_iteration)
Thrust_r_avg = np.zeros(num_iteration)
Thrust = np.zeros(num_iteration)
Thrust_actual = np.zeros(num_iteration)
F_aero_sum = np.zeros((3,num_iteration))
tau_aero_sum =np.zeros((3,num_iteration))

F_l_avg = np.zeros(num_iteration)
F_r_avg = np.zeros(num_iteration)
F_l_vec_avg = np.zeros((3,num_iteration))
F_r_vec_avg = np.zeros((3,num_iteration))
tau_aero_l_avg=np.zeros((3,num_iteration))
tau_aero_r_avg=np.zeros((3,num_iteration))

AoA_l = np.zeros(num_iteration)
AoA_r = np.zeros(num_iteration)

r_filtered_avg_out=np.zeros((3,num_iteration))
q_filtered_avg_out=np.zeros((4,num_iteration))
q_filtered_avg_prev_out=np.zeros((4,num_iteration))
v_filtered_avg_out=np.zeros((3,num_iteration))
w_filtered_avg_out=np.zeros((3,num_iteration))

u_ForceTorque_500Hz=np.zeros((4,num_iteration))


if torque_flag==1:
    mean_stroke_l = -math.pi/8 #-math.pi/15*math.sin(w_resonance*0.1*t);
    mean_stroke_r = math.pi/8 #math.pi/15*math.sin(w_resonance*0.1*t);
    
    half_wave_weight =  0.3 #0.6;
    wave_weight = 0.934;
    twice_wave_weight = 0. #0.216;
    third_wave_weight = 0. #0.047
    half_wave_freq = w_resonance/2.;
    twice_wave_freq = 2.*w_resonance;
    third_wave_freq = 3.*w_resonance;
    torque_async =math.pi*(1.0);
else:
    # Slow varing mean stroke angle
    mean_stroke_l = -math.pi/8 #-math.pi/15*math.sin(w_resonance*0.1*t);
    mean_stroke_r = math.pi/8 #math.pi/15*math.sin(w_resonance*0.1*t);
    half_wave_weight = 0.0;
    wave_weight = 1.;
    twice_wave_weight = 0.0;
    third_wave_weight = 0.;
    half_wave_freq = 0.5*w_resonance;
    twice_wave_freq = 2.*w_resonance;
    third_wave_freq = 3.*w_resonance;
    torque_async =0.0;

for j in range(0,num_iteration):


    if torque_flag==1:
        mean_stroke_l = -math.pi/8 #-math.pi/15*math.sin(w_resonance*0.1*t);
        mean_stroke_r = math.pi/8 #math.pi/15*math.sin(w_resonance*0.1*t);
        
        half_wave_weight =  0.3 #0.6;
        wave_weight = 0.934;
        twice_wave_weight = 0. #0.216;
        third_wave_weight = 0. #0.047
        half_wave_freq = w_resonance/2.;
        twice_wave_freq = 2.*w_resonance;
        third_wave_freq = 3.*w_resonance;
        torque_async =math.pi*(1.0);
    else:
        # Slow varing mean stroke angle
        mean_stroke_l = -math.pi/15*math.sin(w_resonance*0.1*time_out[j]);
        mean_stroke_r = math.pi/15*math.sin(w_resonance*0.1*time_out[j]);
        half_wave_weight = 0.0;
        wave_weight = 1.;
        twice_wave_weight = 0.0;
        third_wave_weight = 0.;
        half_wave_freq = 0.5*w_resonance;
        twice_wave_freq = 2.*w_resonance;
        third_wave_freq = 3.*w_resonance;
        torque_async =0.0;


    

    ratio_rw_timevarying =base_ratio+ratio_rw*(math.exp(-decaying_rate*time_out[j]))
    
    # ubar[:,j]=test_Feedback_Linearization_controller_BS(state_out[:,j])
    # print("r_w:", r_w)
    ubar[:,j]=input_out[:,j]
    q_temp =state_out[3:7,j]
    q_temp_norm =math.sqrt(np.dot(q_temp,q_temp));
    q_temp = q_temp/q_temp_norm;
    quat_temp = Quaternion(q_temp)    # Quaternion

    R = RotationMatrix(quat_temp)
    rpy[:,j]=RollPitchYaw(R).vector()


    q=np.zeros(7)
    qd=np.zeros(6)
    q=state_out[0:7,j]
    qd=state_out[11:17,j]
    q_quat=q[3:7]
    

    phi_l[j]=state_out[7,j]
    phi_r[j]=state_out[8,j]
    psi_l[j]=state_out[9,j]
    psi_r[j]=state_out[10,j]

    # print("qnorm:", np.dot(q[3:7],q[3:7]))

    q0=q[3]
    q1=q[4]
    q2=q[5]
    q3=q[6]

    r=q[0:3]
    v=qd[0:3]
    v_vel=v; 

    w=qd[3:6]
    
    dphi_l[j]=state_out[17,j]
    dphi_r[j]=state_out[18,j]

    # Filtered signal

    num_iter=math.ceil(time_out[j]/time_step_cycle);
    M_mat_quat=np.zeros((4,4))
    # print("num_iter:", num_iter)
    # print("r_shape:", r.shape)
    if num_iter<filter_window_size:
        r_filtered=np.matmul(r.reshape(3,1),np.ones((1,filter_window_size)))
        q_filtered=np.dot(q_quat.reshape(4,1),np.ones((1,filter_window_size)))
        v_filtered=np.dot(v.reshape(3,1),np.ones((1,filter_window_size)))
        w_filtered=np.dot(w.reshape(3,1),np.ones((1,filter_window_size)))

        q_filtered_avg_out[:,j] = q_quat
    else:
        r_filtered[:,0:filter_window_size-1]=r_filtered[:,1:filter_window_size]
        q_filtered[:,0:filter_window_size-1]=q_filtered[:,1:filter_window_size]
        v_filtered[:,0:filter_window_size-1]=v_filtered[:,1:filter_window_size]
        w_filtered[:,0:filter_window_size-1]=w_filtered[:,1:filter_window_size]

        r_filtered[:,filter_window_size-1]=r
        q_filtered[:,filter_window_size-1]=q_quat
        v_filtered[:,filter_window_size-1]=v
        w_filtered[:,filter_window_size-1]=w

        for j_iter in range(0,filter_window_size):
            M_mat_quat = M_mat_quat+np.matmul(q_filtered[:,j_iter].reshape(4,1),q_filtered[:,j_iter].reshape(1,4))
            # print("M_mat_quat:", M_mat_quat)
        # print("M_mat_quat:", M_mat_quat)
        w_eig, vr_eig = np.linalg.eig(M_mat_quat)
        # print("vr_eig:", vr_eig)
        q_filtered_avg_undecided = vr_eig[:, np.argmax(w_eig)]
        error_q_avg_plus = q_filtered_avg_undecided-q_filtered_avg_prev_out[0:4,j-1]
        error_q_avg_minus = -q_filtered_avg_undecided-q_filtered_avg_prev_out[0:4,j-1]
       
        if np.dot(error_q_avg_plus,error_q_avg_plus)>np.dot(error_q_avg_minus,error_q_avg_minus):
            q_filtered_avg_out[:,j]=-q_filtered_avg_undecided


        else:
            q_filtered_avg_out[:,j]=q_filtered_avg_undecided
           

    q_filtered_avg_prev_out[0:4,j]=q_filtered_avg_out[:,j]
    # Averaging filter
    r_filtered_avg_out[:,j]= np.sum(r_filtered,1)/filter_window_size
    v_filtered_avg_out[:,j]= np.sum(v_filtered,1)/filter_window_size
    w_filtered_avg_out[:,j]= np.sum(w_filtered,1)/filter_window_size


    # u[0,j] = ubar[0,j]
    # u[1:4,j]=ubar[1:4,j]
    # u[0,j]=ubar[0,j] # Control

    (Rq, Eq, wIw, I_inv, fdw, taudw, vd_aero, wd_aero)=robobee_plant_aero.GetManipulatorDynamics(state_out[0:7,j], state_out[7:13,j])


    vd = vd_aero+ubar[0:3,j]#+np.dot(Rq,e3)*ubar[0,j]
    wd = wd_aero+np.dot(I_inv,ubar[3:6,j])
    q_eval = state_out[3:7,j] 

    

    wing_l[:,j]= r + np.dot(Rq,r_w-e2*0.012)
    wing_r[:,j]= r + np.dot(Rq,r_w+e2*0.012)
    vehicle_bottom[:,j] = r + np.dot(Rq,(-0.015+rw_l)*e3)
    vehicle_top[:,j] = r + np.dot(Rq,r_w)

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

    r_w_hat = np.zeros((3,3))
    r_w_hat[0,:] = np.array([        0,   -r_w[2],     r_w[1] ])
    r_w_hat[1,:] = np.array([   r_w[2],         0,    -r_w[0] ])
    r_w_hat[2,:] = np.array([  -r_w[1],    r_w[0],          0 ])
    
    Rq_r_w_hat = np.zeros((3,3))
    Rq_r_w_hat=np.dot(Rq,r_w_hat);

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
    vd_all[:,j] = vd
    wd_all[:,j] = wd
    
    if circle_flag==0:
        x_f[j] = xf[0]+radius*math.cos(w_freq*time_out[j])
        y_f[j] = xf[1]+radius*math.sin(w_freq*time_out[j])
        z_f[j] = zd

        dx_f[j] = -radius*math.pow(w_freq,1)*math.sin(w_freq*time_out[j])
        dy_f[j] = radius*math.pow(w_freq,1)*math.cos(w_freq*time_out[j])
        dz_f[j] = 0
        
        ddx_f[j] = -radius*math.pow(w_freq,2)*math.cos(w_freq*time_out[j])
        ddy_f[j] = -radius*math.pow(w_freq,2)*math.sin(w_freq*time_out[j])
        ddz_f[j] = 0

        dddx_f[j] = radius*math.pow(w_freq,3)*math.sin(w_freq*time_out[j])
        dddy_f[j] = -radius*math.pow(w_freq,3)*math.cos(w_freq*time_out[j])
        ddddx_f[j] = radius*math.pow(w_freq,4)*math.cos(w_freq*time_out[j])
        ddddy_f[j] = radius*math.pow(w_freq,4)*math.sin(w_freq*time_out[j])
    elif circle_flag==1:
        # Example 2 : Lissajous curve a=1 b=2
        a=1;
        b=ratio_ab*a;
        

        x_f[j] = radius*math.sin(a*w_freq*time_out[j]+delta_lissajous)
        y_f[j] = radius*math.sin(b*w_freq*time_out[j])
        z_f[j] = zd
        # print("x_f:",x_f)
        # print("y_f:",y_f)
        dx_f[j] = radius*math.pow(a*w_freq,1)*math.cos(a*w_freq*time_out[j]+delta_lissajous)
        dy_f[j] = radius*math.pow(b*w_freq,1)*math.cos(b*w_freq*time_out[j])
        dz_f[j] = 0
        
        ddx_f[j] = -radius*math.pow(a*w_freq,2)*math.sin(a*w_freq*time_out[j]+delta_lissajous)
        ddy_f[j] = -radius*math.pow(b*w_freq,2)*math.sin(b*w_freq*time_out[j])
        ddz_f[j] = 0

        dddx_f[j] = -radius*math.pow(a*w_freq,3)*math.cos(a*w_freq*time_out[j]+delta_lissajous)
        dddy_f[j] = -radius*math.pow(b*w_freq,3)*math.cos(b*w_freq*time_out[j])
        ddddx_f[j] = radius*math.pow(a*w_freq,4)*math.sin(a*w_freq*time_out[j]+delta_lissajous)
        ddddy_f[j] = radius*math.pow(b*w_freq,4)*math.sin(b*w_freq*time_out[j])
    

    r_ref=np.array([x_f[j],y_f[j],z_f[j]])
    dr_ref=np.array([dx_f[j],dy_f[j],dz_f[j]])
    ddr_ref=np.array([ddx_f[j],ddy_f[j],ddz_f[j]])

    dist_CoM_ref[j] = math.sqrt(np.dot(r_ref[0:2]-r[0:2],r_ref[0:2]-r[0:2]))

    eta1 = r + ratio_rw_timevarying*np.dot(Rq,r_w) - np.array([x_f[j],y_f[j],z_f[j]])
    eta2 = v_vel + ratio_rw_timevarying*np.dot(Rq,wr_w) - np.array([dx_f[j],dy_f[j],dz_f[j]])
    eta5 = math.atan2(Rqe1_x,Rqe1_y)-math.pi/8
    
    x2y2 = (math.pow(Rqe1_x,2)+math.pow(Rqe1_y,2)) # x^2+y^2
    
    eta6_temp = np.zeros(3)     #eta6_temp = (ye2T-xe1T)/(x^2+y^2)
    eta6_temp = (Rqe1_y*e2.T-Rqe1_x*e1.T)/x2y2    
    # print("eta6_temp:", eta6_temp)
    # Body frame w  ( multiply R)
    eta6 = np.dot(eta6_temp,np.dot(-Rqe1_hat,np.dot(Rq,w)))

    eta1_all[:,j] = eta1
    eta2_all[:,j] = eta2
    
    # eta = np.hstack([eta1, eta2, eta5, eta6])
    eta = np.hstack([eta1, eta2])

    mu_1 = np.dot(Rq, np.dot(e3_hat,w))

    mu_all = np.hstack([mu_1,eta])
    
    eta_norm_all[j] = np.dot(eta,eta)
    Vx_all[j] =1./1.*np.dot(eta,np.dot(Mout,eta));
        
    A_fl = np.zeros((3,4))
    A_fl[0:3,0] = Rqe3
    A_fl[0:3,1:4] = -ratio_rw_timevarying*np.dot(Rq_r_w_hat,I_inv)
    
    A_fl_pseudo = np.zeros((3,3))
    A_fl_pseudo = np.dot(A_fl,A_fl.T)
    A_fl_pseudo_inv = np.linalg.inv(A_fl_pseudo);

    A_fl_inv = np.dot(A_fl.T,A_fl_pseudo_inv)

    eta3_temp =  vd_aero+ratio_rw_timevarying*np.dot(Rdot,wr_w)-ratio_rw_timevarying*np.dot(Rq_r_w_hat,wd_aero)- np.array([ddx_f[j],ddy_f[j],0])#-np.dot(Rq_r_w_hat,eta3_damping)
    
     # Testing the feedback 
    mu = np.zeros(3)
    # epsilonnn=1e-1
    # mu[0:3] = -1/math.pow(epsilonnn,2)*eta1 -2/math.pow(epsilonnn,2)*eta2
    # FL controller
    R_inv = np.linalg.inv(Rout);
    k = np.matmul(R_inv,np.matmul(Bout.T,Mout))
    # k = np.matmul(Bout.T,Mout)
    mu = -1./1.*np.matmul(k,eta)
    v_aero = -eta3_temp + mu;

    u[0:4,j] = np.dot(A_fl_inv,v_aero)

    eta3 = vd_aero+ratio_rw_timevarying*np.dot(Rdot,wr_w)-ratio_rw_timevarying*np.dot(Rq_r_w_hat,wd_aero) + np.dot(A_fl,u[0:4,j]) - np.array([ddx_f[j],ddy_f[j],0])
    eta3_all[:,j] = eta3
    
    u_all[0:4,j] = u[0:4,j]            # Output Feedback controller to comare the result with CLF-QP solver
    u_all[1:4,j] = u[1:4,j]#-k_yaw*w[2]*e3
    
    u_temp=np.dot(A_fl_inv,-eta3_temp)

    e3_hat_sq = np.dot(e3_hat,e3_hat)
    # d_w_all[:,j] = -np.dot(I_inv,wIw)+np.dot(I_inv,u_temp[1:4])
    # d_w_all[:,j] = -np.dot(I_inv,wIw)-np.dot(w,e3)*np.dot(e3_hat,w)-np.dot(np.dot(e3_hat,e3_hat),np.dot(I_inv,wIw))+g/rw_l*np.dot(e3_hat,np.dot(Rq.T,e3))
    d_w_all[:,j] = np.array([w[1]*w[2], -w[0]*w[2], (Ixx-Iyy)/Izz*w[0]*w[1]])+g/rw_l*np.dot(e3_hat,np.dot(Rq.T,e3))+ratio_rw*np.dot(I_inv,np.dot(r_w_hat,np.dot(r_w_hat,w)))
    d_w_all[:,j] = np.array([w[1]*w[2], -w[0]*w[2], (Ixx-Iyy)/Izz*w[0]*w[1]])+g/(rw_l*ratio_rw_timevarying)*np.dot(e3_hat,np.dot(Rq.T,e3))-bw*ratio_rw/(m*rw_l*ratio_rw_timevarying)*np.dot(e3_hat,wr_w)
    # ISS_temp = (bw/m*np.eye(6)-Mout/2.)
    # ISS_tmep_cross = np.dot(np.matmul(Bout.T,ISS_temp),eta);

    ISS_temp = bw/m*Bout.T-k/1. #np.dot(np.matmul(Bout.T,ISS_temp),eta);
    ISS_tmep_cross = np.dot(ISS_temp,eta);

    # d_w_all[:,j] = d_w_all[:,j] + 1/(rw_l*ratio_rw_timevarying)*np.dot(np.dot(e3_hat,Rq.T),ISS_tmep_cross)-k_yaw/Izz*w[2]*e3
    # d_w_all[:,j] = wd_aero+np.dot(I_inv,u_all[1:4,j])
    # print("error", ISS_tmep_cross-(bw/m*eta2+mu))
    d_w_all[:,j] = d_w_all[:,j]+1/(rw_l*ratio_rw_timevarying)*np.dot(np.dot(e3_hat,Rq.T),ISS_tmep_cross)-k_yaw/Izz*w[2]*e3
    perturb[:,j] = bw/(m*rw_l*ratio_rw_timevarying)*np.dot(np.dot(e3_hat,Rq.T),dr_ref)+1./(rw_l*ratio_rw_timevarying)*np.dot(np.dot(e3_hat,Rq.T),ddr_ref)
    d_w_all[:,j] = d_w_all[:,j]+perturb[:,j]

    fdw_all_output[:,j] = -bw*(eta2-ratio_rw*np.dot(Rq,wr_w))
    #+(1-ratio_rw)*bw*np.dot(I_inv,np.dot(e3_hat_sq,w))*rw_l*rw_l
    # if j>0:
       
    if (time_out[j]-time_out[j-1])>0:

        d_eta1_all[:,j]= (eta1_all[:,j]-eta1_all[:,j-1])/(time_out[j]-time_out[j-1])
        d_eta2_all[:,j]= (eta2_all[:,j]-eta2_all[:,j-1])/(time_out[j]-time_out[j-1])

    else:

        print("time step: ",(time_out[j]-time_out[j-1]) )
        print("j", j)
        print("timeeout[j],", time_out[j])

    # # Lyapunov parts!

    # v_equiv=v_vel#-(-m*g/bw*e3)
    # V_equiv=v_equiv+np.dot(Rq,wr_w)
    # w_equiv = np.cross(e3,Iw)
    # w_equiv2 = np.cross(e3,w)
    # R_equiv = np.cross(e3,Rqe3)

    # d_w_equiv = np.cross(e3,Iwd)
    # d_w_equiv2 = np.cross(e3,wd)
    # q_eval_vec = q_eval[1:4]
    # q_eval_vec_norm=np.sqrt(np.dot(q_eval_vec.T,q_eval_vec))
    # q_eval_vec_normalized = q_eval_vec/q_eval_vec_norm
    # q_equiv = q_eval_vec_normalized-e3

    # temp_V = -bw*rw_l*np.cross(Rqe3,v_vel)
    # if j==0:
    #     int_V2[j] = np.dot(w,np.dot(Rq.T,temp_V))
    #     int_CoM[j]= np.dot(v_vel,fdw)
    #     int_Rot[j]= np.dot(w,taudw)
    # else:
    #     int_V2[j] = int_V2[j-1]+(time_out[j]-time_out[j-1])*(np.dot(w,np.dot(Rq.T,temp_V)))
    #     int_CoM[j] = int_CoM[j-1]+(time_out[j]-time_out[j-1])*(np.dot(v_vel,fdw))
    #     int_Rot[j] = int_Rot[j-1]+(time_out[j]-time_out[j-1])*(np.dot(w,taudw))

    # e3_hat_square = -np.dot(e3_hat,e3_hat)
    # V1_lyapu[j] = 1.*(np.dot(w_equiv2,w_equiv2))#1./2.*np.dot(w_equiv2,w_equiv2)#math.pow(m,2)*math.pow(g,2)/math.pow(bw,1)+np.dot(v_vel,fdw)+np.dot(w,taudw)#1./2.*bw*np.dot(V_equiv,V_equiv)
    # V2_lyapu[j] = 1./2.*np.dot(w,Iw)
    # V3_lyapu[j] =  g/(rw_l*ratio_rw_timevarying)*(1-np.dot(e3,np.dot(Rq,e3)))#m*g*rw_l*np.dot(e3,np.dot(Rq.T,np.dot(Rqe3_hat,e3)))#g/rw_l*(1-np.dot(e3,Rqe3)) #m*g*rw_l*(1-np.dot(e3,Rqe3))
    # Kinetic_all[j]= V1_lyapu[j]/2.+V3_lyapu[j]+Vx_all[j]*kq
    # Kinetic_all_compare[j]= V1_lyapu[j]+V2_lyapu[j]
    
    # mu_1_norm[j]=np.dot(w,w) #(V1_lyapu[j]) #math.sqrt(V1_lyapu[j])

    # d_Kinetic_all[j]=m*np.dot(v_equiv,vd) + np.dot(w,Iwd) #np.dot(Iw, Iwd) #+1./2.*np.dot(vd,vd) 
    # vd_Kinetic_all[j]= -np.dot(mu_all,np.dot(P_ISS,mu_all))-1./1.*((bw/(m*rw_l*ratio_rw_timevarying))*np.dot(mu_1,dr_ref)+(1./(rw_l*ratio_rw_timevarying))*np.dot(mu_1,ddr_ref))#np.dot(w_equiv2,d_w_equiv2) - g/(rw_l*ratio_rw_timevarying)*np.dot(e3,np.dot(Rdot,e3))-np.dot(eta,np.dot(Q,eta))*kq/2
    # vd_Kinetic_upper_all[j]= -min_eval_P_ISS*(eta_norm_all[j]*eta_norm_all[j]+mu_1_norm[j]*(mu_1_norm[j]-mu_M))

    # V_lyapu_upper[j] = vd_Kinetic_all[j] #math.exp(-bw/m*time_out[j])*V1_lyapu[0]+math.exp(-bw*rw_l*time_out[j])*V2_lyapu[0]
    # # Kinetic_all[j]=1./2.*np.dot(fdw,fdw)
    # # print("fdw :", fdw)
    # Total_energy[j] = 1./2.*m*np.dot(v_vel,v_vel) + m*g*r[2] + 1./2.*np.dot(w,Iw)
    # Work_done[j] = int_CoM[j]+int_Rot[j]
    # Conservation_energy[j] = Total_energy[j]-Work_done[j]
    # Angular_momentum[j] =  1./2.*np.dot(w,Iw)

    # # V1_lyapu[j] = m*np.dot(v_vel,vd)+np.dot(w,Iwd)+m*g*v_vel[2]#np.dot(v_vel,fdw)+np.dot(w,taudw)# Total_energy[0]+Work_done[j]
    # if (time_out[j]-time_out[j-1])>0:

    #     d_V_all[j]= (Kinetic_all[j]-Kinetic_all[j-1])/(time_out[j]-time_out[j-1])

    # Inverse mapping


    u_ForceTorque=u_all[0:4,j]

    vicon_sampled_time =truncate(time_out[j]*1000,2)%2
    # print("sampled t: ", vicon_sampled_time)
    
    if vicon_sampled_time==0 and sample_flag==0:
        u_sampled_plot = u_ForceTorque
        sample_flag=1
    if vicon_sampled_time==1 and sample_flag==1:
        sample_flag=0
    
    u_ForceTorque_500Hz[0:4,j]=u_sampled_plot

    u_ForceTorque=u_ForceTorque_500Hz[0:4,j]

    F_desired_thrust = u_ForceTorque[0]*m
    torque_desired = u_ForceTorque[1:4]

    F_l_avg_coeff=1/2.*rho*beta_geometry*CL*(math.pow(wave_weight*w_resonance,2)
        +math.pow(half_wave_weight*0.5*w_resonance,2)
        +math.pow(twice_wave_weight*2*w_resonance,2)
        +math.pow(third_wave_weight*3*w_resonance,2))/Force_scale

    F_r_avg_coeff=1/2.*rho*beta_geometry*CL*(math.pow(wave_weight*w_resonance,2)
        +math.pow(half_wave_weight*0.5*w_resonance,2)
        +math.pow(twice_wave_weight*2*w_resonance,2)
        +math.pow(third_wave_weight*3*w_resonance,2))/Force_scale

    if F_desired_thrust/m<=0:
        print("F_desired_thrust", F_desired_thrust/m)
        F_desired_thrust=0;
        Phi_off_l=0;
        Phi_off_r=0;
        Phi_amp_l=0;
        Phi_amp_r=0;
    else:    
        if math.fabs((torque_desired[1])/F_desired_thrust/r_cp)>1:
            print("F_desired_thrust", F_desired_thrust/m)
            print("sin(phi_off)", (torque_desired[1])/F_desired_thrust/r_cp)

        Phi_off_l_estimate = math.asin((torque_desired[1])/F_desired_thrust/r_cp)
        Phi_amp_sq = (F_desired_thrust*r_cp*math.cos(Phi_off_l_estimate)+torque_desired[0])/(r_cp*math.cos(Phi_off_l_estimate)*F_l_avg_coeff)/2.
        Phi_amp_sq_cross = (F_desired_thrust*r_cp*math.cos(Phi_off_l_estimate)-torque_desired[0])/(r_cp*math.cos(Phi_off_l_estimate)*F_l_avg_coeff)/2.
        
        Phi_amp_avg = (math.sqrt(Phi_amp_sq)+math.sqrt(Phi_amp_sq_cross))/2.
        Phi_amp_diff = (math.sqrt(Phi_amp_sq)-math.sqrt(Phi_amp_sq_cross))/2.


        Phi_off_l = Phi_off_l_estimate*gain_offset
        Phi_off_r = -Phi_off_l_estimate*gain_offset
        
        Phi_amp_l = -(Phi_amp_avg-Phi_amp_diff*gain_diff)*gain_amp#-math.pi/8.5-math.pi/8.5*(zd-r_filtered_avg[2])*0.+math.pi/8.5*(v_filtered_avg[2])*0+math.pi/8.5*(w_filtered_avg[0])*0.05;
        Phi_amp_r = (Phi_amp_avg+Phi_amp_diff*gain_diff)*gain_amp#math.pi/8.5+math.pi/8.5*(zd-r_filtered_avg[2])*0.-math.pi/8.5*(v_filtered_avg[2])*0+math.pi/8.5*(w_filtered_avg[0])*0.05;
    

    ## Harmonic signal


    # Phi_off_l = mean_stroke_l
    # Phi_off_r = mean_stroke_r
    
    # Phi_amp_l = -math.pi/8.5+math.pi/17;
    # Phi_amp_r = math.pi/8.5+math.pi/17;
    
    t= time_out[j]
    
    
    V_l[j] = 1/Transfer_mag_zero*Phi_off_l + Phi_amp_l*(wave_weight/Transfer_mag_reso*math.sin(w_resonance*t)             
            +half_wave_weight/Transfer_mag_reso_half*math.sin(half_wave_freq*t)              
            +twice_wave_weight/Transfer_mag_reso_twice*math.sin(twice_wave_freq*t)
            +third_wave_weight/Transfer_mag_reso_third*math.sin(third_wave_freq*t))

    V_r[j] = 1/Transfer_mag_zero*Phi_off_r + Phi_amp_r*(wave_weight/Transfer_mag_reso*math.sin(w_resonance*t+torque_async)
            +half_wave_weight/Transfer_mag_reso_half*math.sin(half_wave_freq*t+torque_async) 
            +twice_wave_weight/Transfer_mag_reso_twice*math.sin(twice_wave_freq*t+torque_async)
            +third_wave_weight/Transfer_mag_reso_third*math.sin(third_wave_freq*t+torque_async))


    Rq_l = np.zeros((3,3))
    Rq_l[0,:] = np.array([ math.cos(phi_l[j]),   -math.sin(phi_l[j]),     0 ])
    Rq_l[1,:] = np.array([ math.sin(phi_l[j]),    math.cos(phi_l[j]),     0 ])
    Rq_l[2,:] = np.array([               0,                  0,     1 ])
    Rq_le2 = np.dot(Rq_l,e2)
    RqRq_le2 = np.dot(Rq,Rq_le2)
    w_hatRq_le2 = np.dot(w_hat, Rq_le2)
    Rqw_hatRq_le2 = np.dot(Rq, w_hatRq_le2)

    Rq_l_avg = np.zeros((3,3))
    Rq_l_avg[0,:] = np.array([ math.cos(Phi_off_l),   -math.sin(Phi_off_l),     0 ])
    Rq_l_avg[1,:] = np.array([ math.sin(Phi_off_l),    math.cos(Phi_off_l),     0 ])
    Rq_l_avg[2,:] = np.array([               0,                  0,     1 ])
    Rq_l_avge2 = np.dot(Rq_l_avg,e2)

    Rq_r_avg = np.zeros((3,3))
    Rq_r_avg[0,:] = np.array([ math.cos(Phi_off_r),   -math.sin(Phi_off_r),     0 ])
    Rq_r_avg[1,:] = np.array([ math.sin(Phi_off_r),    math.cos(Phi_off_r),     0 ])
    Rq_r_avg[2,:] = np.array([               0,                  0,     1 ])
    Rq_r_avge2 = np.dot(Rq_r_avg,e2)


    # print("x:", x)
    # print("w_hatRq_le2:", w_hatRq_le2)

    Rq_r = np.zeros((3,3))
    Rq_r[0,:] = np.array([ math.cos(phi_r[j]),   -math.sin(phi_r[j]),     0 ])
    Rq_r[1,:] = np.array([ math.sin(phi_r[j]),    math.cos(phi_r[j]),     0 ])
    Rq_r[2,:] = np.array([               0,                  0,     1 ])
    Rq_re2 = np.dot(Rq_r,e2)
    RqRq_re2 = np.dot(Rq,Rq_re2)
    w_hatRq_re2 = np.dot(w_hat, Rq_re2)
    Rqw_hatRq_re2 = np.dot(Rq, w_hatRq_re2)

    dphi_l_hat = np.zeros((3,3))
    dphi_l_hat[0,:] = np.array([      0,   -dphi_l[j],     0 ])
    dphi_l_hat[1,:] = np.array([ dphi_l[j],         0,     0 ])
    dphi_l_hat[2,:] = np.array([      0,         0,     0 ])

    dphi_l_hate2 = np.dot(dphi_l_hat, e2)
    Rq_ldphi_l_hate2 = np.dot(Rq_l, dphi_l_hate2)
    RqRq_ldphi_l_hate2 = np.dot(Rq, Rq_ldphi_l_hate2)

    dphi_r_hat = np.zeros((3,3))
    dphi_r_hat[0,:] = np.array([      0,   -dphi_r[j],     0 ])
    dphi_r_hat[1,:] = np.array([ dphi_r[j],         0,     0 ])
    dphi_r_hat[2,:] = np.array([      0,         0,     0 ])
    
    dphi_r_hate2 = np.dot(dphi_r_hat, e2)
    Rq_rdphi_r_hate2 = np.dot(Rq_r, dphi_r_hate2)
    RqRq_rdphi_r_hate2 = np.dot(Rq, Rq_rdphi_r_hate2)

    # print("dphi_r_hat:", dphi_r_hat)

    Rq_psi_l = np.zeros((3,3))
    Rq_psi_l[0,:] = np.array([  math.cos(psi_l[j]),   0, math.sin(psi_l[j])])
    Rq_psi_l[1,:] = np.array([                0,      0,  0             ])
    Rq_psi_l[2,:] = np.array([ -math.sin(psi_l[j]),   0, math.cos(psi_l[j])])

    Rq_psi_r = np.zeros((3,3))
    Rq_psi_r[0,:] = np.array([  math.cos(psi_r[j]),   0, math.sin(psi_r[j])])
    Rq_psi_r[1,:] = np.array([                0,      0,  0             ])
    Rq_psi_r[2,:] = np.array([ -math.sin(psi_r[j]),   0, math.cos(psi_r[j])])

    # Iw = np.dot(I,w)
    # r_w = rw_l*e3; # Length to the wing from CoM
    # wr_w = np.cross(w,r_w) # w x r
    # wr_w_hat = np.dot(w_hat,r_w)
    # # print("wr_w-wr_w_hat", wr_w-wr_w_hat)

    # wwr_w = np.cross(w,wr_w) # w x (w x r)

    # Wing angle

    eta = np.zeros(3)
    eta_l = np.zeros(3)
    eta_r = np.zeros(3)    

    eta = r+Rqe3*rw_l;
    eta_l = eta+np.dot(Rq,Rq_le2)*r_cp  # phi_l positive : clockwise
    eta_r = eta-np.dot(Rq,Rq_re2)*r_cp # phi_r positive : clockwise

    # Wing velocity

    deta = np.zeros(3)
    deta_l = np.zeros(3)
    deta_r = np.zeros(3)    

    deta = v + np.dot(Rq,wr_w)
    deta_l = deta + Rqw_hatRq_le2*r_cp + RqRq_ldphi_l_hate2*r_cp 
    deta_r = deta - Rqw_hatRq_re2*r_cp - RqRq_rdphi_r_hate2*r_cp 

    
    # Projection to wing stroke plane
    P_proj_wing_l = np.zeros((3,3))
    P_proj_wing_l = np.eye(3)-np.outer(RqRq_le2, RqRq_le2)
    # print("outer_before:", RqRq_le2)
    # print("outer:", P_proj_wing_l)
    P_proj_wing_r = np.zeros((3,3))
    P_proj_wing_r = np.eye(3)-np.outer(RqRq_re2, RqRq_re2)

    ## Decomposing the wing velocity in the projected domain
    
    proj_deta_l_body                = np.dot(P_proj_wing_l,deta)
    proj_deta_l_body_wing_couple    = np.dot(P_proj_wing_l,Rqw_hatRq_le2)
    proj_deta_l_wing                = np.dot(P_proj_wing_l,RqRq_ldphi_l_hate2)

    proj_deta_r_body                =  np.dot(P_proj_wing_r,deta)
    proj_deta_r_body_wing_couple    = -np.dot(P_proj_wing_r,Rqw_hatRq_re2)
    proj_deta_r_wing                = -np.dot(P_proj_wing_r,RqRq_rdphi_r_hate2)

    # nominal velocity in wing stroke plane for CoP
    # print("deta :", deta)
    # print("deta_l :", deta_l)
    # print("deta_r :", deta_r)
    v_ortho_l = np.dot(P_proj_wing_l, deta_l)
    v_ortho_r = np.dot(P_proj_wing_r, deta_r) 
    
    v_ortho_l_norm_sq = np.dot(v_ortho_l,v_ortho_l)
    v_ortho_r_norm_sq = np.dot(v_ortho_r,v_ortho_r)

    v_ortho_l_norm = math.sqrt(v_ortho_l_norm_sq)
    v_ortho_r_norm = math.sqrt(v_ortho_r_norm_sq)

    if v_ortho_l_norm==0 or v_ortho_r_norm==0 :
        v_ortho_l_normalized = np.zeros(3)
        v_ortho_r_normalized = np.zeros(3)
    else:
        v_ortho_l_normalized = v_ortho_l/v_ortho_l_norm
        v_ortho_r_normalized = v_ortho_r/v_ortho_r_norm 

    # print("v_ortho_l_normalized :", v_ortho_l_normalized)
    # print("v_ortho_r_normalized :", v_ortho_r_normalized)   
    # print("v_ortho_l_norm_sq :", v_ortho_l_norm_sq)
    # print("v_ortho_r_norm_sq :", v_ortho_r_norm_sq)   

    # Hinge angle in the global frame

    psi_l_wing = np.dot(Rq_psi_l,-e3) 
    psi_l_body = np.dot(Rq_l,psi_l_wing) 
    psi_l_world = np.dot(Rq,psi_l_body) 

    psi_r_wing = np.dot(Rq_psi_r,-e3) 
    psi_r_body = np.dot(Rq_r,psi_r_wing) 
    psi_r_world = np.dot(Rq,psi_r_body) 


    AoA_inner_product_l = np.dot(v_ortho_l_normalized, -psi_l_world) # Based on Kevin Science Robobtics supplement paper
    AoA_inner_product_r = np.dot(v_ortho_r_normalized, -psi_r_world)
    
    AoA_l[j] = math.acos(AoA_inner_product_l)
    AoA_r[j] = math.acos(AoA_inner_product_r)
    # print("AoA_l:", AoA_l)
    # print("AoA_r:", AoA_r)
    # nominal velocity in wing stroke plane for each blade

    FL_l_r=np.zeros(num_blade)
    FL_r_r=np.zeros(num_blade)
    FD_l_r = np.zeros((3,num_blade))
    FD_r_r = np.zeros((3,num_blade))
    FL_l_vec=np.zeros((3,num_blade))
    FL_r_vec=np.zeros((3,num_blade))
    FD_l_vec=np.zeros((3,num_blade))
    FD_r_vec=np.zeros((3,num_blade))
    F_l_vec=np.zeros((3,num_blade))
    F_r_vec=np.zeros((3,num_blade))
    tau_aero_l =np.zeros((3,num_blade))
    tau_aero_r =np.zeros((3,num_blade))

    chord_r_CoP_z_l=np.zeros(num_blade)
    chord_r_CoP_z_r=np.zeros(num_blade)
    
    chord_r_CoP_world_l=np.zeros((3,num_blade))
    chord_r_CoP_world_r=np.zeros((3,num_blade))

    chord_r_cp_from_root_world_l=Rqe3*rw_l+np.dot(Rq,Rq_l_avge2)*r_cp
    chord_r_cp_from_root_world_r=Rqe3*rw_l-np.dot(Rq,Rq_r_avge2)*r_cp


    for i_iter in range(0,num_blade):
        chord_r_from_root = wing_xr + chordlength_r[i_iter]

        chord_r_from_root_world_l=Rqe3*rw_l+np.dot(Rq,Rq_le2)*chord_r_from_root
        chord_r_from_root_world_r=Rqe3*rw_l-np.dot(Rq,Rq_re2)*chord_r_from_root

        v_ortho_l_r =  proj_deta_l_body + (proj_deta_l_body_wing_couple + proj_deta_l_wing)*chord_r_from_root
        v_ortho_r_r =  proj_deta_r_body + (proj_deta_r_body_wing_couple + proj_deta_r_wing)*chord_r_from_root

        v_ortho_l_r_norm_sq = np.dot(v_ortho_l_r, v_ortho_l_r)
        v_ortho_r_r_norm_sq = np.dot(v_ortho_r_r, v_ortho_r_r)

        # print("v_ortho_r_r_norm_sq", v_ortho_r_r_norm_sq )
        # print("v_ortho_l_r_norm_sq", v_ortho_l_r_norm_sq )

        v_ortho_l_r_norm = math.sqrt(v_ortho_l_r_norm_sq)
        v_ortho_r_r_norm = math.sqrt(v_ortho_r_r_norm_sq)

        if v_ortho_l_r_norm==0 or v_ortho_r_r_norm==0 :
            v_ortho_l_r_normalized = np.zeros(3)
            v_ortho_r_r_normalized = np.zeros(3)
        else:
            v_ortho_l_r_normalized = v_ortho_l_r/v_ortho_l_r_norm
            v_ortho_r_r_normalized = v_ortho_r_r/v_ortho_r_r_norm 

        AoA_inner_product_l_r = np.dot( v_ortho_l_r_normalized, -psi_l_world) # pointing downward : c vector
        AoA_inner_product_r_r = np.dot( v_ortho_r_r_normalized, -psi_r_world) # pointing downward : c vector

        AoA_l_r = math.acos(AoA_inner_product_l_r)
        AoA_r_r = math.acos(AoA_inner_product_r_r)

        # Center of Pressure for each blade : Kevin IROS 2016
        chord_r_CoP_z_l[i_iter] = chordlength_yLE[i_iter]-(0.25 + 0.25/(1+math.exp(5.*(1.-4.*AoA_l_r/math.pi))))*chordlength_cr[i_iter]
        chord_r_CoP_z_r[i_iter] = chordlength_yLE[i_iter]-(0.25 + 0.25/(1+math.exp(5.*(1.-4.*AoA_r_r/math.pi))))*chordlength_cr[i_iter]

        chord_r_CoP_world_l[0:3,i_iter] = chord_r_from_root_world_l+psi_l_world*chord_r_CoP_z_l[i_iter]
        chord_r_CoP_world_r[0:3,i_iter] = chord_r_from_root_world_r+psi_r_world*chord_r_CoP_z_r[i_iter]

        # print("Psi ", psi_l)
        # print("AoA_l_r", AoA_l_r)
        CL_l = CL0*math.sin(2.*AoA_l_r)
        CL_r = CL0*math.sin(2.*AoA_r_r)

        CD_l = (CD0+CDmax)/2.- (CDmax-CD0)/2.*math.cos(2.*AoA_l_r)
        CD_r = (CD0+CDmax)/2.- (CDmax-CD0)/2.*math.cos(2.*AoA_r_r)

        # print("CL_l:", CL_l)

        # print("CD_l:", CD_l)

        FL_l_r[i_iter] = 1./2.*rho*CL_l*v_ortho_l_r_norm_sq*chordlength_cr[i_iter]*blade_width
        FL_r_r[i_iter] = 1./2.*rho*CL_r*v_ortho_r_r_norm_sq*chordlength_cr[i_iter]*blade_width

        # print("FL_l_r", FL_l_r[i_iter])

        # print("FL_r_r", FL_r_r[i_iter])

        FD_l_r[0:3,i_iter] = -1./2.*rho*CD_l*v_ortho_l_r_norm_sq*v_ortho_l_r_normalized*chordlength_cr[i_iter]*blade_width
        FD_r_r[0:3,i_iter] = -1./2.*rho*CD_r*v_ortho_r_r_norm_sq*v_ortho_r_r_normalized*chordlength_cr[i_iter]*blade_width
        
        FL_l_vec[0:3,i_iter] = FL_l_r[i_iter]*np.dot(Rq,e3)/Force_scale
        FL_r_vec[0:3,i_iter] = FL_r_r[i_iter]*np.dot(Rq,e3)/Force_scale
        FD_l_vec[0:3,i_iter] = FD_l_r[0:3,i_iter]/Force_scale
        FD_r_vec[0:3,i_iter] = FD_r_r[0:3,i_iter]/Force_scale
        
        # print("FL_l_vec in body:", np.dot(Rq.T,FL_l_vec[0:3,i_iter]))
        # print("FD_l_vec in body:", np.dot(Rq.T,FD_l_vec[0:3,i_iter]))


        F_l_vec[0:3,i_iter] = FL_l_vec[0:3,i_iter] + FD_l_vec[0:3,i_iter]
        F_r_vec[0:3,i_iter] = FL_r_vec[0:3,i_iter] + FD_r_vec[0:3,i_iter]

        tau_aero_l[0:3,i_iter] = np.cross(chord_r_CoP_world_l[0:3,i_iter],F_l_vec[0:3,i_iter])
        tau_aero_l[0:3,i_iter] = np.dot(Rq.T,tau_aero_l[0:3,i_iter])

        tau_aero_r[0:3,i_iter] = np.cross(chord_r_CoP_world_r[0:3,i_iter],F_r_vec[0:3,i_iter])
        tau_aero_r[0:3,i_iter] = np.dot(Rq.T,tau_aero_r[0:3,i_iter])

    # print("CL_l:", CL_l)  

    # print("CD_l:", CD_l)    

    # FL_l = 1./2.*rho*beta_geometry/(r_cp*r_cp)*CL*v_ortho_l_norm_sq
    # FL_r = 1./2.*rho*beta_geometry/(r_cp*r_cp)*CL*v_ortho_r_norm_sq
    
    FL_l = np.sum(FL_l_r)
    FL_r = np.sum(FL_r_r)
    FD_l = np.sum(FD_l_r)
    FD_r = np.sum(FD_r_r)


    # print("FL_l:", FL_l)
    # print("FD_l:", FD_l)
    
    Thrust_l[j] = FL_l/(m*Force_scale)
    Thrust_r[j] = FL_r/(m*Force_scale)
    Thrust[j] = Thrust_l[j]+Thrust_r[j]

    # print("Thrust left:", Thrust_l)
    # print("Thrust right:", Thrust_r)
    # print("Thrust:", Thrust)

    tau_aero_l_sum = np.sum(tau_aero_l, axis=1)
    tau_aero_r_sum = np.sum(tau_aero_r, axis=1)
    tau_aero_sum[0:3,j] = tau_aero_l_sum+tau_aero_r_sum
    # print("tau_aero_l: ", tau_aero_l_sum)
    # print("tau_aero_r: ", tau_aero_r_sum)
    # print("tau_aero: ", tau_aero)


    F_aero_sum[0:3,j] = (np.sum(F_l_vec,1)+np.sum(F_r_vec,1))/m

    # print("F_aero_sum", F_aero_sum[0:3,j])

    Thrust_actual[j] = np.dot(np.dot(Rq,e3),F_aero_sum[0:3,j])
    # print("Thrust", Thrust)

    u[0:3]=F_aero_sum

    u_all[3:6,j] = tau_aero_sum[0:3,j] 
    # u_all[0,j] = Thrust[j] #1.5*g;

    # print("V_l:", V_l[j])
    # print("phi_l:", phi_l[j])
    # u[4] = V_l
    # u[5] = V_r
    # dpsi_l = math.pi/3.5*math.pow(w_resonance/1.,1)*math.cos(w_resonance/1.*t)
    # dpsi_r = math.pi/3.5*math.pow(w_resonance/1.,1)*math.cos(w_resonance/1.*t+torque_async)
    # u[6] = dpsi_l
    # u[7] = dpsi_r
    
    # zd=xf[2]

    # Computed avg lift force

    F_l_avg_coeff=1/2.*rho*beta_geometry*CL*(math.pow(wave_weight*w_resonance,2)
        +math.pow(half_wave_weight*0.5*w_resonance,2)
        +math.pow(twice_wave_weight*2*w_resonance,2)
        +math.pow(third_wave_weight*3*w_resonance,2))/Force_scale

    F_r_avg_coeff=1/2.*rho*beta_geometry*CL*(math.pow(wave_weight*w_resonance,2)
        +math.pow(half_wave_weight*0.5*w_resonance,2)
        +math.pow(twice_wave_weight*2*w_resonance,2)
        +math.pow(third_wave_weight*3*w_resonance,2))/Force_scale

    F_l_avg[j] = F_l_avg_coeff*math.pow(Phi_amp_l,2)           
    F_r_avg[j] = F_r_avg_coeff*math.pow(Phi_amp_r,2) 
    
    F_l_vec_avg[0:3,j]=F_l_avg[j]*np.dot(Rq,e3)
    F_r_vec_avg[0:3,j]=F_r_avg[j]*np.dot(Rq,e3)
    Thrust_l_avg[j] = F_l_avg[j]/(m)            
    Thrust_r_avg[j] = F_r_avg[j]/(m)             
    # print("Thrust_l_avg", Thrust_l_avg[j])       

    tau_aero_l_avg[0:3,j] = np.cross(chord_r_cp_from_root_world_l,F_l_vec_avg[0:3,j])
    tau_aero_l_avg[0:3,j] = np.dot(Rq.T,tau_aero_l_avg[0:3,j])

    tau_aero_r_avg[0:3,j] = np.cross(chord_r_cp_from_root_world_r,F_r_vec_avg[0:3,j])
    tau_aero_r_avg[0:3,j] = np.dot(Rq.T,tau_aero_r_avg[0:3,j])

    # print("Tau_aero_pitch predict:", r_cp*math.sin(Phi_off_l)*(F_l_avg[j]+F_r_avg[j]))
    # print("Tau_aero_pitch:", tau_aero_l_avg[1,j]+tau_aero_r_avg[1,j])

    # Inverse mapping from force to wing trajectory
    Phi_off_l_estimate = math.asin((tau_aero_r_avg[1,j]+tau_aero_l_avg[1,j])/(F_l_avg[j]+F_r_avg[j])/r_cp)
    
    # print('Phi_off_l_estimate', Phi_off_l_estimate)
    # print('Phi_off_l', Phi_off_l)
    Phi_amp_sq = ((F_l_avg[j]+F_r_avg[j])*r_cp*math.cos(Phi_off_l_estimate)+(tau_aero_r_avg[0,j]+tau_aero_l_avg[0,j]))/(r_cp*math.cos(Phi_off_l_estimate)*F_l_avg_coeff)/2.
    Phi_amp_sq_cross = ((F_l_avg[j]+F_r_avg[j])*r_cp*math.cos(Phi_off_l_estimate)-(tau_aero_r_avg[0,j]+tau_aero_l_avg[0,j]))/(r_cp*math.cos(Phi_off_l_estimate)*F_l_avg_coeff)/2.
    
    # print(tau_aero_r_avg[:,j]+tau_aero_l_avg[:,j])
    # print("tau_roll_estimate", r_cp*math.cos(Phi_off_l_estimate)*4*math.pi/17*math.pi/8.5*F_l_avg_coeff)
    # print("Phi_amp_sq:", Phi_amp_sq)
    # print("Phi_amp_sq_cross:", Phi_amp_sq_cross)
    
    Phi_amp_avg = (math.sqrt(Phi_amp_sq)+math.sqrt(Phi_amp_sq_cross))/2.
    Phi_amp_diff = (math.sqrt(Phi_amp_sq)-math.sqrt(Phi_amp_sq_cross))/2.

    # print("Phi_amp_avg+Phi_amp_diff", -(Phi_amp_avg+Phi_amp_diff)) 
    # print("Phi_amp_l", Phi_amp_l) 
    # print("Phi_amp_avg-Phi_amp_diff", (Phi_amp_avg-Phi_amp_diff)) 
    # print("Phi_amp_r", Phi_amp_r)

    # print("phi_off estimate:", Phi_off_l_estimate)
    # print("phi_off_l:", (Phi_off_l))
    # print("tau_aero_l_avg", tau_aero_l_avg[0:3,j]+tau_aero_r_avg[0:3,j])
    
# Vb_w_all = Ub_w_all*Ub_w
# Ub_w_all = Ub_w_all*mu_M
# Ub_w_all_new = Ub_w_all_new*mu_M_new

# np.savetxt('xalpha1_2to3.out', state_log.data()[0, :], delimiter=',')
# np.savetxt('yalpha1_2to3.out', state_log.data()[1, :], delimiter=',')
# np.savetxt('roll3_2to3.out', rpy[0, :], delimiter=',')
# np.savetxt('pitch3_2to3.out', rpy[1, :], delimiter=',')
# np.savetxt('yaw3_2to3.out', rpy[2, :], delimiter=',')

# np.savetxt('u1_alpha3_2to3.out', ubar[0, :], delimiter=',')
# np.savetxt('u2_alpha3_2to3.out', ubar[1, :], delimiter=',')
# np.savetxt('u3_alpha3_2to3.out', ubar[2, :], delimiter=',')
# np.savetxt('u4_alpha3_2to3.out', ubar[3, :], delimiter=',')

# u1_loaded_alpha3_2to3 = np.loadtxt('u1_alpha3_2to3.out')
# u2_loaded_alpha3_2to3 = np.loadtxt('u2_alpha3_2to3.out')
# u3_loaded_alpha3_2to3 = np.loadtxt('u3_alpha3_2to3.out')
# u4_loaded_alpha3_2to3 = np.loadtxt('u4_alpha3_2to3.out')


# u1_loaded_alpha1_2to3 = np.loadtxt('u1_alpha1_2to3.out')
# u1_loaded_alpha3_2to3 = np.loadtxt('u1_alpha3_2to3.out')
# u1_loaded_alpha10_2to3 = np.loadtxt('u1_alpha10_2to3.out')

# r_loaded_alpha3_2to3 = np.loadtxt('roll3_2to3.out')
# p_loaded_alpha3_2to3 = np.loadtxt('pitch3_2to3.out')
# y_loaded_alpha3_2to3 = np.loadtxt('yaw3_2to3.out')

# x_loaded_alpha1_2to3 = np.loadtxt('xalpha1_2to3.out')
# y_loaded_alpha1_2to3 = np.loadtxt('yalpha1_2to3.out')
# x_loaded_alpha10_2to3 = np.loadtxt('xalpha10_2to3.out')
# y_loaded_alpha10_2to3 = np.loadtxt('yalpha10_2to3.out')
# x_loaded_alpha5_2to3 = np.loadtxt('xalpha5_2to3.out')
# y_loaded_alpha5_2to3 = np.loadtxt('yalpha5_2to3.out')
# x_loaded_alpha3_2to3_modified = np.loadtxt('xalpha3_2to3_modified.out')
# y_loaded_alpha3_2to3_modified = np.loadtxt('yalpha3_2to3_modified.out')

# print(u)
filter_length = num_sample_cycle*num_cycle_filter
filter_num = np.ones(filter_length)/filter_length

Thrust_l_filtered=np.convolve(Thrust_l, filter_num, mode='same')
Thrust_r_filtered=np.convolve(Thrust_r, filter_num, mode='same')
tau_roll_filtered=np.convolve(tau_aero_sum[0,:], filter_num, mode='same')
tau_yaw_filtered=np.convolve(tau_aero_sum[1,:], filter_num, mode='same')
print("Thrust_l_filtered size :", np.size(Thrust_l_filtered)-np.size(state_log.sample_times()))
Thrust_filtered = np.convolve(Thrust_actual, filter_num, mode='same')
# Visualize state and input traces
# print("times",state_log.data()[1,:])RollPitchYaw
if show_flag==1:
    if show_flag_q==1:
        plt.clf()
        fig = plt.figure(1).set_size_inches(6, 6)
        for i in range(0,3):
            # print("i:%d" %i)
            plt.subplot(3, 1, i+1)
            # print("test:", num_state)
            plt.plot(state_log.sample_times()*time_gain, state_log.data()[i, :])
            plt.plot(state_log.sample_times()*time_gain, r_filtered_avg_out[i, :], color='red')
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

        fig = plt.figure(20).set_size_inches(6, 6)
        for i in range(0,4):
          # print("i:%d" %i)
            plt.subplot(4, 1, i+1)
            # print("test:", num_state)
            plt.plot(state_log.sample_times(), state_log.data()[i+3, :])
            plt.plot(state_log.sample_times()*time_gain, q_filtered_avg_out[i, :], color='red')
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
            com_vel_body = tau_sec_gain*state_log.data()[i+11, :];
            plt.plot(state_log.sample_times()*time_gain, com_vel_body)
            plt.plot(state_log.sample_times()*time_gain, v_filtered_avg_out[i, :], color='red')
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
            angular_vel_body = tau_sec_gain*state_log.data()[i+14, :];
            
            plt.plot(state_log.sample_times()*time_gain, angular_vel_body)
            plt.plot(state_log.sample_times()*time_gain, w_filtered_avg_out[i, :], color='red')
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
        for i in range(0,6):
            # print("i:%d" %i)
            plt.subplot(4, 2, i+1)
            # print("test:", num_state)
            if i==0:
                thrust_mg_gain=1; #  1cm = 0.01m
                control = Thrust_actual #ubar[i,:]*thrust_mg_gain;
                compare = Thrust_r_avg+Thrust_l_avg;
                compare2= Thrust
                compare3 = Thrust_filtered
                # plt.plot(state_log.sample_times()*time_gain, compare2, color='green')

                plt.plot(state_log.sample_times()*time_gain, compare3, color='red')
            elif i==1:
                mg_gain=1e0;
                control = mg_gain*ubar[i+2,:];
                compare = tau_aero_l_avg[0,:]+tau_aero_r_avg[0,:];
                compare2 = tau_roll_filtered;
                plt.plot(state_log.sample_times()*time_gain, compare2, color='red')
            elif i==2:
                mg_gain=1e0;
                control = mg_gain*ubar[i+2,:];
                compare = tau_aero_l_avg[1,:]+tau_aero_r_avg[1,:];
                compare2 = tau_yaw_filtered;
                plt.plot(state_log.sample_times()*time_gain, compare2, color='red')
            elif i==3:
                mg_gain=1e0;
                control = mg_gain*ubar[i+2,:];
                compare = tau_aero_l_avg[2,:]+tau_aero_r_avg[2,:];
            else:
                mg_gain=1e0; # 1000mg =1g
                control = mg_gain*ubar[i+2,:];
                compare = mg_gain*u_all[i+2,:];

            plt.plot(state_log.sample_times()*time_gain, control)
            if i<4:
                plt.plot(state_log.sample_times()*time_gain, compare, color='green')
            plt.grid(True)
            # plt.ylabel("x[%d]" % j)
            if i==0:
                
                plt.ylabel("T (m/s^2)")
            elif i==1:
                plt.ylabel(r'$\tau_1$ (mNmm)')
            elif i==2:
                plt.ylabel(r'$\tau_2$ (mNmm)')
            elif i==3:
                plt.ylabel(r"$\tau_3$ (mNmm)")
            elif i==4:
                plt.ylabel(r"$V_l$ (V)")
            elif i==5:
                plt.ylabel(r"$V_r$ (V)")
        plt.xlabel("Time (s)")

        fig = plt.figure(6).set_size_inches(6, 6)
        for i in range(0,2):
            # print("i:%d" %i)
            plt.subplot(2, 1, i+1)
            # print("test:", num_state)
            if i==0:
                thrust_mg_gain=1./m; #  1cm = 0.01m
                control = phi_l;
            else:
                mg_gain=1./m; # 1000mg =1g
                control = phi_r;
            # compare = fdw_all_output[i,:]/m;
            # plt.plot(state_log.sample_times()*time_gain, compare, color='red')
            
            plt.plot(state_log.sample_times()*time_gain, control)
            plt.grid(True)
            # plt.ylabel("x[%d]" % j)
            if i==0:
                plt.ylabel("Stroke ang left (rad)")
            elif i==1:
                plt.ylabel("Stroke ang right (rad)")
            elif i==2:
                plt.ylabel("Drag acc z")
        plt.xlabel("Time (s)")

        fig = plt.figure(70).set_size_inches(6, 6)
        for i in range(0,2):
            # print("i:%d" %i)
            plt.subplot(2, 1, i+1)
            # print("test:", num_state)
            if i==0:
                thrust_mg_gain=1./m; #  1cm = 0.01m
                control = Thrust_l;
                compare = Thrust_l_filtered;
            else:
                mg_gain=1./m; # 1000mg =1g
                control = Thrust_r;
                compare = Thrust_r_filtered;
            plt.plot(state_log.sample_times()*time_gain, control)
            plt.plot(state_log.sample_times()*time_gain, compare, color='red')
            plt.grid(True)
            # plt.ylabel("x[%d]" % j)
            if i==0:
                plt.ylabel("Thrust left (rad)")
            elif i==1:
                plt.ylabel("Thrust right (rad)")
        plt.xlabel("Time (s)")

        fig = plt.figure(71).set_size_inches(6, 6)
        for i in range(0,2):
            # print("i:%d" %i)
            plt.subplot(2, 1, i+1)
            # print("test:", num_state)
            if i==0:
                thrust_mg_gain=1./m; #  1cm = 0.01m
                control = AoA_l;
                # compare = Thrust_l_filtered;
            else:
                mg_gain=1./m; # 1000mg =1g
                control = AoA_r;
                # compare = Thrust_r_filtered;
            plt.plot(state_log.sample_times()*time_gain, control)
            # plt.plot(state_log.sample_times()*time_gain, compare, color='red')
            plt.grid(True)
            # plt.ylabel("x[%d]" % j)
            if i==0:
                plt.ylabel("AoA left (rad)")
            elif i==1:
                plt.ylabel("AoA right (rad)")
        plt.xlabel("Time (s)")
        
        # fig = plt.figure(12).set_size_inches(6, 6)
        # for i in range(0,3):
        #     # print("i:%d" %i)
        #     plt.subplot(3, 1, i+1)
        #     # print("test:", num_state)
        #     if i==0:
        #         thrust_mg_gain=1.; #  1cm = 0.01m
        #         control = wd_all[i,:]*thrust_mg_gain;
        #     else:
        #         mg_gain=1.; # 1000mg =1g
        #         control = mg_gain*wd_all[i,:];
        #     compare = d_w_all[i,:]
        #     plt.plot(state_log.sample_times()*time_gain, compare, color='red')    
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
        fig = plt.figure(13).set_size_inches(6, 6)
        for i in range(0,3):
            # print("i:%d" %i)
            plt.subplot(4, 1, i+1)
            # print("test:", num_state)
            if i==0:
                thrust_mg_gain=1.; #  1cm = 0.01m
                control = eta1_all[i,:]*thrust_mg_gain;
            else:
                mg_gain=1e0; # 1000mg =1g
                control = mg_gain*eta1_all[i,:];

            plt.plot(state_log.sample_times()*time_gain, control)
            plt.grid(True)
            # plt.ylabel("x[%d]" % j)
            if i==0:
                plt.ylabel("eta1 x")
            elif i==1:
                plt.ylabel("eta1 y")
            elif i==2:
                plt.ylabel("eta1 z")
        plt.xlabel("Time (s)")
        fig = plt.figure(14).set_size_inches(6, 6)
        for i in range(0,3):
            # print("i:%d" %i)
            plt.subplot(3, 1, i+1)
            # print("test:", num_state)
            if i==0:
                thrust_mg_gain=1.; #  1cm = 0.01m
                control = eta2_all[i,:]*thrust_mg_gain;
                compare = d_eta1_all[i,:]*thrust_mg_gain;
            else:
                mg_gain=1e0; # 1000mg =1g
                control = mg_gain*eta2_all[i,:];
                compare = mg_gain*d_eta1_all[i,:];

            plt.plot(state_log.sample_times()*time_gain, control)
            plt.plot(state_log.sample_times()*time_gain, compare, color='red')
            plt.grid(True)
            # plt.ylabel("x[%d]" % j)
            if i==0:
                plt.ylabel("eta2 x")
            elif i==1:
                plt.ylabel("eta2 y")
            elif i==2:
                plt.ylabel("eta2 z")
        plt.xlabel("Time (s)")
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
        
        # fig = plt.figure(26).set_size_inches(6, 6)
        # for i in range(0,6):
        #     # print("i:%d" %i)
        #     plt.subplot(6, 1, i+1)
        #     # print("test:", num_state)
        #     if i==0:
        #         thrust_mg_gain=1.; #  1cm = 0.01m
        #         control = eta_norm_all*thrust_mg_gain;

        #     elif i==1:
        #         mg_gain=kq; # 1000mg =1g
        #         control = mg_gain*Vx_all;
        #     elif i==2:
        #         control = Kinetic_all;
        #         compare = Vb_w_all;
        #         plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        #     elif i==3:
        #         # control = eta_norm_all*thrust_mg_gain;
        #         control = V1_lyapu;
        #     elif i==4:
        #         # control = eta_norm_all*thrust_mg_gain;
        #         control = 1/1.*V2_lyapu;
        #         # compare = 1/1.*V3_lyapu;
        #         # plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        #     elif i==5:
        #         # control = eta_norm_all*thrust_mg_gain;
        #         control = d_V_all;
        #         compare = vd_Kinetic_upper_all;
        #         plt.plot(state_log.sample_times()*time_gain, compare, color='red')
        #     plt.plot(state_log.sample_times()*time_gain, control)
        #     plt.grid(True)
        #     # plt.ylabel("x[%d]" % j)
        #     if i==0:
        #         plt.ylabel("eta_norm")
        #     elif i==1:
        #         plt.ylabel("Vx")
        #     elif i==2:
        #         plt.ylabel("Kinetic energy")
        #     elif i==3:
        #         plt.ylabel("V1")
        #     elif i==4:
        #         plt.ylabel("V2")
        #     elif i==5:
        #         plt.ylabel("derivative of Kinetic energy")
        # plt.xlabel("Time (s)")

        # fig = plt.figure(27).set_size_inches(6, 6)
        # plt.plot(eta1_all[0,:]+x_f, eta1_all[1,:]+y_f, color='blue')
        # # plt.plot(x_loaded_alpha1_2to3[500:800], y_loaded_alpha1_2to3[500:800], '--', color='black', label=r'$\alpha= 1.1$')
        # # plt.plot(state_log.data()[0,500:800], state_log.data()[1,500:800], color='black', label=r'$\alpha= 3$')
        # # plt.plot(x_loaded_alpha10_2to3[300:], y_loaded_alpha10_2to3[300:],  color='green')
        # # plt.plot(x_loaded_alpha5_2to3[500:800], y_loaded_alpha5_2to3[500:800], '--', color='green', label=r'$\alpha= 10$')
        # # plt.plot(x_loaded_alpha3_2to3_modified[500:], y_loaded_alpha3_2to3_modified[500:], '-', color='blue', label='CoM')
        # plt.plot(x_f, y_f,color='red')
        
        # plt.axis('equal')
        # ax = plt.gca();
        # ax.set_xlim(-radius*2.0, radius*2.0)
        # ax.set_ylim(-radius*2.0, radius*2.0)
        # plt.grid(True)
        # plt.ylabel(r'$y$')
        # plt.xlabel(r'$x$')    
        # ax.legend()         
        
        # fig = plt.figure(28).set_size_inches(6, 6)
        # control = mu_1_norm;
        # ax = plt.gca();
        # ax.plot(state_log.sample_times()*time_gain, control)
        # ax.plot(state_log.sample_times()*time_gain, Ub_w_all, label='$M_1$',color='red')
        # ax.plot(state_log.sample_times()*time_gain, Ub_w_all_new, '--', label='$M_2$',color='red')
        # plt.grid(True)
        # ax.legend() 
        # plt.ylabel('w$_1^2$+w$_2^2$')
        # plt.xlabel("Time (s)")
        
        # fig = plt.figure(29).set_size_inches(6, 6)
        # ax = plt.gca(projection='3d')
        # ax.set_xlim(-radius*2.0, radius*2.0)
        # ax.set_ylim(-radius*2.0, radius*2.0)
        # ax.set_zlim(0, z_f[0]+0.05)      
        # ax.set_xlabel(r'$x$')
        # ax.set_ylabel(r'$y$')
        # ax.set_zlabel(r'$z$')
        # ax.plot(x_f+eta1_all[0,:], y_f+eta1_all[1,:], z_f+eta1_all[2,:], label='VCP')
        # for j in range(1,150,3):
        #     listx=[vehicle_bottom[0, j], x_f[j]+eta1_all[0,j]]
        #     listy=[vehicle_bottom[1, j], y_f[j]+eta1_all[1,j]]
        #     listz=[vehicle_bottom[2, j], z_f[j]+eta1_all[2,j]]

        #     vehicle_listx=[vehicle_bottom[0, j], vehicle_top[0, j]]
        #     vehicle_listy=[vehicle_bottom[1, j], vehicle_top[1, j]]
        #     vehicle_listz=[vehicle_bottom[2, j], vehicle_top[2, j]]
        #     wing_listx=[wing_l[0, j], wing_r[0,j]]
        #     wing_listy=[wing_l[1, j], wing_r[1,j]]
        #     wing_listz=[wing_l[2, j], wing_r[2,j]]
        #     wing_list2x=[wing_l[0, j], state_log.data()[0, j]]
        #     wing_list2y=[wing_l[1, j], state_log.data()[1, j]]
        #     wing_list2z=[wing_l[2, j], state_log.data()[2, j]]
        #     wing_list3x=[wing_r[0, j], state_log.data()[0, j]]
        #     wing_list3y=[wing_r[1, j], state_log.data()[1, j]]
        #     wing_list3z=[wing_r[2, j], state_log.data()[2, j]]

        #     ax.plot(listx, listy, listz,color='black')
        #     ax.plot(vehicle_listx, vehicle_listy, vehicle_listz,color='black', linewidth=7.0)
        #     ax.plot(wing_listx, wing_listy, wing_listz,color='black', linewidth=0.5)
        #     ax.plot(wing_list2x, wing_list2y, wing_list2z,color='black', linewidth=0.5)
        #     ax.plot(wing_list3x, wing_list3y, wing_list3z,color='black', linewidth=0.5)
        #     ax.scatter(state_log.data()[0, j], state_log.data()[1, j], state_log.data()[2, j],'o', color='red')
        #     # ax.scatter(x_f[j]+eta1_all[0,j], y_f[j]+eta1_all[1,j], z_f[j]+eta1_all[2,j],'o', color='green')
        # ax.plot(x_f, y_f, z_f, label='VCP Reference',color='red')
        # ax.legend()  
        

        # fig = plt.figure(30).set_size_inches(6, 6)
        
        # plt.plot(state_log.sample_times()*time_gain, u1_loaded_alpha1_2to3, color='black',)
        
        # plt.plot(state_log.sample_times()*time_gain, u1_loaded_alpha3_2to3, color='blue',)
        # plt.plot(state_log.sample_times()*time_gain, u1_loaded_alpha10_2to3, color='green')
        # plt.grid(True)
        # plt.ylabel(r"$T (m/s^2)$")
            
        # plt.xlabel("Time (s)")    
        # ax.legend()         
        


        # fig = plt.figure(31).set_size_inches(6, 6)
        # for i in range(0,3):
        #     # print("i:%d" %i)
        #     plt.subplot(3, 1, i+1)
        #     # print("test:", num_state)
        #     plt.plot(state_log.sample_times()*time_gain, rpy[i,:],color='blue',label=r'$\alpha= 10$')
        #     plt.grid(True)
        #     j=i+3
        #     # plt.ylabel("x[%d]" % j)
        #     if i==0:
        #         plt.plot(state_log.sample_times()*time_gain, r_loaded_alpha3_2to3, color='black',label=r'$\alpha= 3$')
        
        #         plt.ylabel("roll (rad)")
        #         ax.legend()
        #     elif i==1:
        #         plt.plot(state_log.sample_times()*time_gain, p_loaded_alpha3_2to3, color='black',)
        
        #         plt.ylabel("pitch (rad)")
        #     elif i==2:
        #         plt.plot(state_log.sample_times()*time_gain, y_loaded_alpha3_2to3, color='black')
        
        #         plt.ylabel("yaw (rad)")

        # plt.grid(True)
            
        # plt.xlabel("Time (s)")    
        
        # fig = plt.figure(32).set_size_inches(6, 6)
        # for i in range(0,4):
        #     # print("i:%d" %i)
        #     plt.subplot(4, 1, i+1)
        #     # print("test:", num_state)
        #     if i==0:
        #         thrust_mg_gain=1; #  1cm = 0.01m
        #         control = ubar[i,:]*thrust_mg_gain;
        #         compare = u1_loaded_alpha3_2to3;
        #     elif i==1:
        #         mg_gain=1e0; # 1000mg =1g
        #         control = mg_gain*ubar[i,:];
        #         compare = u2_loaded_alpha3_2to3;
        #     elif i==2:
        #         mg_gain=1e0; # 1000mg =1g
        #         control = mg_gain*ubar[i,:];
        #         compare = u3_loaded_alpha3_2to3;
        #     elif i==3:
        #         mg_gain=1e0; # 1000mg =1g
        #         control = mg_gain*ubar[i,:];
        #         compare = u4_loaded_alpha3_2to3;
        
        #     plt.plot(state_log.sample_times()*time_gain, control,color='blue')
        #     plt.plot(state_log.sample_times()*time_gain, compare, color='black')
        #     plt.grid(True)
        #     # plt.ylabel("x[%d]" % j)
        #     if i==0:
        #         plt.ylabel("T (m/s^2)")
        #     elif i==1:
        #         plt.ylabel(r'$\tau_1$ (mNmm)')
        #     elif i==2:
        #         plt.ylabel(r'$\tau_2$ (mNmm)')
        #     elif i==3:
        #         plt.ylabel(r"$\tau_3$ (mNmm)")
        # plt.xlabel("Time (s)")


        # fig = plt.figure(33).set_size_inches(6, 6)
        
        # plt.plot(state_log.sample_times()*time_gain, dist_CoM_ref, color='black',)
        # plt.grid(True)
        # plt.ylabel(r"$r-r_{ref} (m/s^2)$")
            
        # plt.xlabel("Time (s)")    
        # ax.legend()       

    plt.grid(True)
    plt.show()
