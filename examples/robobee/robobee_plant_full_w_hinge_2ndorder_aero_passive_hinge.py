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
                       Iyy = 2., Izz = 3., g = 10., rw_l = 1, bw=1., m_eq=1., b_eq=1., k_eq=1., H_mag=1., k_wing=1.4, Ixx_wing=1.54, m_wing=0.52 
    ,l_wing=0.52,r_cp_wing=0.5, damp_wing=0.,
                        input_max = 10.):
        VectorSystem.__init__(self,
            8,                           # One input (torque at reaction wheel).
            21)                           # Four outputs (theta, phi, dtheta, dphi)
        self._DeclareContinuousState(21)  # Four states (theta, phi, dtheta, dphi).

        self.m = float(m)
        self.Ixx = float(Ixx)
        self.Iyy = float(Iyy)
        self.Izz = float(Izz)
        self.g = float(g)
        self.rw_l = float(rw_l)
        self.bw = float(bw)*1.0
        self.m_eq = float(m_eq)
        self.b_eq = float(b_eq)
        self.k_eq = float(k_eq)
        self.H_mag = float(H_mag)
        self.k_wing=float(k_wing)
        self.Ixx_wing=float(Ixx_wing)
        self.m_wing=float(m_wing)
        self.l_wing=float(l_wing)
        self.r_cp_wing=float(r_cp_wing)
        self.damp_wing=float(damp_wing)
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
        fdw = -self.bw*(v+np.dot(Rq,wr_w))
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
        
        vd_aero =  np.dot((-self.g*np.eye(3)),e3) #+fdw/self.m # \dot{v} = -ge3 +R(q)e3 u[0] : u[0] Thrust is a unit of acceleration
        wd_aero = -np.dot(I_inv,wIw) #+ np.dot(I_inv, taudw)
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

        I=np.zeros((3,3))       # Intertia matrix
        Ixx= self.Ixx;
        Iyy= self.Iyy;
        Izz= self.Izz;
        rw_l=self.rw_l
        I[0,0]=Ixx;
        I[1,1]=Iyy;
        I[2,2]=Izz;
        # Use the manipulator equation to get qdd.
        qq = x[0:7]
        phi_l = x[7]
        phi_r = x[8]
        psi_l = x[9]
        psi_r = x[10]
        
        qqd = x[11:17]
        dphi_l = x[17]
        dphi_r = x[18]
        
        dpsi_l = x[19]
        dpsi_r = x[20]

        wx= qqd[3]   # Body velocity in "body frame"
        wy= qqd[4]
        wz= qqd[5]
        w =np.array([wx,wy,wz])
        
        (Rq, Eq, wIw, I_inv, fdw, taudw, vd_aero, wd_aero) = self.GetManipulatorDynamics(qq, qqd)
        
        e1=np.array([1,0,0]) # e3 elementary vector
        e2=np.array([0,1,0]) # e3 elementary vector
        e3=np.array([0,0,1]) # e3 elementary vector

        # # print("e3,", e3.shape)
        
        # Awkward slice required on tauG to get shapes to agree --        # numpy likes to collapse the other dot products in this expression
        # to vectors.
        # epsilonn = 0.01 # Error pe
        r = qq[0:3]
        rd = qqd[0:3] # np.vstack([qd[0],qd[1],qd[2]]); 
        qd = np.dot(Eq.T,qqd[3:6])/2.   # \dot{quat}=1/2*E(q)^Tw
        

        # w_hat = np.zeros((3,3))
        # w_hat[0,:] = np.array([     0,   -w[2],     w[1] ])
        # w_hat[1,:] = np.array([  w[2],       0,    -w[0] ])
        # w_hat[2,:] = np.array([ -w[1],    w[0],        0 ])

        # Rqe3 = np.dot(Rq,e3)
        # Rqe3_hat = np.zeros((3,3))
        # Rqe3_hat[0,:] = np.array([        0,   -Rqe3[2],     Rqe3[1] ])
        # Rqe3_hat[1,:] = np.array([  Rqe3[2],          0,    -Rqe3[0] ])
        # Rqe3_hat[2,:] = np.array([ -Rqe3[1],    Rqe3[0],           0 ])
        
        # Rq_l = np.zeros((3,3))
        # Rq_l[0,:] = np.array([ math.cos(phi_l),   -math.sin(phi_l),     0 ])
        # Rq_l[1,:] = np.array([ math.sin(phi_l),    math.cos(phi_l),     0 ])
        # Rq_l[2,:] = np.array([               0,                  0,     1 ])
        # Rq_le2 = np.dot(Rq_l,e2)
        # RqRq_le2 = np.dot(Rq,Rq_le2)
        # w_hatRq_le2 = np.dot(w_hat, Rq_le2)
        # Rqw_hatRq_le2 = np.dot(Rq, w_hatRq_le2)

        # print("x:", x)
        # print("w_hatRq_le2:", w_hatRq_le2)

        # Rq_r = np.zeros((3,3))
        # Rq_r[0,:] = np.array([ math.cos(phi_r),   -math.sin(phi_r),     0 ])
        # Rq_r[1,:] = np.array([ math.sin(phi_r),    math.cos(phi_r),     0 ])
        # Rq_r[2,:] = np.array([               0,                  0,     1 ])
        # Rq_re2 = np.dot(Rq_r,e2)
        # RqRq_re2 = np.dot(Rq,Rq_re2)
        # w_hatRq_re2 = np.dot(w_hat, Rq_re2)
        # Rqw_hatRq_re2 = np.dot(Rq, w_hatRq_re2)

        # dphi_l_hat = np.zeros((3,3))
        # dphi_l_hat[0,:] = np.array([      0,   -dphi_l,     0 ])
        # dphi_l_hat[1,:] = np.array([ dphi_l,         0,     0 ])
        # dphi_l_hat[2,:] = np.array([      0,         0,     0 ])

        # dphi_l_hate2 = np.dot(dphi_l_hat, e2)
        # Rq_ldphi_l_hate2 = np.dot(Rq_l, dphi_l_hate2)
        # RqRq_ldphi_l_hate2 = np.dot(Rq, Rq_ldphi_l_hate2)

        # dphi_r_hat = np.zeros((3,3))
        # dphi_r_hat[0,:] = np.array([      0,   -dphi_r,     0 ])
        # dphi_r_hat[1,:] = np.array([ dphi_r,         0,     0 ])
        # dphi_r_hat[2,:] = np.array([      0,         0,     0 ])
        
        # dphi_r_hate2 = np.dot(dphi_r_hat, e2)
        # Rq_rdphi_r_hate2 = np.dot(Rq_r, dphi_r_hate2)
        # RqRq_rdphi_r_hate2 = np.dot(Rq, Rq_rdphi_r_hate2)

        # print("dphi_r_hat:", dphi_r_hat)

        # Rq_psi_l = np.zeros((3,3))
        # Rq_psi_l[0,:] = np.array([  math.cos(psi_l),   0, math.sin(psi_l)])
        # Rq_psi_l[1,:] = np.array([                0,   0,  0             ])
        # Rq_psi_l[2,:] = np.array([ -math.sin(psi_l),   0, math.cos(psi_l)])

        # Rq_psi_r = np.zeros((3,3))
        # Rq_psi_r[0,:] = np.array([  math.cos(psi_r),   0, math.sin(psi_r)])
        # Rq_psi_r[1,:] = np.array([                0,   0,  0             ])
        # Rq_psi_r[2,:] = np.array([ -math.sin(psi_r),   0, math.cos(psi_r)])

        # Iw = np.dot(I,w)
        # r_w = rw_l*e3; # Length to the wing from CoM
        # wr_w = np.cross(w,r_w) # w x r
        # wr_w_hat = np.dot(w_hat,r_w)
        # # print("wr_w-wr_w_hat", wr_w-wr_w_hat)

        # wwr_w = np.cross(w,wr_w) # w x (w x r)

        # # Wing angle

        # eta = np.zeros(3)
        # eta_l = np.zeros(3)
        # eta_r = np.zeros(3)    

        # eta = r+Rqe3*rw_l;
        # eta_l = eta+np.dot(Rq,Rq_le2)*r_cp  # phi_l positive : clockwise
        # eta_r = eta-np.dot(Rq,Rq_re2)*r_cp # phi_r positive : clockwise

        # # Wing velocity

        # deta = np.zeros(3)
        # deta_l = np.zeros(3)
        # deta_r = np.zeros(3)    

        # deta = v + np.dot(Rq,wr_w)
        # deta_l = deta + Rqw_hatRq_le2*r_cp + RqRq_ldphi_l_hate2*r_cp 
        # deta_r = deta - Rqw_hatRq_re2*r_cp - RqRq_rdphi_r_hate2*r_cp 

        
        # # Projection to wing stroke plane
        # P_proj_wing_l = np.zeros((3,3))
        # P_proj_wing_l = np.eye(3)-np.outer(RqRq_le2, RqRq_le2)
        # # print("outer_before:", RqRq_le2)
        # # print("outer:", P_proj_wing_l)
        # P_proj_wing_r = np.zeros((3,3))
        # P_proj_wing_r = np.eye(3)-np.outer(RqRq_re2, RqRq_re2)

        # ## Decomposing the wing velocity in the projected domain
        
        # proj_deta_l_body                = np.dot(P_proj_wing_l,deta)
        # proj_deta_l_body_wing_couple    = np.dot(P_proj_wing_l,Rqw_hatRq_le2)
        # proj_deta_l_wing                = np.dot(P_proj_wing_l,RqRq_ldphi_l_hate2)

        # proj_deta_r_body                =  np.dot(P_proj_wing_r,deta)
        # proj_deta_r_body_wing_couple    = -np.dot(P_proj_wing_r,Rqw_hatRq_re2)
        # proj_deta_r_wing                = -np.dot(P_proj_wing_r,RqRq_rdphi_r_hate2)

        # # nominal velocity in wing stroke plane for CoP
        # print("deta :", deta)
        # print("deta_l :", deta_l)
        # print("deta_r :", deta_r)
        # v_ortho_l = np.dot(P_proj_wing_l, deta_l)
        # v_ortho_r = np.dot(P_proj_wing_r, deta_r) 
        
        # v_ortho_l_norm_sq = np.dot(v_ortho_l,v_ortho_l)
        # v_ortho_r_norm_sq = np.dot(v_ortho_r,v_ortho_r)

        # v_ortho_l_norm = math.sqrt(v_ortho_l_norm_sq)
        # v_ortho_r_norm = math.sqrt(v_ortho_r_norm_sq)

        # if v_ortho_l_norm==0 or v_ortho_r_norm==0 :
        #     v_ortho_l_normalized = np.zeros(3)
        #     v_ortho_r_normalized = np.zeros(3)
        # else:
        #     v_ortho_l_normalized = v_ortho_l/v_ortho_l_norm
        #     v_ortho_r_normalized = v_ortho_r/v_ortho_r_norm 

        # print("v_ortho_l_normalized :", v_ortho_l_normalized)
        # print("v_ortho_r_normalized :", v_ortho_r_normalized)   
        # print("v_ortho_l_norm_sq :", v_ortho_l_norm_sq)
        # print("v_ortho_r_norm_sq :", v_ortho_r_norm_sq)   

        # # Hinge angle in the global frame

        # psi_l_wing = np.dot(Rq_psi_l,-e3) 
        # psi_l_body = np.dot(Rq_l,psi_l_wing) 
        # psi_l_world = np.dot(Rq,psi_l_body) 

        # psi_r_wing = np.dot(Rq_psi_r,-e3) 
        # psi_r_body = np.dot(Rq_r,psi_r_wing) 
        # psi_r_world = np.dot(Rq,psi_r_body) 


        # AoA_inner_product_l = np.dot(v_ortho_l_normalized, -psi_l_world) # Based on Kevin Science Robobtics supplement paper
        # AoA_inner_product_r = np.dot(v_ortho_r_normalized, -psi_r_world)
        
        # AoA_l = math.acos(AoA_inner_product_l)
        # AoA_r = math.acos(AoA_inner_product_r)
        # print("AoA_l:", AoA_l)
        # print("AoA_r:", AoA_r)
        # # nominal velocity in wing stroke plane for each blade

        # FL_l_r=np.zeros(num_blade)
        # FL_r_r=np.zeros(num_blade)
        # FD_l_r = np.zeros((3,num_blade))
        # FD_r_r = np.zeros((3,num_blade))
        # FL_l_vec=np.zeros((3,num_blade))
        # FL_r_vec=np.zeros((3,num_blade))
        # FD_l_vec=np.zeros((3,num_blade))
        # FD_r_vec=np.zeros((3,num_blade))
        # F_l_vec=np.zeros((3,num_blade))
        # F_r_vec=np.zeros((3,num_blade))
        # tau_aero_l =np.zeros((3,num_blade))
        # tau_aero_r =np.zeros((3,num_blade))

        # chord_r_CoP_z_l=np.zeros(num_blade)
        # chord_r_CoP_z_r=np.zeros(num_blade)
        
        # chord_r_CoP_world_l=np.zeros((3,num_blade))
        # chord_r_CoP_world_r=np.zeros((3,num_blade))



        # for i_iter in range(0,num_blade):
        #     chord_r_from_root = wing_xr + chordlength_r[i_iter]

        #     chord_r_from_root_world_l=Rqe3*rw_l+np.dot(Rq,Rq_le2)*chord_r_from_root
        #     chord_r_from_root_world_r=Rqe3*rw_l-np.dot(Rq,Rq_re2)*chord_r_from_root

        #     v_ortho_l_r =  proj_deta_l_body + (proj_deta_l_body_wing_couple + proj_deta_l_wing)*chord_r_from_root
        #     v_ortho_r_r =  proj_deta_r_body + (proj_deta_r_body_wing_couple + proj_deta_r_wing)*chord_r_from_root

        #     v_ortho_l_r_norm_sq = np.dot(v_ortho_l_r, v_ortho_l_r)
        #     v_ortho_r_r_norm_sq = np.dot(v_ortho_r_r, v_ortho_r_r)

        #     # print("v_ortho_r_r_norm_sq", v_ortho_r_r_norm_sq )
        #     # print("v_ortho_l_r_norm_sq", v_ortho_l_r_norm_sq )

        #     v_ortho_l_r_norm = math.sqrt(v_ortho_l_r_norm_sq)
        #     v_ortho_r_r_norm = math.sqrt(v_ortho_r_r_norm_sq)

        #     if v_ortho_l_r_norm==0 or v_ortho_r_r_norm==0 :
        #         v_ortho_l_r_normalized = np.zeros(3)
        #         v_ortho_r_r_normalized = np.zeros(3)
        #     else:
        #         v_ortho_l_r_normalized = v_ortho_l_r/v_ortho_l_r_norm
        #         v_ortho_r_r_normalized = v_ortho_r_r/v_ortho_r_r_norm 

        #     AoA_inner_product_l_r = np.dot( v_ortho_l_r_normalized, -psi_l_world) # pointing downward : c vector
        #     AoA_inner_product_r_r = np.dot( v_ortho_r_r_normalized, -psi_r_world) # pointing downward : c vector

        #     AoA_l_r = math.acos(AoA_inner_product_l_r)
        #     AoA_r_r = math.acos(AoA_inner_product_r_r)

        #     # Center of Pressure for each blade : Kevin IROS 2016
        #     chord_r_CoP_z_l[i_iter] = chordlength_yLE[i_iter]-(0.25 + 0.25/(1+math.exp(5.*(1.-4.*AoA_l_r/math.pi))))*chordlength_cr[i_iter]
        #     chord_r_CoP_z_r[i_iter] = chordlength_yLE[i_iter]-(0.25 + 0.25/(1+math.exp(5.*(1.-4.*AoA_r_r/math.pi))))*chordlength_cr[i_iter]

        #     chord_r_CoP_world_l[0:3,i_iter] = chord_r_from_root_world_l+psi_l_world*chord_r_CoP_z_l[i_iter]
        #     chord_r_CoP_world_r[0:3,i_iter] = chord_r_from_root_world_r+psi_r_world*chord_r_CoP_z_r[i_iter]

        #     # print("Psi ", psi_l)
        #     # print("AoA_l_r", AoA_l_r)
        #     CL_l = CL0*math.sin(2.*AoA_l_r)
        #     CL_r = CL0*math.sin(2.*AoA_r_r)

        #     CD_l = (CD0+CDmax)/2.- (CDmax-CD0)/2.*math.cos(2.*AoA_l_r)
        #     CD_r = (CD0+CDmax)/2.- (CDmax-CD0)/2.*math.cos(2.*AoA_r_r)

        #     # print("CL_l:", CL_l)

        #     # print("CD_l:", CD_l)

        #     FL_l_r[i_iter] = 1./2.*rho*CL_l*v_ortho_l_r_norm_sq*chordlength_cr[i_iter]*blade_width
        #     FL_r_r[i_iter] = 1./2.*rho*CL_r*v_ortho_r_r_norm_sq*chordlength_cr[i_iter]*blade_width

        #     # print("FL_l_r", FL_l_r[i_iter])

        #     # print("FL_r_r", FL_r_r[i_iter])

        #     FD_l_r[0:3,i_iter] = -1./2.*rho*CD_l*v_ortho_l_r_norm_sq*v_ortho_l_r_normalized*chordlength_cr[i_iter]*blade_width
        #     FD_r_r[0:3,i_iter] = -1./2.*rho*CD_r*v_ortho_r_r_norm_sq*v_ortho_r_r_normalized*chordlength_cr[i_iter]*blade_width
            
        #     FL_l_vec[0:3,i_iter] = FL_l_r[i_iter]*np.dot(Rq,e3)/Force_scale
        #     FL_r_vec[0:3,i_iter] = FL_r_r[i_iter]*np.dot(Rq,e3)/Force_scale
        #     FD_l_vec[0:3,i_iter] = FD_l_r[0:3,i_iter]/Force_scale
        #     FD_r_vec[0:3,i_iter] = FD_r_r[0:3,i_iter]/Force_scale
            
        #     # print("FL_l_vec in body:", np.dot(Rq.T,FL_l_vec[0:3,i_iter]))
        #     # print("FD_l_vec in body:", np.dot(Rq.T,FD_l_vec[0:3,i_iter]))


        #     F_l_vec[0:3,i_iter] = FL_l_vec[0:3,i_iter] + FD_l_vec[0:3,i_iter]
        #     F_r_vec[0:3,i_iter] = FL_r_vec[0:3,i_iter] + FD_r_vec[0:3,i_iter]

        #     tau_aero_l[0:3,i_iter] = np.cross(chord_r_CoP_world_l[0:3,i_iter],F_l_vec[0:3,i_iter])
        #     tau_aero_l[0:3,i_iter] = np.dot(Rq.T,tau_aero_l[0:3,i_iter])

        #     tau_aero_r[0:3,i_iter] = np.cross(chord_r_CoP_world_r[0:3,i_iter],F_r_vec[0:3,i_iter])
        #     tau_aero_r[0:3,i_iter] = np.dot(Rq.T,tau_aero_r[0:3,i_iter])

        # FL_l = np.sum(FL_l_r)
        # FL_r = np.sum(FL_r_r)
        # FD_l = np.sum(FD_l_r)
        # FD_r = np.sum(FD_r_r)


        # print("FL_l:", FL_l)
        # print("FD_l:", FD_l)
        

        # F_aero_sum = (np.sum(F_l_vec,1)+np.sum(F_r_vec,1))/m

        # print("F_aero_sum", F_aero_sum)

        # u[0:3]=F_aero_sum

        # tau_aero_l_sum = np.sum(tau_aero_l, axis=1)
        # tau_aero_r_sum = np.sum(tau_aero_r, axis=1)
        # tau_aero = tau_aero_l_sum+tau_aero_r_sum
        # print("tau_aero_l: ", tau_aero_l_sum)
        # print("tau_aero_r: ", tau_aero_r_sum)
        # print("tau_aero: ", tau_aero)

        # u[3:6] = tau_aero 

        #  # u[4] = dphi_l
        # # u[5] = dphi_r
        
        # # u[6] = dpsi_l
        # # u[7] = dpsi_r

        # # u[0] = Thrust #1.5*g;
        # Thrust = np.dot(np.dot(Rq,e3),F_aero_sum)
        # print("Thrust", Thrust)

        # chordlength_r = np.loadtxt('chordlength_list_r.txt')
        # print("chordlength r:", chordlength_r[1])

        vd = vd_aero + u[0:3] #+np.dot(Rq,e3)*u[0]
        wd = wd_aero+np.dot(I_inv, u[3:6])

        dphid_l = dphi_l;
        dphid_r = dphi_r;
        ddphid_l = -self.b_eq/self.m_eq*dphi_l - self.k_eq/self.m_eq*phi_l + self.H_mag/self.m_eq*u[6];
        ddphid_r = -self.b_eq/self.m_eq*dphi_r - self.k_eq/self.m_eq*phi_r + self.H_mag/self.m_eq*u[7];




        dpsid_l = dpsi_l;
        dpsid_r = dpsi_r;
        ddpsid_l = -self.k_wing/self.Ixx_wing*psi_l-self.m_wing*self.l_wing*math.cos(psi_l)*ddphid_l*self.r_cp_wing/self.Ixx_wing-self.damp_wing*dpsi_l/self.Ixx_wing;
        ddpsid_r = -self.k_wing/self.Ixx_wing*psi_r+self.m_wing*self.l_wing*math.cos(psi_r)*ddphid_r*self.r_cp_wing/self.Ixx_wing-self.damp_wing*dpsi_r/self.Ixx_wing;
        # ddpsid_l = -30*(psi_l-math.pi/4)-10*dpsi_l;
        # ddpsid_r = -30*(psi_r-math.pi/4)-10*dpsi_r;
        

        # print("ddpsid_l:", ddpsid_l)
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

        return np.hstack([1.*rd, 1.*qd, dphid_l, dphid_r, dpsid_l, dpsid_r, 1.*vd, 1.*wd, ddphid_l, ddphid_r, ddpsid_l, ddpsid_r])


    # This method calculates the time derivative of the state,
    # which allows the system to be simulated forward in time.
    def _DoCalcVectorTimeDerivatives(self, context, u, x, xdot):
        q = x[0:7]
        qd = x[11:17]
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
            19,                           # Four inputs: full state inertial wheel pendulum..
            8)                           # One output (torque for reaction wheel).
        self.feedback_rule = feedback_rule

    # This method calculates the output of the system from the
    # input by applying the supplied feedback rule.
    def _DoCalcVectorOutput(self, context, u, x, y):
        # Remember that the input "u" of the controller is the
        # state of the plant
        time_t = context.get_time()
        # print("time_t:", time_t)

        y[:] = self.feedback_rule(u,time_t)
        # Hybrid switching controller
        # if time_t<2:
        #     y[:] = self.feedback_rule(u,time_t)
        # elif time_t>2 and time_t<2.2:
        #     y[:] = np.zeros(4);
        # else:
        #     y[:] = self.feedback_rule(u,time_t)

class RigidBodySelection(VectorSystem):
    ''' System to control the robobee. Must be handed
    a function with signature:
        u = f(t, x)
    that computes control inputs for the pendulum. '''

    def __init__(self):
        VectorSystem.__init__(self,
            21,                           # Four inputs: full state inertial wheel pendulum..
            13)                           # One output (torque for reaction wheel).
        
    # This method calculates the output of the system from the
    # input by applying the supplied feedback rule.
    def _DoCalcVectorOutput(self, context, u, x, y):
        # Remember that the input "u" of the controller is the
        # state of the plant
        y[:] = np.hstack([u[0:7],u[11:17]])

class RigidBodySelection_controller(VectorSystem):
    ''' System to control the robobee. Must be handed
    a function with signature:
        u = f(t, x)
    that computes control inputs for the pendulum. '''

    def __init__(self):
        VectorSystem.__init__(self,
            21,                           # Four inputs: full state inertial wheel pendulum..
            19)                           # One output (torque for reaction wheel).
        
    # This method calculates the output of the system from the
    # input by applying the supplied feedback rule.
    def _DoCalcVectorOutput(self, context, u, x, y):
        # Remember that the input "u" of the controller is the
        # state of the plant
        y[:] = np.hstack([u[0:7],u[11:17],u[7:11], u[17:19]]) # Reorder the state [q, qd, phi, psi]

def RunSimulation(robobee_plantBS_torque, control_law, x0=np.random.random((21, 1)), duration=30):
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
        m_eq = robobee_plantBS_torque.m_eq, 
        b_eq = robobee_plantBS_torque.b_eq, 
        k_eq = robobee_plantBS_torque.k_eq, 
        H_mag = robobee_plantBS_torque.H_mag,
        k_wing=robobee_plantBS_torque.k_wing, 
        Ixx_wing=robobee_plantBS_torque.Ixx_wing, 
        m_wing=robobee_plantBS_torque.m_wing,
        l_wing=robobee_plantBS_torque.l_wing,
        r_cp_wing=robobee_plantBS_torque.r_cp_wing,
        damp_wing=robobee_plantBS_torque.damp_wing,
        input_max = robobee_plantBS_torque.input_max))

    Rigidbody_selector = builder.AddSystem(RigidBodySelection())
    Rigidbody_selector_controller = builder.AddSystem(RigidBodySelection_controller())

    print("1. Connecting plant and controller\n")
    controller = builder.AddSystem(robobee_controller)

    # builder.Connect(plant.get_output_port(0),Rigidbody_selector.get_input_port(0))
    builder.Connect(plant.get_output_port(0),Rigidbody_selector_controller.get_input_port(0))       
    print("Input plant:", plant.get_input_port(0).size())
    print("Output controller:", controller.get_output_port(0).size())
    
    # builder.Connect(Rigidbody_selector.get_output_port(0), controller.get_input_port(0))
    builder.Connect(Rigidbody_selector_controller.get_output_port(0), controller.get_input_port(0))
    
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))

    # Create a logger to capture the simulation of our plant
    set_time_interval = 1./170.*1/15 #*0.0005
    time_interval_multiple = 1;
    publish_period = set_time_interval*time_interval_multiple

    print("2. Connecting plant to the logger\n")
    
    input_log = builder.AddSystem(SignalLogger(8))
    # input_log._DeclarePeriodicPublish(publish_period, 0.0)
    builder.Connect(controller.get_output_port(0), input_log.get_input_port(0))

    state_log = builder.AddSystem(SignalLogger(21))
    # state_log._DeclarePeriodicPublish(publish_period, 0.0)
    
    builder.Connect(plant.get_output_port(0), state_log.get_input_port(0))
    
    # Drake visualization
    print("3. Connecting plant output to DrakeVisualizer\n")
    
    rtree = RigidBodyTree(FindResourceOrThrow("drake/examples/robobee/robobee_full_w_hinge_arena.urdf"), FloatingBaseType.kQuaternion )
    print("Rigid body size : ", rtree.get_num_positions())
    lcm = DrakeLcm()
    visualizer = builder.AddSystem(DrakeVisualizer(tree=rtree,
       lcm=lcm, enable_playback=True))
    print("Input visualizer:", visualizer.get_input_port(0).size())
    print("Output plant:", plant.get_output_port(0).size())
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
    
    for ii in range(5):
        simulator.set_target_realtime_rate(0.1)
        visualizer.ReplayCachedSimulation()

    return input_log, state_log

