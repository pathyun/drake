/*
  RobobeeCLFQPController.h

  Objective : Compute the CLF-QP controller with quaternion
  Algorithm : y1 = x
              y2 = y
              y3 = z
              y4 = yaw(q) // as a function of quaternion
              z = q^Tq 
              Detailed note to be written soon (Proof of Diffeomorphism)

  Remark : 1) 
  Author : Nak-seung Patrick Hyun
  Date : 08/07/2018
*/


#pragma once
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <memory>
#include <Eigen/Core>
#include <Eigen/Eigenvalues> 


#include "drake/examples/robobee/robobee_plant.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/math/rotation_matrix.h"

// #include "drake/common/symbolic_expression.h"

namespace drake {
namespace examples {
namespace robobee {

namespace {

Eigen::MatrixXd default_Mout = Eigen::MatrixXd::Identity(14,14);
Eigen::MatrixXd default_Aout = Eigen::MatrixXd::Identity(14,14);
Eigen::MatrixXd default_Bout = Eigen::MatrixXd::Zero(14,4);

}  // namespace

using std::cout;
using std::endl;

template <typename T>
class RobobeeCLFQPController : public systems::LeafSystem<T> {
 public:
  RobobeeCLFQPController(double m_arg, const Eigen::Matrix3d& I_arg)
      : robobee_{}, robobee_context_(robobee_.CreateDefaultContext()),
      g_{9.81}, m_(m_arg), min_e_Q_(0), max_e_P_(0), I_(I_arg), Mout_(default_Mout), Aout_(default_Aout), Bout_(default_Bout)  {
    this->DeclareInputPort(systems::kVectorValued, kInputDimension);
    this->DeclareVectorOutputPort(systems::BasicVector<T>(kStateDimension),
                                &RobobeeCLFQPController::CalcControl);
    /// Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  // [0] LQR setup for Output feedback controller

    double gain_R = 1.; 
    double gain_Q = 1e2;
    double position_gain = 1e4;
    double dposition_gain = 1e3;
    double ddposition_gain = 1e2;
    
    Eigen::Matrix4d R = gain_R*Eigen::Matrix4d::Identity();
    Eigen::MatrixXd Q = gain_Q*Eigen::MatrixXd::Identity(14,14);
    // Eigen::MatrixXd Mout_ = Eigen::MatrixXd::Zero(14,14);
    
    
    Q.block(0,0,3,3)= position_gain*Eigen::Matrix3d::Identity(); 
    Q.block(3,3,3,3)= dposition_gain*Eigen::Matrix3d::Identity();
    Q.block(6,6,3,3)= ddposition_gain*Eigen::Matrix3d::Identity();
    Q.block(10,10,3,3)= ddposition_gain*Eigen::Matrix3d::Identity();
    
    // Eigen::MatrixXd Aout_ = Eigen::MatrixXd::Zero(14,14);
    // Eigen::MatrixXd Bout_ = Eigen::MatrixXd::Zero(14,4);
    
    Aout_.block(0,3,3,3)=Eigen::Matrix3d::Identity();
    Aout_.block(3,6,3,3)=Eigen::Matrix3d::Identity();
    Aout_.block(6,10,3,3)=Eigen::Matrix3d::Identity();
    Aout_(9,13)=1;

    Bout_.block(10,0,4,4)= Eigen::Matrix4d::Identity();

    // std::cout << "\n Q:" << Q;
    // std::cout << "\n R:" << R;
    
    Mout_ =  drake::math::ContinuousAlgebraicRiccatiEquation(  Aout_, Bout_, Q, R); // Solution to CARE

    Eigen::SelfAdjointEigenSolver<MatrixX<T>> es;
    es.compute(Q);
    VectorX<T> eval_Q = es.eigenvalues().transpose();
    es.compute(Mout_);
    VectorX<T> eval_P = es.eigenvalues().transpose();

    // cout << "\n eval_Q : " << eval_Q << "\n";
    min_e_Q_ = eval_Q.minCoeff();
    max_e_P_ = eval_P.maxCoeff(); 
    // cout << "\n min_e_Q_ : " << min_e_Q_ << "\n";
    // cout << "\n max_e_P_ : " << max_e_P_ << "\n";
   }

  
  void CalcControl(const systems::Context<T>& context,
                         systems::BasicVector<T>* output) const {

    
    const VectorX<T> x = this->EvalVectorInput(context, 0)->get_value();

    // const Vector4<T> u(1,0,0,0);

    // [0] Reference trajectory generation (example for circle and hovering at 0.3)
    T t = context.get_time();
    T x_f, y_f, z_f, dx_f, dy_f, ddx_f, ddy_f, dddx_f, dddy_f, ddddx_f, ddddy_f, radius, w_freq, T_period;

    T_period = 5.; // Period
    w_freq = 2*M_PI*1/T_period; // Rad freq
    radius = 0.5;
    x_f = radius*cos(w_freq*t);
    y_f = radius*sin(w_freq*t);
    dx_f = -radius*pow(w_freq,1)*sin(w_freq*t);
    dy_f = radius*pow(w_freq,1)*cos(w_freq*t);
    ddx_f = -radius*pow(w_freq,2)*cos(w_freq*t);
    ddy_f = -radius*pow(w_freq,2)*sin(w_freq*t);
    dddx_f = radius*pow(w_freq,3)*sin(w_freq*t);
    dddy_f = -radius*pow(w_freq,3)*cos(w_freq*t);
    ddddx_f = radius*pow(w_freq,4)*cos(w_freq*t);
    ddddy_f = radius*pow(w_freq,4)*sin(w_freq*t);

    z_f = 0.3; // Hovering height

    // std::cout << "time: " << t << "\n";

    // [1] Useful Matrix and vectors

    T q0, q1, q2, q3;

    Vector3<T> e1(1,0,0);
    Vector3<T> e2(0,1,0);
    Vector3<T> e3(0,0,1);

    VectorX<T> q = Eigen::VectorXd::Zero(7);
    VectorX<T> qd = Eigen::VectorXd::Zero(6);
    
    q = x.template segment<7>(0);
    qd = x.template segment<6>(7);
    T xi1 = x(14);
    T xi2 = x(13);
    Vector3<T> v=qd.template segment<3>(0);
    Vector3<T> w=qd.template segment<3>(3);

    // Convert to Eigen quaternion
    Vector4<T> quat_vector = q.template segment<4>(3);
    q0=quat_vector(0);
    q1=quat_vector(1);
    q2=quat_vector(2);
    q3=quat_vector(3);

    quat_vector(3)=q0;
    quat_vector(0)=q1;
    quat_vector(1)=q2;
    quat_vector(2)=q3;

    const Eigen::Quaternion<T> quat(quat_vector);

    const drake::math::RotationMatrix<T> Rq(quat);
    
    MatrixX<T> Eq = Eigen::MatrixXd::Zero(3,4);

    Eq.block(0,0,1,4) =  Eigen::Vector4d(-1*q1,  q0, -1*q3,  q2).transpose();
    Eq.block(1,0,1,4) << Eigen::Vector4d(-1*q2,  q3,  q0, -1*q1).transpose();
    Eq.block(2,0,1,4) << Eigen::Vector4d(-1*q3, -1*q2,  q1,  q0).transpose();

    const Vector3<T> wIw = w.cross(I_ * w);            // Expressed in B
    
    const MatrixX<T> I_inv = I_.inverse();
    
    MatrixX<T> F1q = Eigen::MatrixXd::Zero(3,4);

    F1q.block(0,0,1,4) = Eigen::Vector4d( q2,  q3,  q0, q1).transpose();
    F1q.block(1,0,1,4) = Eigen::Vector4d(-q1, -q0,  q3, q2).transpose();
    F1q.block(2,0,1,4) = Eigen::Vector4d( q0, -q1, -q2, q3).transpose();
    
    // cout <<"q:" << q << "\n";
    // cout <<"qd:"<< qd << "\n";
    // cout <<"v:"<< v << "\n";
    // cout <<"w"<< w << "\n";
    // cout <<"xi1:"<< xi1 << "\n";
    // cout <<"xi2:"<< xi2 << "\n";
    
    
    Vector3<T> Rqe3 = Eigen::Vector3d::Zero();
    Matrix3<T> Rqe3_hat = Eigen::Matrix3d::Zero();
    Rqe3 = Rq.matrix()*e3;
    Rqe3_hat.block(0,0,1,3) = Eigen::Vector3d(       0, -Rqe3(2),  Rqe3(1)).transpose();
    Rqe3_hat.block(1,0,1,3) = Eigen::Vector3d( Rqe3(2),        0, -Rqe3(0)).transpose();
    Rqe3_hat.block(2,0,1,3) = Eigen::Vector3d(-Rqe3(1),  Rqe3(0),        0).transpose();

    Vector3<T> Rqe1 = Eigen::Vector3d::Zero();
    Matrix3<T> Rqe1_hat = Eigen::Matrix3d::Zero();
    Rqe1 = Rq.matrix()*e1;
    Rqe1_hat.block(0,0,1,3) = Eigen::Vector3d(       0, -Rqe1(2),  Rqe1(1)).transpose();
    Rqe1_hat.block(1,0,1,3) = Eigen::Vector3d( Rqe1(2),        0, -Rqe1(0)).transpose();
    Rqe1_hat.block(2,0,1,3) = Eigen::Vector3d(-Rqe1(1),  Rqe1(0),        0).transpose();

    T Rqe1_x=e2.dot(Rqe1);
    T Rqe1_y=e1.dot(Rqe1);

    Matrix3<T> w_hat = Eigen::Matrix3d::Zero();
    w_hat.block(0,0,1,3) = Eigen::Vector3d(    0, -w(2),  w(1)).transpose();
    w_hat.block(1,0,1,3) = Eigen::Vector3d( w(2),     0, -w(0)).transpose();
    w_hat.block(2,0,1,3) = Eigen::Vector3d(-w(1),  w(0),     0).transpose();

    Vector3<T> Rw = Eigen::Vector3d::Zero();
    Matrix3<T> Rw_hat = Eigen::Matrix3d::Zero();
    Rw = Rq.matrix()*w;
    Rw_hat.block(0,0,1,3) = Eigen::Vector3d(     0,   -Rw(2),     Rw(1) );
    Rw_hat.block(1,0,1,3) = Eigen::Vector3d(  Rw(2),       0,    -Rw(0) );
    Rw_hat.block(2,0,1,3) = Eigen::Vector3d( -Rw(1),    Rw(0),        0 );
    // #- Checking the derivation
    // cout << "\n Rq : \n"<< Rq.matrix() << "\n";
    // cout << "\n F1q : \n"<< F1q << "\n";
    // cout << "\n Eq : \n"<< Eq << "\n";
    // cout << "\n F1qEqT-(-Rqe3_hat) : \n"<< F1q*Eq.transpose()-(-Rqe3_hat) << "\n";
    // Vector3<T> Rqe3_cal = Eigen::Vector3d::Zero();
    // Rqe3_cal(0) = 2*(q3*q1+q0*q2);
    // Rqe3_cal(1) = 2*(q3*q2-q0*q1);
    // Rqe3_cal(2) = (q0*q0+q3*q3-q1*q1-q2*q2);

    // cout << "Rqe3 - Rqe3_cal: "<< Rqe3-Rqe3_cal;

    // # Four output
    T y1, y2, y3, y4;

    y1 = q(0)-x_f;
    y2 = q(1)-y_f;
    y3 = q(2)-z_f;
    y4 = atan2(Rqe1_x,Rqe1_y)-M_PI/8;
    
    Vector3<T>eta1 = Eigen::Vector3d::Zero();
    T eta5 =y4;

    eta1 = Eigen::Vector3d(y1,y2,y3);
    eta5 = y4;
    
    // cout <<"y4" << y4;
   
    // # First derivative of first three output and yaw output
    T dy1, dy2, dy3, dy4, x2y2, eta6;
    Vector3<T> eta2 = Eigen::Vector3d::Zero();
    eta2 = v - Eigen::Vector3d(dx_f,dy_f,0);
    dy1 = eta2(0);
    dy2 = eta2(1);
    dy3 = eta2(2);
    dy4 = 0;
    // cout << "eta2 :" << eta2;
    
    x2y2 = (pow(Rqe1_x,2)+pow(Rqe1_y,2));

    Vector3<T> eta6_temp = Eigen::Vector3d::Zero(); // eta6_temp = (ye2T-xe1T)/(x^2+y^2)
    eta6_temp = (Rqe1_y*e2.transpose()-Rqe1_x*e1.transpose())/x2y2;  
    // print("eta6_temp:", eta6_temp)
     // Body frame w  ( multiply R)
    eta6 = eta6_temp.dot(-Rqe1_hat*Rw)-dy4;

    // World frame w
    // eta6 = eta6_temp.dot(-Rqe1_hat*w) -dy4;
    // print("Rqe1_hat:", Rqe1_hat)

    // # Second derivative of first three output
    T ddy1, ddy2, ddy3;
    Vector3<T> eta3 = Eigen::Vector3d::Zero(); // eta6_temp = (ye2T-xe1T)/(x^2+y^2)
    eta3 = -g_*e3+Rqe3*xi1 - Eigen::Vector3d(ddx_f,ddy_f,0);
    ddy1 = eta3(0);
    ddy2 = eta3(1);
    ddy3 = eta3(2);

    // # Third derivative of first three output
    T dddy1, dddy2, dddy3;
    Vector3<T> eta4 = Eigen::Vector3d::Zero(); 
    // # Body frame w ( multiply R)
    eta4 = Rqe3*xi2+(-Rqe3_hat*Rw)*xi1 - Eigen::Vector3d(dddx_f,dddy_f,0);
    // # World frame w 
    // eta4 = Rqe3*xi2+F1q*Eq.transpose()*w*xi1 - Eigen::Vector3d(dddx_f,dddy_f,0);
    dddy1 = eta4(0);
    dddy2 = eta4(1);
    dddy3 = eta4(2);

    // # Fourth derivative of first three output
    Vector3<T> B_qw_temp = Eigen::Vector3d::Zero();
    Vector3<T> B_qw = Eigen::Vector3d::Zero();
    
    // # Body frame w 
    B_qw_temp = xi1*(-Rw_hat*Rqe3_hat*Rw+Rqe3_hat*Rq.matrix()*I_inv*wIw); //# np.dot(I_inv,wIw)*xi1-2*w*xi2
    B_qw      = B_qw_temp+xi2*(-2*Rqe3_hat*Rw) - Eigen::Vector3d(ddddx_f,ddddy_f,0); //   #np.dot(Rqe3_hat,B_qw_temp)

    // # World frame w
    // B_qw_temp = xi1*(-w_hat*Rqe3_hat*w+Rqe3_hat*I_inv*wIw); // # np.dot(I_inv,wIw)*xi1-2*w*xi2
    // B_qw      = B_qw_temp+xi2*(-2*Rqe3_hat*w) - Eigen::Vector3d(ddddx_f,ddddy_f,0); //   #np.dot(Rqe3_hat,B_qw_temp)

    // # Second derivative of yaw output
    T dRqe1_x, dRqe1_y, alpha1, B_yaw;

    // # Body frame w
    dRqe1_x = e2.dot(-Rqe1_hat*Rw); 
    dRqe1_y = e1.dot(-Rqe1_hat*Rw); 
    // # World frame w
    // dRqe1_x = e2.dot(-Rqe1_hat*w); // # \dot{x}
    // dRqe1_y = e1.dot(-Rqe1_hat*w); // # \dot{y}

    alpha1 = 2*(Rqe1_x*dRqe1_x+Rqe1_y*dRqe1_y)/x2y2; //# (2xdx +2ydy)/(x^2+y^2)
    // # alpha2 = math.pow(dRqe1_y,2)-math.pow(dRqe1_x,2)

    Vector3<T> B_yaw_temp3 = Eigen::Vector3d::Zero();
    
    // Body frame w
    B_yaw_temp3 = alpha1*Rqe1_hat*Rw+Rqe1_hat*Rq.matrix()*I_inv*wIw-Rw_hat*Rqe1_hat*Rw;

    // World frame w

    // B_yaw_temp3 = alpha1*Rqe1_hat*w+Rqe1_hat*I_inv*wIw-w_hat*Rqe1_hat*w;

    B_yaw = eta6_temp.dot(B_yaw_temp3); // # +alpha2 :Could be an error in math.

    Vector3<T> g_yaw = Eigen::Vector3d::Zero();
     // Body frame w
    g_yaw = -eta6_temp.transpose()*Rqe1_hat*Rq.matrix()*I_inv;

    // World frame w
    // g_yaw = -eta6_temp.transpose()*Rqe1_hat*I_inv;


    // print("g_yaw:", g_yaw)
    // # Decoupling matrix A(x)\in\mathbb{R}^4
    MatrixX<T> A_fl = Eigen::MatrixXd::Zero(4,4);
    A_fl.block(0,0,3,1) = Rqe3;
    // Body frame w
    A_fl.block(0,1,3,3) = -Rqe3_hat*Rq.matrix()*I_inv*xi1;
    // World frame w 
    // A_fl.block(0,1,3,3) = -Rqe3_hat*I_inv*xi1;
    A_fl.block(3,1,1,3) = g_yaw.transpose();

    T A_fl_det;
    MatrixX<T> A_fl_inv = Eigen::MatrixXd::Zero(4,4);
    
    A_fl_inv = A_fl.inverse();
    A_fl_det = A_fl.determinant();
    //# print("I_inv:", I_inv)
    // cout << "\n Rqe3 : \n" << Rqe3 ;
    // cout << "\n -Rqe3_hat*I_inv*xi1 : \n" << -Rqe3_hat*I_inv*xi1 ;
    // cout << "\n g_yaw : \n" << g_yaw ;
    
    cout << "\n A_fl: \n"<< A_fl << "\n";
    // cout << "\n A_fl_det: \n"<< A_fl_det << "\n";



    // # Output dyamics
    VectorX<T> eta = Eigen::VectorXd::Zero(14);
    eta << eta1, eta2, eta3, eta5, eta4, eta6;
    
    T eta_norm;
    eta_norm = eta.dot(eta);
    // # Full feedback controller
    VectorX<T> U_temp = Eigen::VectorXd::Zero(4);

    U_temp <<  B_qw, B_yaw;

    // cout << "\n B_qw: \n" << B_qw;
    // cout << "\n B_yaw: \n" << B_yaw;
    // cout << "\n U_temp: \n" << U_temp;
    // cout << "\n eta_norm: \n" << eta.dot(Mout_*eta) << "\n";
    
    VectorX<T> mu = Eigen::VectorXd::Zero(4);
    MatrixX<T> k = Eigen::MatrixXd::Zero(4,14);
                
    k = Bout_.transpose()*Mout_;
    // cout<< "\n Bout_ : \n" << Bout_;
    // cout<< "\n Mout_ : \n" << Mout_;
    // cout<< "\n k : \n" << k;


    mu = -k*eta;

    VectorX<T> v_temp = Eigen::VectorXd::Zero(4);
    VectorX<T> U_fl = Eigen::VectorXd::Zero(4);
    v_temp=-U_temp+mu;

    // cout << "\n v_temp : \n" << v_temp ;
    
    U_fl = A_fl_inv*v_temp; //       # Feedback controller
    // U_fl =A_fl.ldlt().solve(v_temp);

    // cout << "\n Feedback Controller : \n" << U_fl << "\n";

// CLF-QP problem
    solvers::MathematicalProgram prog;
    auto u_var = prog.NewContinuousVariables(4, "u_var");
    solvers::GurobiSolver solver;

    bool avail = solver.available();
    cout << "\n Gurobi Available? : " << avail ;

// CLF-QP set up

    Eigen::MatrixXd FP_PF = Eigen::MatrixXd::Zero(14,14);
    Eigen::MatrixXd PG = Eigen::MatrixXd::Zero(14,4);     
    FP_PF = Aout_.transpose()*Mout_+Mout_*Aout_;
    PG = Mout_*Bout_;
    
    T L_FVx, Vx;
    Eigen::VectorXd L_GVx = Eigen::VectorXd::Zero(4);
    Eigen::VectorXd L_fhx_star = Eigen::VectorXd::Zero(4); 
    Eigen::RowVectorXd phi1_decouple = Eigen::RowVectorXd::Zero(4); 
    
    L_FVx = eta.dot(FP_PF*eta);
    L_GVx = 2*eta.transpose()*PG; // # row vector
    L_fhx_star = U_temp;

    // cout << "\n L_fhx_star : " << L_fhx_star <<"\n";

    // cout << "\n U_temp : " << U_temp <<"\n";
  
    Vx = eta.dot(Mout_*eta);

    T phi0_exp, constraint_gain;

    constraint_gain = 1e0;
    // phi0_exp = L_FVx+L_GVx.dot(L_fhx_star)+(min_e_Q_/max_e_P_)*Vx*1; //   # exponentially stabilizing
    phi0_exp = L_FVx+L_GVx.dot(L_fhx_star)+min_e_Q_*eta_norm*1e0;    // # more exact bound - exponentially stabilizing
    phi1_decouple = L_GVx.transpose()*A_fl;

    cout << "\n size of phi1_decouple : "<< phi1_decouple.rows() << "\n";
    phi0_exp = phi0_exp*constraint_gain;
    phi1_decouple = phi1_decouple.transpose()*constraint_gain;
    // cout << "\n phi1_decouple : "<< phi1_decouple <<"\n";
    // cout << "\n L_GVx : "<< L_GVx <<"\n";
    
    // # # Solve QP
    // Vector4<drake::symbolic::Variables> v_var = Eigen::VectorXd::Zero(4); // WRONG
    Eigen::VectorXd c_QP = Eigen::VectorXd::Zero(4);
    // Eigen::MatrixXd Quadratic_Positive_def = Eigen::MatrixXd::Identity(4,4);
    // Quadratic_Positive_def = 1e4*Quadratic_Positive_def;
    // Quadratic_Positive_def(0,0) = 1e0;
    Eigen::MatrixXd Quadratic_Positive_def = Eigen::MatrixXd::Zero(4,4);
    T QP_det;

    // // v_var = A_fl*u_var  + L_fhx_star;                // TO DO: How to convert Matrix <symbolic::variable> to Extpression?
    Quadratic_Positive_def = 2*A_fl.transpose()*A_fl;
    QP_det = 1*Quadratic_Positive_def.determinant();
    c_QP = 2*L_fhx_star.transpose()*A_fl;
    
   
    
    // TO DO : Convert the Matrix<symbolic::variable> to Expression so that we can have general expression of the cost function.
    // # CLF_QP_cost_v = np.dot(v_var,v_var) // Exact quadratic cost
    // CLF_QP_cost_v_effective = np.dot(u_var, np.dot(Quadratic_Positive_def,u_var))+np.dot(c_QP,u_var) # Quadratic cost without constant term
    // # CLF_QP_cost_u = np.dot(u_var,u_var)
    // phi1 = np.dot(phi1_decouple,u_var)
    // auto quadratic_cost = symbolic::Expression(u_var[0]);
    
    prog.AddQuadraticCost(Quadratic_Positive_def, c_QP, u_var);
    prog.AddLinearConstraint(phi1_decouple.transpose(), -1e32, -phi0_exp, u_var);

    cout << "\n Quadratic_Positive_def : "<< Quadratic_Positive_def <<"\n";
    // cout << "\n c_QP :" << c_QP.transpose() <<"\n";
    cout << "\n CLF value:" << Vx <<"\n"; //  # Current CLF value
    cout << "\n eta norm value:" << eta_norm <<"\n"; //  # Current CLF value
    cout << "\n min_e_Q_:" << min_e_Q_ <<"\n"; //  # Current CLF value
    cout << "\n max_e_P_ :" << max_e_P_ <<"\n"; //  # Current CLF value

    prog.SetInitialGuess(u_var, U_fl);
    solvers::SolverId solverid = solver.solver_id();
    // prog.SetSolverOption(solverid, "print_level", 5); // # CAUTION: Assuming that solver used Ipopt

    clock_t t_clock;
    
    t_clock = clock();
    solvers::SolutionResult result = solver.Solve(prog);
    // solvers::SolutionResult result = prog.Solve();  // # Solve with default osqp
    t_clock = clock() - t_clock;
    
    cout << "\n Solution result : " << result <<"\n";
    std::vector<solvers::Binding<solvers::LinearConstraint>> allconstraint = prog.GetAllLinearConstraints(); // Get the vectors of Binding of constraints
    const solvers::Binding<solvers::LinearConstraint>* allconstraint_binding = allconstraint.data(); // Convert it to Binding class pointer   

    Eigen::VectorXd allconstraint_vector = prog.EvalBindingAtSolution(allconstraint_binding[0]); // Evaluate the Binding for the first (0) constraint at the soultion 


    cout << "\n all constraint : " << allconstraint_vector[0] + phi0_exp <<"\n";
    cout << "\n phi10_exp :" << phi0_exp << "\n";
    cout << "\n phi1_decouple :" << phi1_decouple << "\n";
    // solver.Solve(prog)
    // cout << "Optimal u : " << prog.GetSolution(u_var) << "\n";
    Eigen::VectorXd U_CLF_QP = Eigen::VectorXd::Zero(4);
    U_CLF_QP = prog.GetSolution(u_var);
    
    cout << "\n Quadratic Programming spent "<< (static_cast<float>(t_clock))*1000/CLOCKS_PER_SEC <<" milisecond(ms)."<< endl;

    Eigen::VectorXd v_CLF = Eigen::VectorXd::Zero(4);
    Eigen::VectorXd v_FL = Eigen::VectorXd::Zero(4);
    
    T phi1_opt, phi1_opt_FL;
    phi1_opt = phi1_decouple.dot(U_CLF_QP);
    phi1_opt_FL = phi1_decouple.dot(U_fl);

    cout<< "\n FL u: "<< U_fl;
    cout<< "\n CLF u:"<< U_CLF_QP;
    

    v_FL = A_fl*U_fl+L_fhx_star;
    v_CLF = A_fl*U_CLF_QP+L_fhx_star;

    cout<< "\n Cost FL: "<< v_FL.norm();
    cout<< "\n Cost CLF: "<< v_CLF.norm();
    cout<< "\n Total energy FL: "<< U_fl.norm();
    cout<< "\n Total evergy CLF: "<< U_CLF_QP.norm();
    cout<< "\n Constraint FL : "<< phi0_exp+phi1_opt_FL;
    cout<< "\n Constraint CLF : "<< phi0_exp+phi1_opt;
    cout<< "\n Vdot FL : "<<  L_FVx+L_GVx.dot(L_fhx_star)+phi1_opt_FL;
    cout<< "\n Vdot CLF : "<<  L_FVx+L_GVx.dot(L_fhx_star)+phi1_opt;
    
    // CLF-QP Control
    Eigen::VectorXd u = Eigen::VectorXd::Zero(4);
    u = U_CLF_QP;

    output->set_value(u);


  }

 private:
  RobobeePlant<T> robobee_;
  // The implementation above is (and must remain) careful to not store hidden
  // state in here.  This is only used to avoid runtime allocations.

  static constexpr int kStateDimension{4};
  static constexpr int kInputDimension{15};

  const std::unique_ptr<systems::Context<T>> robobee_context_;

  double g_;           // Gravitational acceleration (m/s^2).
  double m_;           // Mass of the robot (kg).
  double min_e_Q_;
  double max_e_P_;
  Eigen::Matrix3d I_;  // Moment of Inertia about the Center of Mass
  Eigen::MatrixXd Mout_;  // Moment of Inertia about the Center of Mass
  Eigen::MatrixXd Aout_;  // A for Output linear system
  Eigen::MatrixXd Bout_;  // B for Output linear system
};

}  // namespace acrobot
}  // namespace examples
}  // namespace drake
