/*
  run_robobee_FL_quaternion.cc

  Objective : Solve trajectory tracking for Robobee dynamics (w/ quaternion)
  Algorithm : Output Feedback Linearization with to stable zereo dynamics z= qTq-1 
  Controller : RobobeeFLController.h

  Remark : 1) get_output_port() for integrator expect no integer argument otherwise it will destruct itself
           2) Need to carefully wire everything in diagram to run simulation other wise Segmentation fault(core dump)
  
  Author : Nak-seung Patrick Hyun
  Date : 08/07/2018
*/

#include <memory>

#include <gflags/gflags.h>

#include "drake/math/continuous_algebraic_riccati_equation.h"
#include "drake/common/find_resource.h"
#include "drake/common/is_approx_equal_abstol.h"
#include "drake/examples/robobee/robobee_plant.h"
#include "drake/examples/robobee/RobobeeFLController.h"
#include "drake/systems/primitives/multiplexer.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/primitives/integrator.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

#include "drake/multibody/joints/floating_base_types.h"

// DEFINE_int32(simulation_trials, 2, "Number of trials to simulate.");
DEFINE_double(simulation_real_time_rate, 1.0, "Real time rate");
DEFINE_double(trial_duration, 10.0, "Duration of execution of each trial");

namespace drake {
using systems::DiagramBuilder;
using systems::Simulator;
using systems::Context;
using systems::ContinuousState;
using systems::VectorBase;

namespace examples {
namespace robobee {
namespace {

int do_main() {


// [1] Build the system

  lcm::DrakeLcm lcm;

  DiagramBuilder<double> builder;
 
  std::cout << "1. Building RigidBodyTree for Robobee \n";
  auto tree = std::make_unique<RigidBodyTree<double>>();                  // Make a rigidbodytree
  parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      FindResourceOrThrow("drake/examples/robobee/robobee_twobar.urdf"),
      multibody::joints::kQuaternion, tree.get());

  // // The nominal hover position is at (0, 0, 1.0) in world coordinates.
  // const Eigen::Vector3d kNominalPosition{((Eigen::Vector3d() << 0.0, 0.0, 0.10).
  //     finished())};

  std::cout << "2-1. Adding Robobee Plant \n";
  auto robobee = builder.AddSystem<RobobeePlant<double>>();
  robobee->set_name("robobee");

  std::cout << "2-2. Adding Two Integrators \n"; 

  int num_input_integrator;
  num_input_integrator =1;

  auto integrator1 = builder.AddSystem<systems::Integrator<double>>(1);
  integrator1->set_name("integrator1");
  // const auto& port = integrator1->get_output_port();

  auto integrator2 = builder.AddSystem<systems::Integrator<double>>(1);
  integrator2->set_name("integrator2");

  std::cout << "2-3. Adding Three Multiplexer \n"; 

  int num_input_Mux2toV;
  num_input_Mux2toV =2;

  std::cout << "2-3-1. Mux 2 to Vector \n"; 

  auto Mux2toV = builder.AddSystem<systems::Multiplexer<double>>(num_input_Mux2toV);

  Mux2toV->set_name("Mux2toV");

  std::vector<int> Mux2toV_input_sizes(2);
  Mux2toV_input_sizes[0]=13;
  Mux2toV_input_sizes[1]=2;

  std::cout << "2-3-2. Mux 15 to Vector \n"; 

  auto Mux15toV = builder.AddSystem<systems::Multiplexer<double>>(Mux2toV_input_sizes);
  Mux15toV->set_name("Mux15toV");
  
  std::vector<int> Mux4toV_input_sizes(2);
  Mux4toV_input_sizes[0]=1;
  Mux4toV_input_sizes[1]=3;

  std::cout << "2-3-3. Mux 4 to Vector \n"; 

  auto Mux4toV = builder.AddSystem<systems::Multiplexer<double>>(Mux4toV_input_sizes);
  Mux4toV->set_name("Mux4toV");
  
  int num_input_Mux3toV;
  num_input_Mux3toV =3;

  std::cout << "2-3-4. Mux 3 to Vector \n"; 

  auto Mux3toV = builder.AddSystem<systems::Multiplexer<double>>(num_input_Mux3toV);
  Mux3toV->set_name("Mux3toV");
  std::cout << "2-4. Adding one Demultiplexer V(4) to 4 \n"; 

  int num_output_Mux2toV =4;

  auto DemuxVto4 = builder.AddSystem<systems::Demultiplexer<double>>(num_output_Mux2toV, 1); // Creating 4 output ports
  DemuxVto4->set_name("DemuxVto4");
  
  // std::cout << DemuxVto4->get_num_output_ports();
  //   // std::cout << "2. Adding Controller \n";
  //   // auto controller = builder.AddSystem(FeedbackLinearizationController(
  //   //     robobee, kNominalPosition));
  //   // controller->set_name("controller");
  
  std::cout << "2-5. Adding Output Feedback Linearization Controller \n"; 
  
  RobobeePlant<double> robobeeP;
  double m_, g_;
  g_ = 9.81;
  Eigen::Matrix3d I_;
  
  m_=robobeeP.m();
  I_=robobeeP.I();
  auto controller = builder.AddSystem<RobobeeFLController>(m_, I_);
  controller->set_name("controller");
   
  std::cout << controller->get_num_output_ports();

  std::cout << "3. Adding Visualizer \n";
  auto visualizer =
      builder.AddSystem<drake::systems::DrakeVisualizer>(*tree, &lcm);
  visualizer->set_name("visualizer");


  std::cout << "4. Connect systems \n";
  std::cout << "4-1. Backstepped system \n";
  
  builder.Connect(robobee->get_output_port(0), Mux15toV->get_input_port(0)); //  Connect 13 state to Mux15toV
  builder.Connect(integrator1->get_output_port(), Mux2toV->get_input_port(0)); // Connect two integrator output to Mux2toV
  builder.Connect(integrator2->get_output_port(), Mux2toV->get_input_port(1));
  builder.Connect(Mux2toV->get_output_port(0), Mux15toV->get_input_port(1));    // Connect 2 more state to Mux15toV

  // std::cout << "4-1. Non-Backstepped system \n";
  
  // builder.Connect(robobee->get_output_port(0), controller->get_input_port(0)); //  Connect 13 state to Mux15toV
  // builder.Connect(controller->get_output_port(0), DemuxVto4->get_input_port(0)); // Get the controller 4 control
  // builder.Connect(DemuxVto4->get_output_port(0), integrator1->get_input_port()); // Snap control feed in to integrator1
  // builder.Connect(integrator1->get_output_port(), Mux4toV->get_input_port(0)); //  Connect 13 state to Mux15toV
  
  // builder.Connect(Mux3toV->get_output_port(0), Mux4toV->get_input_port(1));


  std::cout << "4-2. Connecting BS Controler \n";
  // Connect thee controller to Demux
  builder.Connect(Mux15toV->get_output_port(0), controller->get_input_port(0)); // Feed in to Controller 15 iniput
  builder.Connect(controller->get_output_port(0), DemuxVto4->get_input_port(0)); // Get the controller 4 control
  builder.Connect(DemuxVto4->get_output_port(0), integrator1->get_input_port()); // Snap control feed in to integrator1
  builder.Connect(integrator1->get_output_port(), integrator2->get_input_port());// Jerk control feed in to integrator2

  std::cout << "4-2-1. Connecting BS Controler (torque) \n";
  builder.Connect(DemuxVto4->get_output_port(1), Mux3toV->get_input_port(0)); // Torque roll to Mux3toV
  builder.Connect(DemuxVto4->get_output_port(2), Mux3toV->get_input_port(1)); // Torque pitch to Mux3toV
  builder.Connect(DemuxVto4->get_output_port(3), Mux3toV->get_input_port(2)); // Torque yaw to Mux3toV
  
  std::cout << "4-3. Connecting Actual Controler \n";

  builder.Connect(integrator2->get_output_port(), Mux4toV->get_input_port(0));
  builder.Connect(Mux3toV->get_output_port(0), Mux4toV->get_input_port(1));
  builder.Connect(Mux4toV->get_output_port(0), robobee->get_input_port(0));
  
  std::cout << "4-4. Connecting to DrakeVisualizer \n";
  builder.Connect(robobee->get_output_port(0), visualizer->get_input_port(0));
  
  // std::cout << "num_input visualizer:" << visualizer->get_output_port();


  std::cout << "5. Build diagram and run simulator \n";
  auto diagram = builder.Build();
  Simulator<double> simulator(*diagram);
  VectorX<double> x0 = VectorX<double>::Zero(13);
  VectorX<double> x1 = VectorX<double>::Zero(15);

  // const VectorX<double> kNominalState{((Eigen::VectorXd(13) << kNominalPosition, 1.0, 0.0, 0.0, 0.0,
  // Eigen::VectorXd::Zero(6)).finished())};

  // srand(42);

    x0 = VectorX<double>::Random(13)*1;
    
    // #-[0] Initial condition
    Vector3<double> r0(0,0,0.05); 
    Vector4<double> q0;
    double theta0 = M_PI/4; //  # angle of rotation
    Vector3<double> v0_q(0.,0.,1.);
    q0(0)=cos(theta0/2);  // #q0
    double v0_norm = sqrt(v0_q.dot(v0_q));
    Vector3<double> v0_normalized;
    v0_normalized =v0_q/v0_norm;
    
    q0 << q0(0), sin(theta0/2)*v0_normalized;

    Vector3<double> v0(0,0,0);
    Vector3<double> w0(-1,0,1);
    double xi10, xi20;
    xi10 = g_;
    xi20 =0;
    
    x0 << r0, q0, v0, w0;
    

    x1(2)=0.03;
    x1(3)=1.;
    std::cout << "\n State need to initialize: " << simulator.get_mutable_context().get_num_total_states() << "\n";

    // // Initial condition for integrator
    VectorX<double> xi2(1);
    xi2[0]= 0;
    VectorX<double> xi1(1);
    xi1[0]= 9.8;
    
    std::vector<const systems::System<double>*> list_syst = diagram->GetSystems();
    std::cout << "\n Subsystem 0 is robobee? : " << list_syst[0]->get_name() ;
    std::cout << "\n Subsystem 1 is robobee? : " << list_syst[1]->get_name() ;
    std::cout << "\n Subsystem 2 is robobee? : " << list_syst[2]->get_name() << "\n";

    // // // diagram->GetMutableSubsystemContext (list_syst[1], simulator.get_mutable_context())
    // // // systems::System<double> subsystem(&list_syst[1]);
    // // // diagram->GetMutableSubsystemContext(list_syst[1], simulator.get_mutable_context());
    
    robobee->set_state( & diagram->GetMutableSubsystemContext(list_syst[0][0], & simulator.get_mutable_context()),x0);
    integrator1->set_integral_value( & diagram->GetMutableSubsystemContext(list_syst[1][0], & simulator.get_mutable_context()),xi2);
    integrator2->set_integral_value( & diagram->GetMutableSubsystemContext(list_syst[2][0], & simulator.get_mutable_context()),xi1);
    
    // simulator.get_mutable_context()
    //     .get_mutable_continuous_state_vector()
    //     .SetFromVector(x0);


    simulator.Initialize();
    simulator.set_target_realtime_rate(FLAGS_simulation_real_time_rate);
    simulator.get_mutable_integrator()->set_fixed_step_mode(false);
    simulator.get_mutable_integrator()->set_maximum_step_size(0.005);
    simulator.StepTo(FLAGS_trial_duration);

  return 0;
}

}  // namespace
}  // namespace quadrotor
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::robobee::do_main();
}
