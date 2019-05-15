// Generates a swing-up trajectory for robobee and displays the trajectory
// in DrakeVisualizer. Trajectory generation code is based on
// pendulum_swing_up.cc.

#include <iostream>
#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/examples/robobee/robobee_plant_rpy.h" // Load RobobeePlan with Roll Pitch Yaw 12 state
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/joints/floating_base_types.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/rigid_body_tree_construction.h"

using drake::solvers::SolutionResult;


namespace drake {
namespace examples {
namespace robobee {

typedef trajectories::PiecewisePolynomial<double> PiecewisePolynomialType;

namespace {
DEFINE_double(realtime_factor, 1.0,
              "Playback speed.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");

int do_main() {
  systems::DiagramBuilder<double> builder;

  RobobeePlant<double> robobee;
  auto context = robobee.CreateDefaultContext();

 
  // auto robobee = builder.AddSystem<RobobeePlant<double>>();
  // robobee->set_name("robobee");
  // //RobobeePlant<double> robobee;
  // auto context = robobee->CreateDefaultContext();

  const int kNumTimeSamples = 20;
  const double kMinimumTimeStep = 0.02;
  const double kMaximumTimeStep = 0.5;
  systems::trajectory_optimization::DirectCollocation dircol(
      &robobee, *context, kNumTimeSamples, kMinimumTimeStep,
      kMaximumTimeStep);

  dircol.AddEqualTimeIntervalsConstraints();

  // 
  // const double kTorqueLimit = 8000;

  std::cout << "State size: " << robobee.get_num_states() << "\n";

  std::cout << "dir state size" << context->get_continuous_state().size();
  auto u = dircol.input();
  // dircol.AddConstraintToAllKnotPoints(-kTorqueLimit <= u(0));
  // dircol.AddConstraintToAllKnotPoints(u(0) <= kTorqueLimit);

// [0] Initial configuration in SE(3)
  Eigen::VectorXd x0=Eigen::VectorXd::Zero(12);

  // Position r
  x0(0) = 0.;
  x0(1) = 0.;
  x0(2) = 0.;

  // Orientation q (quaternion)
  Eigen::Vector3d rpy0=Eigen::Vector3d:: Zero(3); 
  rpy0(0) = 1.;
  rpy0(1) = 0.;
  rpy0(2) = 1.;
  
  x0(3) =rpy0(0);
  x0(4) =rpy0(1);
  x0(5) =rpy0(2);
  
  // Angular velocity w
  Eigen::Vector3d w0 = Eigen::Vector3d::Zero(3);
  w0(0)= -0.1;
  w0(1)=  0.;
  w0(2)=  0.1 ;
  
  x0(9)=w0(0); // w1=1;
  x0(10)=w0(1);// w2=1;
  x0(11)=w0(2);// w3=1;
  std::cout << "Intial condition x0:" << x0 <<"\n";

// [1] Final configuration in SE(3)
  Eigen::VectorXd xf=Eigen::VectorXd::Zero(12);
  
  // Position r
  xf(0) = 0.;
  xf(1) = 0.;
  xf(2) = 0.3;

  // Orientation q (quaternion)
  Eigen::Vector3d rpyf=Eigen::Vector3d:: Zero(3); 
  rpyf(0) = M_PI;
  rpyf(1) = 0.;
  rpyf(2) = 0.;
  
  xf(3) =rpyf(0);
  xf(4) =rpyf(1);
  xf(5) =rpyf(2);

  // Angular velocity w
  Eigen::Vector3d wf = Eigen::Vector3d::Zero(3);
  wf(0)= -0.;
  wf(1)=  0.;
  wf(2)=  0. ;
  
  xf(9)=wf(0); // w1=1;
  xf(10)=wf(1);// w2=1;
  xf(11)=wf(2);// w3=1;


  std::cout << "Final condition xf:" << xf <<"\n";
 
 // const Eigen::VectorXd xG(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//  dircol.AddLinearConstraint(dircol.initial_state() == x0);
//  dircol.AddLinearConstraint(dircol.final_state() == xG);
  dircol.AddBoundingBoxConstraint(x0, x0,
                                dircol.initial_state());
  dircol.AddBoundingBoxConstraint(xf, xf,
                                dircol.final_state());
  // dircol.AddLinearConstraint(dircol.initial_state() == x0);
  // dircol.AddLinearConstraint(dircol.final_state() == xG);

  //const double R = 10;  // Cost on input "effort".
  Eigen::Matrix4d R = 20*Eigen::Matrix4d::Identity();

  dircol.AddRunningCost( (u.transpose()*R) * u);

  const double timespan_init = 10;
  auto traj_init_x =
      PiecewisePolynomialType::FirstOrderHold({0, timespan_init}, {x0, x0});
  dircol.SetInitialTrajectory(PiecewisePolynomialType(), traj_init_x);


  SolutionResult result = dircol.Solve();

  if (result != SolutionResult::kSolutionFound) {
    std::cerr << "No solution found.\n";
    return 1;
  }  

  const trajectories::PiecewisePolynomial<double> pp_xtraj =
      dircol.ReconstructStateTrajectory();
  auto state_source = builder.AddSystem<systems::TrajectorySource>(pp_xtraj);

  lcm::DrakeLcm lcm;
  auto tree = std::make_unique<RigidBodyTree<double>>();
  parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      FindResourceOrThrow("drake/examples/robobee/robobee.urdf"),
      multibody::joints::kRollPitchYaw, tree.get());
  
  auto publisher = builder.AddSystem<systems::DrakeVisualizer>(*tree, &lcm);

  // By default, the simulator triggers a publish event at the end of each time
  // step of the integrator. However, since this system is only meant for
  // playback, there is no continuous state and the integrator does not even get
  // called. Therefore, we explicitly set the publish frequency for the
  // visualizer.

  std::cout << state_source->get_output_port().size() << std::endl;
  std::cout << publisher->get_input_port(0).size() << std::endl;

  publisher->set_publish_period(1.0 / 60.0);

  builder.Connect(state_source->get_output_port(),
                  publisher->get_input_port(0));

  auto diagram = builder.Build();

  systems::Simulator<double> simulator(*diagram);

  simulator.set_target_realtime_rate(FLAGS_realtime_factor);
  simulator.Initialize();
  // simulator.StepTo(pp_xtraj.end_time());

  std::cout << "Ending time: "<< pp_xtraj.end_time() << "\n";

  Eigen::VectorXd pp_temp = pp_xtraj.value(0.2);
  std::cout << "pp_xtraj: " << pp_temp <<"\n";


  simulator.StepTo(pp_xtraj.end_time());

  
  return 0;
}

}  // namespace
}  // namespace robobee
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::robobee::do_main();
}
