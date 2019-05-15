/// Quaternion robobee lqr controller.
/// 
/// Solution to CARE give singular matrix.... 07232018   
///

#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/is_approx_equal_abstol.h"
#include "drake/examples/robobee/robobee_plant.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_int32(simulation_trials, 1, "Number of trials to simulate.");
DEFINE_double(simulation_real_time_rate, 1.0, "Real time rate");
DEFINE_double(trial_duration, 7.0, "Duration of execution of each trial");

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
  lcm::DrakeLcm lcm;

  DiagramBuilder<double> builder;

  std::cout << "1. Building RigidBodyTree for Robobee \n";
  auto tree = std::make_unique<RigidBodyTree<double>>();
  parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      FindResourceOrThrow("drake/examples/robobee/robobee.urdf"),
      multibody::joints::kQuaternion, tree.get());

  // The nominal hover position is at (0, 0, 1.0) in world coordinates.
  const Eigen::Vector3d kNominalPosition{((Eigen::Vector3d() << 0.0, 0.0, 0.10).
      finished())};

  std::cout << "2. Adding Robobee Plant \n";
  auto robobee = builder.AddSystem<RobobeePlant<double>>();
  robobee->set_name("robobee");


  std::cout << "2. Adding Controller \n";
  auto controller = builder.AddSystem(StabilizingLQRController(
      robobee, kNominalPosition));
  controller->set_name("controller");


  std::cout << "3. Adding Visualizer \n";
  auto visualizer =
      builder.AddSystem<drake::systems::DrakeVisualizer>(*tree, &lcm);
  visualizer->set_name("visualizer");


  std::cout << "4. Connect systems \n";
  builder.Connect(robobee->get_output_port(0), controller->get_input_port());
  builder.Connect(controller->get_output_port(), robobee->get_input_port(0));
  builder.Connect(robobee->get_output_port(0), visualizer->get_input_port(0));


  std::cout << "5. Build diagram and run simulator \n";
  auto diagram = builder.Build();
  Simulator<double> simulator(*diagram);
  VectorX<double> x0 = VectorX<double>::Zero(13);

  const VectorX<double> kNominalState{((Eigen::VectorXd(13) << kNominalPosition, 1.0, 0.0, 0.0, 0.0,
  Eigen::VectorXd::Zero(6)).finished())};

  srand(42);

  for (int i = 0; i < FLAGS_simulation_trials; i++) {
    auto diagram_context = diagram->CreateDefaultContext();
    x0 = VectorX<double>::Random(13)*1;
    x0(0)=0;
    x0(1)=0;
    x0(2)=0.03;
    x0(3)=1.;// Quaternion theta=0
    x0(4)=0;
    x0(5)=0;
    x0(6)=0;
    x0(7)=0;// CoM velocities
    x0(8)=0;
    x0(9)=0;


    simulator.get_mutable_context()
        .get_mutable_continuous_state_vector()
        .SetFromVector(x0);

    simulator.Initialize();
    simulator.set_target_realtime_rate(FLAGS_simulation_real_time_rate);
    simulator.StepTo(FLAGS_trial_duration);

    // Goal state verification.
    const Context<double>& context = simulator.get_context();
    const ContinuousState<double>& state = context.get_continuous_state();
    const VectorX<double>& position_vector = state.CopyToVector();
    std::cout << position_vector(3);
    if (!is_approx_equal_abstol(
        position_vector, kNominalState, 1e-2)) {
      
      throw std::runtime_error("Target state is not achieved.");
    }

    simulator.reset_context(std::move(diagram_context));
  }
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
