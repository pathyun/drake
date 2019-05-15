#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/examples/quadrotor/quadrotor_plant.h"

using std::make_unique;
using std::unique_ptr;
using std::vector;

namespace drake {
namespace pydrake {

PYBIND11_MODULE(quadrotor, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::systems;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::examples::quadrotor;

  m.doc() = "Bindings for the Acrobot example.";

  py::module::import("pydrake.systems.framework");

  // TODO(eric.cousineau): At present, we only bind doubles.
  // In the future, we will bind more scalar types, and enable scalar
  // conversion. Issue #7660.
  using T = double;

  py::class_<QuadrotorPlant<T>, LeafSystem<T>>(m, "QuadrotorPlant")
      .def(py::init<>())
      .def("set_state", &QuadrotorPlant<T>::set_state)
      .def("CalcKineticEnergy", &QuadrotorPlant<T>::get_input_size)
      .def("get_num_states", &QuadrotorPlant<T>::get_num_states)
      .def("m", &QuadrotorPlant<T>::m)
      .def("g", &QuadrotorPlant<T>::g);

}

}  // namespace pydrake
}  // namespace drake
