#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/examples/robobee/robobee_plant.h"

using std::make_unique;
using std::unique_ptr;
using std::vector;

namespace drake {
namespace pydrake {

PYBIND11_MODULE(robobee, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::systems;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::examples::robobee;

  m.doc() = "Bindings for the Robobee example.";

  py::module::import("pydrake.systems.framework");

  // TODO(eric.cousineau): At present, we only bind doubles.
  // In the future, we will bind more scalar types, and enable scalar
  // conversion. Issue #7660.
  using T = double;

  py::class_<RobobeePlant<T>, LeafSystem<T>>(m, "RobobeePlant")
      .def(py::init<double, const Eigen::Matrix3d& >(),
           py::arg("m"), py::arg("I_arg"))
      .def("set_state", &RobobeePlant<T>::set_state)
      .def("get_input_size", &RobobeePlant<T>::get_input_size)
      .def("get_num_states", &RobobeePlant<T>::get_num_states)
      .def("m", &RobobeePlant<T>::m)
      .def("g", &RobobeePlant<T>::g);

}

}  // namespace pydrake
}  // namespace drake
