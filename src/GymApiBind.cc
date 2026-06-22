#include <GymApi.h>
#include <typedefs.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(omnetbind, m) {
    m.doc() = "binding module to run Omnet++ simulation from Python";
    
    // Bind rl::Observation to allow conversion to Python
    py::class_<rl::Observation>(m, "Observation")
        .def(py::init<>())
        .def("size", &rl::Observation::size)
        .def("__len__", &rl::Observation::size)
        .def("__getitem__", [](const rl::Observation& obs, std::size_t i) -> py::object {
            const auto& field = obs.at(i);
            return std::visit([](auto&& v) -> py::object {
                return py::cast(v);
            }, field);
        })
        .def("to_list", [](const rl::Observation& obs) {
            py::list result;
            for (const auto& field : obs.data()) {
                result.append(std::visit([](auto&& v) -> py::object {
                    return py::cast(v);
                }, field));
            }
            return result;
        });
    
    // Translates python function calls to GymApi methods
    py::class_<GymApi>(m, "OmnetGymApi")
        .def(py::init<>())
        .def("initialise", &GymApi::initialise)
        .def("sim_time", &GymApi::simTime)
        .def("reset", &GymApi::reset)
        .def("step", &GymApi::step)
        .def("shutdown", &GymApi::shutdown)
        .def("cleanup", &GymApi::cleanupmemory);

}

    
