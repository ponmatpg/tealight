#include <nanobind/stl/string.h>
#include <string>

namespace nb = nanobind;

class Calculator {
public:
  Calculator(int initial_value) : value_(initial_value) {}

  int add(int x) {
    value_ += x;
    return value_;
  }

  int get_value() const { return value_; }

  std::string describe() const {
    return "Calculator with value: " + std::to_string(value_);
  }

private:
  int value_;
};

NB_MODULE(libexample, m) {
  m.doc() = "Example nanobind module";

  nb::class_<Calculator>(m, "Calculator")
      .def(nb::init<int>())
      .def("add", &Calculator::add)
      .def("get_value", &Calculator::get_value)
      .def("describe", &Calculator::describe);

  m.def("multiply", [](int a, int b) { return a * b; }, "Multiply two numbers");
}