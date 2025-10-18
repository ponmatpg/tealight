#include <functional>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <sstream>
#include <string>
#include <vector>

namespace nb = nanobind;

template <typename T = double> class Value {
public:
  Value(T val) : value_(val) {}

  // mathematical operators
  Value operator+(Value &other) {
    Value out = Value(this->value_ + other.value_);
    return out;
  }

  Value operator-(Value &other) { return Value(value_ - other.value_); }

  Value operator*(Value &other) { return Value(value_ * other.value_); }

  Value operator/(Value &other) { return Value(value_ * other.value_); }

  std::string describe() const {
    std::stringstream ss;
    ss << "Value: " << value_ << std::endl;
    return ss.str();
  }

private:
  T value_;
  T grad_{0};
  std::vector<T> children_;
};

NB_MODULE(libvalue, m) {
  m.doc() = "Differentiable scalar value";

  using FLOAT_T = double;
  using ValueType = Value<FLOAT_T>;

  nb::class_<ValueType>(m, "Value")
      .def(nb::init<FLOAT_T>())
      .def("__add__",
           [](ValueType &self, ValueType &other) -> ValueType {
             return self + other;
           })
      .def("__sub__",
           [](ValueType &self, ValueType &other) -> ValueType {
             return self - other;
           })
      .def("__mul__",
           [](ValueType &self, ValueType &other) -> ValueType {
             return self * other;
           })
      .def("__truediv__",
           [](ValueType &self, ValueType &other) -> ValueType {
             return self / other;
           })
      .def("describe", &Value<FLOAT_T>::describe);
}