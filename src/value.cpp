#include <cmath>
#include <functional>
#include <iostream>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <ostream>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;

template <typename FpType = double> class Value {
public:
  Value(FpType val, std::vector<Value<FpType> *> children = {})
      : value_(val), children_(children) {}
  Value(Value &other) = delete;

  FpType grad() const { return grad_; }

  void zero_grad() { grad_ = 0; }

  std::string describe() const {
    std::stringstream ss;
    ss << "Value: " << value_ << " Gradient: " << grad_ << " location: " << this
       << std::endl;
    return ss.str();
  }

  // mathematical operators
  Value &operator+(Value &other) {
    Value *out = new Value(this->value_ + other.value_, {this, &other});

    out->backward_ = [this, &other, out]() {
      this->grad_ += out->grad_;
      other.grad_ += out->grad_;
    };

    return *out;
  }

  Value &operator-(Value &other) {
    Value *out = new Value(this->value_ - other.value_, {this, &other});

    out->backward_ = [this, &other, out]() {
      this->grad_ += out->grad_;
      other.grad_ += out->grad_ * -1;
    };

    return *out;
  }

  Value &operator*(Value &other) {
    Value *out = new Value(this->value_ * other.value_, {this, &other});

    out->backward_ = [this, &other, out]() {
      this->grad_ += out->grad_ * other.value_;
      other.grad_ += out->grad_ * this->value_;
    };

    return *out;
  }

  Value &operator/(Value &other) {
    Value *out = new Value(this->value_ / other.value_, {this, &other});

    out->backward_ = [this, &other, out]() {
      this->grad_ += out->grad_ / other.value_;
      other.grad_ +=
          out->grad_ * -1 * this->value_ / (other.value_ * other.value_);
    };

    return *out;
  }

  // represents this ^ other
  Value &pow(Value &other) {
    Value *out =
        new Value(std::pow(this->value_, other.value_), {this, &other});

    out->backward_ = [this, &other, out]() {
      this->grad_ += out->grad_ * out->value_ * other.value_ / this->value_;
      other.grad_ += out->grad_ * out->value_ * std::log(this->value_);
    };

    return *out;
  }

  Value &relu() {
    Value *out = new Value(value_ > 0 ? value_ : 0, {this});

    out->backward_ = [this, out]() {
      this->grad_ += out->grad_ * (this->value_ > 0 ? 1 : 0);
    };

    return *out;
  }

  void backward() {
    std::vector<Value<FpType> *> topo_sorted{};
    std::unordered_set<Value<FpType> *> visited{};
    auto build_topo = [&visited, &topo_sorted](this auto &&self,
                                               Value<FpType> *v) -> void {
      if (auto it = visited.find(v); it == visited.end()) {
        visited.insert(v);
        for (auto p : v->children_) {
          self(p);
        }
        topo_sorted.push_back(v);
      }
    };
    build_topo(this);

    grad_ = 1;
    for (auto v : std::views::reverse(topo_sorted)) {
      v->backward_();
    }
  }

private:
  FpType value_;
  FpType grad_{0};
  std::function<void(void)> backward_ = []() {};
  std::vector<Value<FpType> *> children_;
};

NB_MODULE(libvalue, m) {
  m.doc() = "Differentiable scalar value";

  using FLOAT_T = double;
  using ValueType = Value<FLOAT_T>;

  nb::class_<ValueType>(m, "Value")
      .def(nb::init<FLOAT_T>())
      .def(
          "__add__",
          [](ValueType &self, ValueType &other) -> ValueType & {
            return self + other;
          },
          nb::rv_policy::reference)
      .def(
          "__sub__",
          [](ValueType &self, ValueType &other) -> ValueType & {
            return self - other;
          },
          nb::rv_policy::reference)
      .def(
          "__mul__",
          [](ValueType &self, ValueType &other) -> ValueType & {
            return self * other;
          },
          nb::rv_policy::reference)
      .def(
          "__truediv__",
          [](ValueType &self, ValueType &other) -> ValueType & {
            return self / other;
          },
          nb::rv_policy::reference)
      .def(
          "__pow__",
          [](ValueType &self, ValueType &other) -> ValueType & {
            return self.pow(other);
          },
          nb::rv_policy::reference)
      .def("grad", &ValueType::grad)
      .def("describe", &ValueType::describe)
      .def("backward", &ValueType::backward);
}

template <typename FpType = double> class Module {
public:
  Module() = default;
  Module(Module<FpType> &other) = delete;

  void zero_grad() {
    for (auto &p : parameters_) {
      p.zero_grad();
    }
  }

private:
  std::vector<Value<FpType>> parameters_;
}; // class Module

/*
template <typename FpType = double> class Neuron : public Module<FpType> {
public:
  Neuron(std::size_t input_size) : Module<FpType>() {};
  Neuron(Neuron<FpType> &other) = delete;

private:
  std::vector<Value<FpType>> weights_;
  Value<FpType> bias;
}; // class Neuron
*/