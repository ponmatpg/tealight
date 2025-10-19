#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
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

template <typename FpType = double> class Module {
public:
  Module() = default;
  Module(Module<FpType> &other) = delete;

  void zero_grad() {
    for (auto &p : parameters()) {
      p->zero_grad();
    }
  }

  virtual std::vector<Value<FpType> *> parameters() = 0;

}; // class Module

template <typename FpType = double> class Neuron final : public Module<FpType> {
public:
  Neuron(std::size_t input_size, bool has_activation = false)
      : Module<FpType>(), input_size_(input_size),
        has_activation_(has_activation) {
    for (int i = 0; i < input_size_; ++i) {
      Value<FpType> *w = new Value<FpType>(0.01);
      weights_.push_back(w);
    }
    bias_ = new Value<FpType>(0);
  };
  Neuron(Neuron<FpType> &other) = delete;

  std::vector<Value<FpType> *> parameters() override {
    std::vector<Value<FpType> *> params;
    for (auto &w : weights_) {
      params.emplace_back(w);
    }
    params.emplace_back(bias_);
    return params;
  }

  Value<FpType> &operator()(std::vector<Value<FpType> *> x) {
    Value<FpType> &sum_ = *bias_;
    for (int i = 0; i < x.size(); ++i) {
      sum_ = sum_ + (*weights_[i]) * (*x[i]);
    }

    if (has_activation_) {
      return sum_.relu();
    } else {
      return sum_;
    }
  }

private:
  std::size_t input_size_;
  bool has_activation_;
  std::vector<Value<FpType> *> weights_;
  Value<FpType> *bias_{0};
}; // class Neuron

template <typename FpType = double> class Layer final : public Module<FpType> {
public:
  Layer(std::size_t input_size, std::size_t output_size, bool has_activation)
      : Module<FpType>(), input_size_(input_size), output_size_(output_size) {
    for (int i = 0; i < output_size; ++i) {
      Neuron<FpType> *n = new Neuron<FpType>(input_size, has_activation);
      neurons_.push_back(n);
    }
  };
  Layer(Layer<FpType> &other) = delete;

  std::vector<Value<FpType> *> parameters() override {
    std::vector<Value<FpType> *> params;
    for (auto &n : neurons_) {
      for (auto &p : n->parameters()) {
        params.emplace_back(p);
      }
    }
    return params;
  }

  std::vector<Value<FpType> *> operator()(std::vector<Value<FpType> *> x) {
    std::vector<Value<FpType> *> out;
    for (auto &n : neurons_) {
      out.push_back(&(*n)(x));
    }
    return out;
  }

private:
  std::size_t input_size_;
  std::size_t output_size_;
  std::vector<Neuron<FpType> *> neurons_;
}; // class Layer

template <typename FpType = double> class MLP final : public Module<FpType> {
public:
  MLP(std::size_t input_size, std::vector<std::size_t> output_sizes)
      : Module<FpType>() {
    sizes_.push_back(input_size);
    for (auto sz : output_sizes) {
      sizes_.push_back(sz);
    }

    for (int i = 0; i < sizes_.size() - 1; ++i) {
      bool has_activation = (i + 1) == sizes_.size() - 1;
      Layer<FpType> *layer =
          new Layer<FpType>(sizes_[i], sizes_[i + 1], has_activation);
      layers_.push_back(layer);
    }
  };
  MLP(MLP<FpType> &other) = delete;

  std::vector<Value<FpType> *> parameters() override {
    std::vector<Value<FpType> *> params;
    for (auto &l : layers_) {
      for (auto &p : l->parameters()) {
        params.push_back(p);
      }
    }
    return params;
  }

  std::vector<Value<FpType> *> operator()(std::vector<Value<FpType> *> x) {
    for (auto &l : layers_) {
      x = (*l)(x);
    }
    return x;
  }

private:
  std::vector<std::size_t> sizes_;
  std::vector<Layer<FpType> *> layers_;
}; // class MLP

NB_MODULE(libvalue, m) {
  m.doc() = "Differentiable scalar value";

  using FLOAT_T = double;
  using ValueType = Value<FLOAT_T>;
  using NeuronType = Neuron<FLOAT_T>;
  using LayerType = Layer<FLOAT_T>;
  using MLPType = MLP<FLOAT_T>;

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
      .def("__str__", &ValueType::describe)
      .def("__repr__", &ValueType::describe)
      .def("backward", &ValueType::backward);

  nb::class_<NeuronType>(m, "Neuron")
      .def(nb::init<int, bool>(), nb::arg("input_size"),
           nb::arg("has_activation") = false)
      .def(
          "__call__",
          [](NeuronType &self, std::vector<ValueType *> x) -> ValueType & {
            return self(x);
          },
          nb::rv_policy::reference)
      .def("zero_grad", &NeuronType::zero_grad)
      .def("parameters", &NeuronType::parameters, nb::rv_policy::reference);

  nb::class_<LayerType>(m, "Layer")
      .def(nb::init<int, int, bool>())
      .def(nb::init<int, int, bool>(), nb::arg("input_size"),
           nb::arg("output_size"), nb::arg("has_activation") = false)
      .def(
          "__call__",
          [](LayerType &self, std::vector<ValueType *> x)
              -> std::vector<ValueType *> { return self(x); },
          nb::rv_policy::reference)
      .def("zero_grad", &LayerType::zero_grad)
      .def("parameters", &LayerType::parameters, nb::rv_policy::reference);

  nb::class_<MLPType>(m, "MLP")
      .def(nb::init<std::size_t, std::vector<std::size_t>>(),
           nb::arg("input_size"), nb::arg("output_sizes"))
      .def(
          "__call__",
          [](MLPType &self, std::vector<ValueType *> x)
              -> std::vector<ValueType *> { return self(x); },
          nb::rv_policy::reference)
      .def("zero_grad", &MLPType::zero_grad)
      .def("parameters", &MLPType::parameters, nb::rv_policy::reference);
}
