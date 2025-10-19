#include <cmath>
#include <cstddef>
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
    for (auto &p : parameters()) {
      p.zero_grad();
    }
  }

  virtual std::vector<Value<FpType> &> parameters() = 0;

}; // class Module

template <typename FpType = double> class Neuron : public Module<FpType> {
public:
  Neuron(std::size_t input_size, bool has_activation = false)
      : Module<FpType>(), input_size_(input_size),
        has_activation_(has_activation) {
    for (int i = 0; i < input_size_; ++i) {
      weights_.emplace_back(0.01);
    }
    bias_ = Value<FpType>(0);
  };
  Neuron(Neuron<FpType> &other) = delete;

  std::vector<Value<FpType> &> parameters() override {
    std::vector<Value<FpType> &> params;
    for (auto &w : weights_) {
      params.emplace_back(w);
    }
    params.emplace_back(bias_);
    return params;
  }

  Value<FpType> &operator()(std::vector<Value<FpType> &> x) {
    Value<FpType> &sum_ = x[0];
    for (int i = 1; i < x.size(); ++i) {
      sum_ = sum_ + x[i];
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
  std::vector<Value<FpType>> weights_;
  Value<FpType> bias_;
}; // class Neuron

template <typename FpType = double> class Layer : public Module<FpType> {
public:
  Layer(std::size_t input_size, std::size_t output_size)
      : Module<FpType>(), input_size_(input_size), output_size_(output_size) {
    for (int i = 0; i < output_size; ++i) {
      neurons_.emplace_back(input_size);
    }
  };
  Layer(Layer<FpType> &other) = delete;

  std::vector<Value<FpType> &> parameters() override {
    std::vector<Value<FpType> &> params;
    for (auto &n : neurons_) {
      for (auto &p : n.parameters()) {
        params.emplace_back(p);
      }
    }
    return params;
  }

  std::vector<Value<FpType> &> operator()(std::vector<Value<FpType> &> x) {
    std::vector<Value<FpType> &> out;
    for (auto &n : neurons_) {
      out.emplace_back(n(x));
    }
  }

private:
  std::size_t input_size_;
  std::size_t output_size_;
  std::vector<Neuron<FpType>> neurons_;
}; // class Layer

template <typename FpType = double> class MLP : public Module<FpType> {
public:
  MLP(std::size_t input_size, std::vector<std::size_t> output_sizes)
      : Module<FpType>() {
    sizes_.push_back(input_size);
    for (auto sz : output_sizes) {
      sizes_.push_back(sz);
    }

    for (int i = 0; i < sizes_.size() - 1; ++i) {
      bool has_activation = (i + 1) == sizes_.size() - 1;
      layers_.emplace_back(sizes_[i], sizes_[i + 1], has_activation);
    }
  };
  MLP(Layer<FpType> &other) = delete;

  std::vector<Value<FpType> &> parameters() override {
    std::vector<Value<FpType> &> params;
    for (auto &l : layers_) {
      for (auto &p : l.parameters()) {
        params.emplace_back(p);
      }
    }
    return params;
  }

  std::vector<Value<FpType> &> operator()(std::vector<Value<FpType> &> x) {
    for (auto &l : layers_) {
      x = l(x);
    }
    return x;
  }

private:
  std::vector<std::size_t> sizes_;
  std::vector<Layer<FpType>> layers_;
}; // class MLP