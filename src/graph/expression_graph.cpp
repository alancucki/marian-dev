#include "graph/expression_graph.h"
#include <sstream>

#include "tensors/tensor_operators.h"

namespace marian {

ExpressionGraph::ExpressionGraph(bool inference, bool optimized)
    : inferenceOnly_(inference), optimized_(optimized), backend_(nullptr) {}

void ExpressionGraph::setDevice(DeviceId deviceId) {
  if(!backend_) {
    backend_ = BackendByDevice(deviceId, Config::seed);
    params_ = New<Parameters>();
    params_->init(backend_);
    tensors_ = New<Tensors>(backend_);
  }
}

Expr ExpressionGraph::dropout(float prob, const Shape& shape) {
  return Expression<ConstantNode>(
      shared_from_this(), shape, [prob, this](Tensor t) { Dropout(t, prob); });
}

void ExpressionGraph::checkNan(Tensor t) {
  ABORT_IF(throwNaN_, "Not implemented");
  // ABORT_IF(throwNaN_ && gpu::IsNan(t), "Tensor has NaN"); // XXX
  ABORT_IF(gpu::IsNan(t), "Tensor has NaN"); // XXX
}

bool ExpressionGraph::checkNan(Expr e, bool print, bool grad) {

  // // values
  // size_t totSize = shape_.elements();
  // std::vector<T> values(totSize);
  // get(values);

  auto out = e->val();
  if(grad) {
    out = e->grad();
  }
  std::cout << "POINTER: " << out << "\n";
  if(out != nullptr) {

      if(print) {
        for(int i = 0; i < out->shape().elements(); ++i) {
          std::cout << out->data()[i] << " ";
        }
      }
    if(gpu::IsNan(out)) {
      std::cerr << this << " " << out << " " << e->label() << " " << e->type() << "\n";
      if(print) {
        for(int i = 0; i < out->shape().elements(); ++i) {
          std::cout << out->data()[i] << " ";
        }
      }
      std::cout << "\n";
      return true;
    }
  }
  return false;
}
}
