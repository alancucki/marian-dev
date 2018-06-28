#include <vector>

#include <boost/filesystem.hpp>

#include "common/config.h"
#include "examples/iris/helper.cpp"
#include "marian.h"

using namespace marian;
using namespace data;
using namespace keywords;

template<typename T>
void printVec(std::vector<T>& v) {
  for(int i = 0; i < v.size(); ++i)
    std::cout << v[i] << " ";
  std::cout << std::endl;
}

// Generate pseudorandom number across frameworks
typedef long long int lli;
class RandomStream {
  private:
    lli seed_, a_, b_, m_;
  public:
    RandomStream(lli seed = 1234, lli a = 13127263419, lli b = 17131269421, lli m = 1000000)
      : seed_(seed), a_(a), b_(b), m_(m) {}

    int next() {
      seed_ = (seed_ * a_ + b_) % m_;
      return (int)seed_;
    }

    float nextUniform() {
      return next() / (float)m_ - 0.5;
    }

    std::vector<float> vec(int n) {
      std::vector<float> out;
      while(n--)
        out.push_back(nextUniform());
      return out;
    }
};

// void fill(Expr& t, RandomStream& stream) {
//   float* data = t->data();
//   for(int i = 0; i < t->shape().elements(); ++i) {
//     float v = stream.nextUniform();
//     std::cout << v << " ";
//     data[i] = v;
//   }
// }

Expr LayerStrictlyBalancedMoE(Ptr<ExpressionGraph> graph,
                              // Ptr<Options> options,
                              std::string prefix,
                              Expr input,
                              RandomStream& randStream,
                              int k,
                              int hidExperts,
                              int numExperts) {
  using namespace keywords;

  // TODO Set some pre-conditions
  // ABORT_IF(depthFfn < 1, "Filter depth {} is smaller than 1", depthFfn);

  std::string act("relu");
  int dimModel = input->shape()[-1];
  int numTokens = input->shape()[-3] * input->shape()[-2];
  int m = (int)std::ceil(1.0 * k * numTokens / numExperts);

  // (TOKENS, DIM)
  auto inputFlat = reshape(input, {numTokens, dimModel});

  // Compute the gate
  auto gateW = graph->param(
      prefix + "_GateW", {dimModel, numExperts},
      inits::from_vector(randStream.vec(dimModel * numExperts)));  // glorot_uniform);

  // (TOKENS, EXPERTS)
  auto gate = dot(inputFlat, gateW);
  // auto topk = top_k(transpose(gate), m);

  // (EXPERTS, M)
  auto tr = transpose(gate);  // BUG transpose zeroes the gradient (instead of adding zeroes)
  // auto topInds = top_k_inds(tr, m);
  // topInds->setTrainable(false);
  tr->setTrainable(false);

  // topLogits = reshape(
  //     topLogits, Shape({topLogits->shape()[-2], topLogits->shape()[-1], 1}));

  auto topMask = transpose(top_k_mask(tr, m));
  topMask->setTrainable(false);
  // auto topLogits = balanced_moe_normalize_gate_with_mask(soft, topMask, numTokens);

  // (TOKENS, EXPERTS)
  auto soft = softmax(gate);
  auto topLogits = balanced_moe_normalize_gate_with_mask(soft, topMask, m);

  // (EXPERTS, M, DIM)
  // auto sliced = balanced_moe_slicer(inputFlat, topInds);
  auto sliced = balanced_moe_slicer_with_mask(inputFlat, topMask, m);

  // (EXPERTS, M, DIM)
  auto W1 = graph->param(prefix + "_Expert__W1",
                         {numExperts, dimModel, hidExperts},
                         // inits::glorot_uniform);
                         inits::from_vector(randStream.vec(numExperts * dimModel * hidExperts)));
  auto W2 = graph->param(prefix + "_Expert__W2",
                         {numExperts, hidExperts, dimModel},
                         // inits::glorot_uniform);
                         inits::from_vector(randStream.vec(numExperts * dimModel * hidExperts)));
  auto exp1 = bdot(sliced, W1);
  auto exp2 = relu(exp1);
  inputFlat->debug(inputFlat->label());
  W1->debug(W1->label());
  //// W2->debug(W2->label());
  gateW->debug(gateW->label());
  gate->debug(gate->label());
  // topInds->debug(topInds->label());

  exp1->debug(exp1->label());
  exp2->debug(exp2->label());
  auto expOut = bdot(exp2, W2);
  auto weighted = expOut * reshape(topLogits, {numExperts, m, 1});
 
  // (TOKENS, DIM)
  // auto stitched = balanced_moe_stitcher(weighted, topInds, numTokens);
  auto stitched = balanced_moe_stitcher_with_mask(weighted, topMask);

  // (BSZ, SEQ_LEN, DIM)
  auto reshaped = reshape(stitched, input->shape());
  gate->debug("GATE" + gate->label());
  gateW->debug("GATE_W" + gateW->label());
  soft->debug("SOFTMAX" + soft->label());
  topLogits->debug("WEIGHTS" + topLogits->label());
  // topInds->debug("TOP_INDS" + topInds->label());
  topMask->debug("TOP_MASK" + topMask->label());
  expOut->debug("EXP_OUT" + expOut->label());
  weighted->debug("WEIGHTED" + weighted->label());
  sliced->debug("SLICED" + sliced->label());
  inputFlat->debug("INPUT_FLAT" + inputFlat->label());
  return reshaped;
//// 
////   gateW->debug(gateW->label());
////   inputFlat->debug(inputFlat->label());
////   W1->debug(W1->label());
////   W2->debug(W2->label());
//// 
////   exp1->debug(exp1->label());
////   exp2->debug(exp2->label());
////   gate->debug(gate->label());
////   topLogits->debug(topLogits->label());
////   topk->debug(topk->label());
////   topInds->debug(topInds->label());
////   shredded->debug(shredded->label());
////   expOut->debug(expOut->label());
//// 
////   std::cout << "M:           " << m << std::endl;
////   std::cout << "INPUT:       " << input->shape() << std::endl;
////   std::cout << "INPUT FLAT:  " << inputFlat->shape() << std::endl;
////   std::cout << "GATE:        " << gate->shape() << std::endl;
////   std::cout << "TOP INDS:    " << topInds->shape() << std::endl;
////   std::cout << "SHREDED:     " << shredded->shape() << std::endl;
////   std::cout << "EXP2:        " << exp2->shape() << std::endl;
////   std::cout << "W2:          " << W2->shape() << std::endl;
////   std::cout << "EXP_OUT:     " << expOut->shape() << std::endl;
////   std::cout << "STITCHED:    " << stitched->shape() << std::endl;
////   std::cout << "RESHAPED:    " << reshaped->shape() << std::endl;
////   return reshaped;
}

Expr testMoE4d(Ptr<ExpressionGraph> graph) {
  graph->clear();

  RandomStream randStream(1234);

  int k = 2;
  int dim = 3;
  int seqLen = 2;
  int batchSize = 3;
  int hidExperts = 7;
  int numExperts = 5;

  auto inp4d = graph->param("INPUT4D",
                            {1, batchSize, seqLen, dim},
                            // inits::uniform());
                            inits::from_vector(randStream.vec(1 * batchSize * seqLen * dim)));
  inp4d->set_name("INPUT4D");
  auto out = LayerStrictlyBalancedMoE(graph, "", inp4d, randStream, k, hidExperts, numExperts);
  auto summed = sum(sum(sum(out, {2}), {1}), {0});
  summed->debug(summed->label());
  return summed;
}

int main() {
  // Initialize global settings
  createLoggers();

  // Disable randomness by setting a fixed seed for random number generator
  // Config::seed = 123456;
  // I'm using a simple random number generator here and in Python code for reproducibility
  {
    auto graph = New<ExpressionGraph>();
    graph->setDevice({0, DeviceType::gpu});
    graph->reserveWorkspaceMB(128);

    auto out = testMoE4d(graph);
    graph->forward();
    graph->backward();
  }

  return 0;
}


/*
Expr LayerMoE(Ptr<ExpressionGraph> graph,
              // Ptr<Options> options,
              std::string prefix,
              Expr input,
              bool inference = false) {
  using namespace keywords;

  int dimModel = input->shape()[-1];

  // TODO Set some pre-conditions
  // ABORT_IF(depthFfn < 1, "Filter depth {} is smaller than 1", depthFfn);

  // TODO Get options
  // int hidExperts = options->get<int>("mixofexperts-dim-hid");
  // int numExperts = options->get<int>("mixofexperts-num-experts");
  // auto act = options->get<std::string>("mixofexperts-ffn-activation");
  int k = 2;
  int hidExperts = 7;
  int numExperts = 3;
  int numTokens = input->shape()[0] * input->shape()[1];
  std::string act("relu");

  // TODO Compute the gate
  auto gateW = graph->param(
      prefix + "_GateW", {dimModel, numExperts}, inits::glorot_uniform);
  auto gate = dot(input, gateW);

  auto topLogits = cols(reshape(gate, {numTokens, numExperts}), {0, 1});
  std::cout << "topLogits: " << topLogits->shape() << std::endl;
  std::vector<size_t> topIndsFlat = { 0, 1, 0, 2, 1, 2, 1, 1, 0, 0, 2, 2 };

  // TODO
  // auto topLogits = top_k(topIndsFlat, reshape(gate, {numTokens, numExperts}));

  std::vector<std::vector<size_t> > exp2token(numExperts);
  for(int i = 0; i < numTokens; ++i) {
    for(int e = 0; e < k; ++e) {
      size_t exp = topIndsFlat[i * k + e];
      exp2token[exp].push_back(i);
    }
  }

  auto inputFlat = reshape(input, {numTokens, dimModel});
  inputFlat->pauseForwarding = true;

  std::vector<Expr> expertOutputChunks;
  for(int e = 0; e < numExperts; ++e) {
    if(exp2token[e].size() == 0)
      continue;
    auto shard = rows(inputFlat, exp2token[e]);
    auto W1 = graph->param(prefix + "_Expert_" + std::to_string(e) + "_W1",
                           {dimModel, hidExperts},
                           inits::glorot_uniform);
    auto W2 = graph->param(prefix + "_Expert_" + std::to_string(e) + "_W2",
                           {hidExperts, dimModel},
                           inits::glorot_uniform);

    std::cout << input->shape() << std::endl;
    std::cout << inputFlat->shape() << std::endl;
    printVec(exp2token[e]);
    std::cout << shard->shape() << std::endl;

    // TODO Bias ???
    auto out = dot(relu(dot(shard, W1)), W2);
    expertOutputChunks.push_back(out);
  }
  auto expertOutput = concatenate(expertOutputChunks);
  std::cout << "expertOutput->shape() " << expertOutput->shape() << std::endl;

  // Trace back (expert, expert's output) for each token
  std::vector<std::vector<size_t> > token2flatexp(numTokens);
  size_t pos = 0;
  for(int e = 0; e < numExperts; ++e) {
    for(auto token : exp2token[e]) {
       token2flatexp[token].push_back(pos);
       pos++;
    }
  }
  for(auto v : token2flatexp) {
    for(auto elem : v)
      std::cout << elem << " ";
    std::cout << std::endl;
  }

  std::vector<Expr> finalOutputs;
  for(int i = 0; i < numTokens; ++i) {
    std::vector<size_t> experts = token2flatexp[i];
    auto outputs = rows(expertOutput, experts);
    // Get softmax weights
    auto weights = transpose(rows(topLogits, {i}));
    std::cout << "A " << outputs->shape() << " " << weights->shape() << std::endl;
    auto mult = outputs * weights;
    std::cout << "mult shape " << mult->shape() << std::endl;
    auto out = sum(mult, axis=0);
    finalOutputs.push_back(out);
  }
  // auto output = reshape(concatenate(finalOutputs));
  auto output = concatenate(finalOutputs);

  output = reshape(output, {BATCH_SIZE, SEQ_LEN, dimModel});
  std::cout << "finalOutput->shape() " << output->shape() << std::endl;
  return output;
}

Expr buildSampleBdot(Ptr<ExpressionGraph> graph) {
  graph->clear();

  // Define the hidden layer
  auto W1 = graph->param("W1", {BATCH_SIZE, M, K}, inits::uniform());
  auto W2 = graph->param("W2", {BATCH_SIZE, K, N}, inits::uniform());
  std::vector<size_t> inds = {1, 0};
  std::vector<size_t> inds2 = {1, 1};
  auto mult = bdot(W1, W2, inds, inds2);
  // auto mult = bdot(W1, W2);

  W1->debug(W1->label());
  W2->debug(W2->label());
  mult->debug(mult->label());

  return sum(mult);
}
*/
