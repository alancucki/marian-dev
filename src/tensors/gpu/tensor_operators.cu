//#include <thrust/transform_reduce.h>

#include "tensors/tensor_operators.h"

#include "functional/functional.h"
#include "functional/tensor.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/cuda_helpers.h"

#include "3rd_party/reduce_all.h"

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#define MYASSERT(condition, msg) \
  if (!(condition)) { printf(msg) ; return; }

namespace marian {

namespace gpu {

struct isnan_test {
  __host__ __device__ bool operator()(const float a) const { return isnan(a); }
};

__device__ inline float stableLogit(float x) {
  if(x >= 0) {
    float z = expf(-x);
    return 1.0 / (1.0 + z);
  } else {
    float z = expf(x);
    return z / (1.0 + z);
  }
}

bool IsNan(Tensor in) {
  // return false; // XXX

  cudaSetDevice(in->getDevice().no);
  thrust::device_ptr<float> begin = thrust::device_pointer_cast(in->data());
  thrust::device_ptr<float> end
     = thrust::device_pointer_cast(in->data() + in->size());
  return thrust::transform_reduce(
     begin, end, isnan_test(), 0, thrust::plus<bool>());
  return false;
}

void ConcatCont(Tensor out, const std::vector<Tensor>& inputs, int axis) {


  cudaSetDevice(out->getDevice().no);
  int step = 1;
  for(int i = 0; i < axis; ++i)
    step *= out->shape()[i];

  size_t offset1 = 0;
  for(int i = 0; i < step; ++i) {
    for(auto in : inputs) {
      size_t size = in->shape().elements() / step;
      size_t offset2 = i * size;

      cudaMemcpy(out->data() + offset1,
                      in->data() + offset2,
                      size * sizeof(float),
                      cudaMemcpyDeviceToDevice);

      offset1 += size;
    }
  }
  cudaStreamSynchronize(0);
}

__global__ void gInsertCols(float* out,
                            const float* in,
                            size_t rows,
                            size_t cols,
                            size_t cols_out,
                            size_t cols_in,
                            size_t offset_out,
                            size_t offset_in,
                            float beta) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols_out + offset_out;
      const float* rowIn = in + j * cols_in + offset_in;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i] + beta * rowOut[i];
      }
    }
  }
}

void Concatenate1(Tensor out, const std::vector<Tensor>& inputs) {
  cudaSetDevice(out->getDevice().no);

  int rows = out->shape().elements() / out->shape().back();

  size_t offset = 0;
  int cols_out = out->shape().back();

  for(auto in : inputs) {
    ABORT_IF(rows != in->shape().elements() / in->shape().back(),
             "First dimension must be equal");
    int cols_in = in->shape().back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols_in);

    gInsertCols<<<blocks, threads>>>(
        out->data(), in->data(), rows, cols_in, cols_out, cols_in, offset, 0, 0);
    offset += cols_in;
  }
  cudaStreamSynchronize(0);
}


__global__ void gJoin2(float* out, size_t rowBatch, size_t cols,
                       const float* in1, size_t inStride1,
                       const float* in2, size_t inStride2) {

  int outStride = inStride1 + inStride2;
  int rows = rowBatch * outStride;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      float* rowOut = out + j * cols;

      int curBatch = j / outStride;
      int curPos = j % outStride;

      int jIn1 = (curBatch * inStride1) + curPos;
      int jIn2 = (curBatch * inStride2) + curPos - inStride1;

      const float* rowIn1 = in1 + jIn1 * cols;
      const float* rowIn2 = in2 + jIn2 * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          if(curPos < inStride1)
            rowOut[i] = rowIn1[i];
          else
            rowOut[i] = rowIn2[i];
        }
      }

    }
  }
}


void Concatenate2(Tensor out, Tensor in1, Tensor in2) {
  cudaSetDevice(out->getDevice().no);

  size_t rows = out->shape().elements() / out->shape().back();
  size_t cols = out->shape().back();

  size_t rowStride1 = in1->shape()[-2];
  size_t rowStride2 = in2->shape()[-2];

  size_t rowBatch = rows / out->shape()[-2];

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);

  gJoin2<<<blocks, threads>>>(out->data(),
                              rowBatch,
                              cols,
                              in1->data(),
                              rowStride1,
                              in2->data(),
                              rowStride2);

  cudaStreamSynchronize(0);
}

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax) {
  if(ax == out->shape().size() - 1)
    Concatenate1(out, inputs);
  else if(ax == out->shape().size() - 2 && inputs.size() == 2)
    Concatenate2(out, inputs[0], inputs[1]);
  else
    ConcatCont(out, inputs, ax);
}

void Split1(std::vector<Tensor>& outputs, const Tensor in) {
  cudaSetDevice(in->getDevice().no);

  size_t offset = 0;
  int rows = in->shape().elements() / in->shape().back();
  int cols_in = in->shape().back();
  for(auto out : outputs) {
    ABORT_IF(rows != out->shape().elements() / out->shape().back(),
             "First dimension must be equal");
    int cols_out = out->shape().back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols_out);

    gInsertCols<<<blocks, threads>>>(
        out->data(), in->data(), rows, cols_out, cols_out, cols_in, 0, offset, 1);
    offset += cols_out;
  }
  cudaStreamSynchronize(0);
}

// @TODO: this function is just a temporary fix until I come up with
// something better for the situation below.
__global__ void gAddRow(float* out, const float* in, int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out[index] = in[index] + out[index];
    }
  }
}

void SplitCont(std::vector<Tensor>& outputs, const Tensor in, int axis) {
  cudaSetDevice(in->getDevice().no);

  int step = 1;
  for(int i = 0; i < axis; ++i)
    step *= in->shape()[i];

  int offset1 = 0;
  for(int i = 0; i < step; ++i) {
    for(auto out : outputs) {
      int size = out->shape().elements() / step;
      int offset2 = i * size;

      // BUG: this is does not add gradients
      //cudaMemcpyAsync(out->data() + offset2,
      //                in->data() + offset1,
      //                size * sizeof(float),
      //                cudaMemcpyDeviceToDevice);

      // @TODO: this is a quick but bad fix for the above bug
      int threads = std::min(MAX_THREADS, size);
      int blocks = std::min(MAX_BLOCKS, size / threads + (size % threads != 0));

      gAddRow<<<blocks, threads>>>(out->data() + offset2,
                                   in->data() + offset1,
                                   size);
      offset1 += size;
    }
  }
  cudaStreamSynchronize(0);
}

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax) {
  if(ax == in->shape().size() - 1)
    Split1(outputs, in);
  else
    SplitCont(outputs, in, ax);
}

__global__ void gTransposeND(
    functional::Tensor<float> out,
    const functional::Tensor<float> in,
    const functional::Array<int, functional::Shape::size()> permute,
    float beta) {
  constexpr size_t N = functional::Shape::size();
  functional::Array<int, N> oDims;
  functional::Array<int, N> pDims;

  int length = out.shape().elements();
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out.shape().dims(index, oDims);
      for(int i = 0; i < N; ++i)
        pDims[permute[i]] = oDims[i];
      out[index] = in[pDims] + beta * out[index];
    }
  }
}

__global__
void gTranspose0213(float* out, const float* in,
                    int rows,
                    int cols,
                    int stride1,
                    int stride2,
                    float beta) {

  int stride = stride1 * stride2;
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;

      int z = j / stride;
      int y = (j % stride) / stride1;
      int x = (j % stride) % stride1;
      int j2 = z * stride + x * stride2 + y;

      const float* rowIn = in + j2 * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i] + beta * rowOut[i];
      }
    }
  }

}

__global__
void gTranspose0213_beta0(float* out, const float* in,
                    int rows,
                    int cols,
                    int stride1,
                    int stride2) {

  int stride = stride1 * stride2;
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;

      int z = j / stride;
      int y = (j % stride) / stride1;
      int x = (j % stride) % stride1;
      int j2 = z * stride + x * stride2 + y;

      const float* rowIn = in + j2 * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i];
      }
    }
  }

}

__global__
void gTranspose0213_beta1(float* out, const float* in,
                    int rows,
                    int cols,
                    int stride1,
                    int stride2) {

  int stride = stride1 * stride2;
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;

      int z = j / stride;
      int y = (j % stride) / stride1;
      int x = (j % stride) % stride1;
      int j2 = z * stride + x * stride2 + y;

      const float* rowIn = in + j2 * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i] + rowOut[i];
      }
    }
  }

}

void TransposeND(Tensor out, Tensor in, const std::vector<int>& vAxis, float beta) {
  cudaSetDevice(out->getDevice().no);
  if(vAxis == std::vector<int>({0, 2, 1, 3})) {

    int rows = out->shape().elements() / out->shape().back();
    int cols = out->shape().back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols);

    int stride1 = out->shape()[-2];
    int stride2 = out->shape()[-3];

    if(fabs(beta - 1.f) < 1e-10) {
      gTranspose0213_beta1<<<blocks, threads>>>(out->data(), in->data(),
                                          rows, cols, stride1, stride2);
    } else {
      gTranspose0213_beta0<<<blocks, threads>>>(out->data(), in->data(),
                                          rows, cols, stride1, stride2);
    }
  }
  else {

    functional::Array<int, functional::Shape::size()> axes;
    int diff = functional::Shape::size() - vAxis.size();
    for(int i = 0; i < axes.size(); ++i)
      if(i < diff)
        axes[i] = i;
      else
        axes[i] = vAxis[i - diff] + diff;

    int length = out->shape().elements();
    int threads = std::min(MAX_THREADS, length);
    int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    gTransposeND<<<blocks, threads>>>(out, in, axes, beta);
  }
  if(IsNan(in))
    std::cout << in << " NaNs or infs spotted in transpose(in)\n";
  if(IsNan(out))
    std::cout << in << " NaNs or infs spotted in transpose(out)\n";
}

__global__ void gSoftmax(float* out,
                         functional::Shape outShape,
                         const float* in,
                         const float* mask,
                         const functional::Shape maskShape) {
  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

  bool broadcast = outShape != maskShape;
  functional::Array<int, functional::Shape::size()> dims;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;

      extern __shared__ float _share[];

      float* _max = _share + blockDim.x;
      _max[threadIdx.x] = -CUDA_FLT_MAX;  // mask
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float mVal = 1.f;
          if(mask) {
            int mIndex = id + j * cols;
            if(broadcast) {
              outShape.dims(mIndex, dims);
              mIndex = maskShape.bindex(dims);
            }
            mVal = mask[mIndex];
          }

          if(mVal && sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float max = _max[0];
      __syncthreads();

      float* _sum = _share + blockDim.x;

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float mVal = 1.f;
          if(mask) {
            int mIndex = id + j * cols;
            if(broadcast) {
              outShape.dims(mIndex, dims);
              mIndex = maskShape.bindex(dims);
            }
            mVal = mask[mIndex];
          }

          float ex = 0;
          if(mVal)
            ex = __expf(sp[id] - max);
          so[id] = ex;

          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          so[id] = so[id] / _sum[0];
        }
      }
    }
  }
}

void Softmax(Tensor out, Tensor in, Tensor mask) {
  cudaSetDevice(out->getDevice().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t k = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(float) * threads * 2;

  if(mask)
    gSoftmax<<<blocks, threads, shared>>>(
        out->data(), out->shape(), in->data(), mask->data(), mask->shape());
  else
    gSoftmax<<<blocks, threads, shared>>>(
        out->data(), out->shape(), in->data(), 0, out->shape());
}

__global__ void gLogSoftmax(float* out,
                            const functional::Shape outShape,
                            const float* in) {
  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;

      extern __shared__ float _share[];

      float* _max = _share + blockDim.x;
      _max[threadIdx.x] = sp[threadIdx.x];
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          if(sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float max = _max[0];
      __syncthreads();

      float* _sum = _share + blockDim.x;

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float sm = sp[id] - max;
          float ex = __expf(sm);
          so[id] = sm;
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols)
          so[id] -= __logf(_sum[0]);
      }
    }
  }
}

void LogSoftmax(Tensor out, Tensor in) {
  cudaSetDevice(out->getDevice().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t k = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(float) * threads * 2;

  gLogSoftmax<<<blocks, threads, shared>>>(
      out->data(), out->shape(), in->data());
}

__global__ void gMax(float* out,
                     const functional::Shape inShape,
                     const float* in) {
  int rows = inShape.elements() / inShape.back();
  int cols = inShape.back();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* sp = in + j * cols;

      extern __shared__ float _share[];
      // int* maxInd = (float*)_share[blockDim.x];

      float* _max = _share; // + blockDim.x;
      _max[threadIdx.x] = sp[threadIdx.x];
      // maxInd[threadIdx.x] = threadIdx.x;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          if(sp[id] > _max[threadIdx.x]) {
            _max[threadIdx.x] = sp[id];
            // maxInd[threadIdx.x] = id;
          }
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
            // maxInd[threadIdx.x] = maxInd[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      out[j] = _max[0];
      __syncthreads();
    }
  }
}

void Max(Tensor out, Tensor in) {
  cudaSetDevice(out->getDevice().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t k = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  // int shared = sizeof(float) * threads * 2;
  int shared = sizeof(float) * threads + sizeof(int) * threads;

  gMax<<<blocks, threads, shared>>>(
      out->data(), in->shape(), in->data());
}

// blocks compute rows, thread compute columns within each row
// softmax is applied to each row separately
__global__ void gMaxGrad(float* grad,
                         const float* adj,
                         const float* val, // Assume val to be a 1-D array
                         const float* in,
                         const int rows,
                         const int cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = grad + j * cols;
      const float* sp = in + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          so[id] = (sp[id] == val[j] ? adj[j] * val[j] : 0.0);
        }
      }
      __syncthreads();
    }
  }
}

void MaxGrad(Tensor grad, Tensor adj, Tensor val, Tensor in) {
  cudaSetDevice(adj->getDevice().no);

  // grad is m-by-k, val is 1-by-k matrix
  int m = grad->shape().elements() / grad->shape().back();
  int k = grad->shape().back();

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  gMaxGrad<<<blocks, threads>>>(
      grad->data(), adj->data(), val->data(), in->data(), m, k);
}

__device__ inline bool isIn(const int32_t val, const int32_t* arr, const int32_t len) {
  for(auto ptr = arr; ptr < arr + len; ++ptr) {
    if(*ptr == val)
      return true;
  }
  return false;
}

__global__ void gTopK(float* out,
                      const int32_t numK,
                      const functional::Shape inShape,
                      const float* in,
                      const bool returnInds = false) {
  int rows = inShape.elements() / inShape.back();
  int cols = inShape.back();

  for(int k = 0; k < numK; ++k) {

    for(int bid = 0; bid < rows; bid += gridDim.x) {
      int j = bid + blockIdx.x;
      if(j < rows) {
        const float* sp = in + j * cols;

        extern __shared__ float _share[];
        int32_t* maxInd = (int32_t*)&_share[blockDim.x];
        int32_t* outInds = (int32_t*)&_share[blockDim.x * 2];

        // TODO Why was it shifted by blockDim.x ?
        float* _max = _share; // + blockDim.x;
        _max[threadIdx.x] = (isIn((int32_t)threadIdx.x, outInds, k) ? -1e8 : sp[threadIdx.x]);
        maxInd[threadIdx.x] = threadIdx.x;
        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols) {
            if(!isIn((int32_t)id, outInds, k) && sp[id] > _max[threadIdx.x]) {
              _max[threadIdx.x] = sp[id];
              maxInd[threadIdx.x] = id;
            }
          }
        }
        __syncthreads();
        int len = blockDim.x;
        while(len != 1) {
          __syncthreads();
          int skip = (len + 1) >> 1;
          if(threadIdx.x < (len >> 1)) {
            if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
              _max[threadIdx.x] = _max[threadIdx.x + skip];
              maxInd[threadIdx.x] = maxInd[threadIdx.x + skip];
            }
          }
          len = (len + 1) >> 1;
        }
        __syncthreads();
        out[k + j * numK] = (returnInds ? (float)maxInd[0] : _max[0]);
        outInds[k] = maxInd[0];
        __syncthreads();
      }
    }
  }
}

__global__ void gTopKMask(float* out,
                          const int32_t numK,
                          const functional::Shape inShape,
                          const float* in) {
  int rows = inShape.elements() / inShape.back();
  int cols = inShape.back();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      // Zero the output row
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols)
          out[j * cols + id] = -1.0;
      }

      for(int k = 0; k < numK; ++k) {
        const float* sp = in + j * cols;

        extern __shared__ float _share[];
        int32_t* maxInd = (int32_t*)&_share[blockDim.x];
        int32_t* outInds = (int32_t*)&_share[blockDim.x * 2];

        // TODO Why was it shifted by blockDim.x ?
        float* _max = _share; // + blockDim.x;
        _max[threadIdx.x] = (isIn((int32_t)threadIdx.x, outInds, k) ? -1e8 : sp[threadIdx.x]);
        maxInd[threadIdx.x] = threadIdx.x;
        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols) {
            if(!isIn((int32_t)id, outInds, k) && sp[id] > _max[threadIdx.x]) {
              _max[threadIdx.x] = sp[id];
              maxInd[threadIdx.x] = id;
            }
          }
        }
        __syncthreads();
        int len = blockDim.x;
        while(len != 1) {
          __syncthreads();
          int skip = (len + 1) >> 1;
          if(threadIdx.x < (len >> 1)) {
            if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
              _max[threadIdx.x] = _max[threadIdx.x + skip];
              maxInd[threadIdx.x] = maxInd[threadIdx.x + skip];
            }
          }
          len = (len + 1) >> 1;
        }
        __syncthreads();
        out[maxInd[0] + j * cols] = (float)k;
        outInds[k] = maxInd[0];
        __syncthreads();
      }
    }
  }
}

void TopK(Tensor out, const size_t k, Tensor in) {
  cudaSetDevice(out->getDevice().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t n = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)n);
  // int shared = sizeof(float) * threads * 2;
  int shared = (sizeof(float) * threads + sizeof(int32_t) * (threads + k)) * 2;

  gTopK<<<blocks, threads, shared>>>(
      out->data(), k, in->shape(), in->data(), false);
}

void TopKInds(Tensor out, const size_t k, Tensor in) {
  cudaSetDevice(out->getDevice().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t n = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)n);
  // int shared = sizeof(float) * threads * 2;
  int shared = (sizeof(float) * threads + sizeof(int32_t) * (threads + k)) * 2;

  gTopK<<<blocks, threads, shared>>>(
      out->data(), k, in->shape(), in->data(), true);
}

void TopKMask(Tensor out, const size_t k, Tensor in) {
  cudaSetDevice(out->getDevice().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t n = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)n);
  int shared = (sizeof(float) * threads + sizeof(int32_t) * (threads + k)) * 2;

  gTopKMask<<<blocks, threads, shared>>>(
      out->data(), k, in->shape(), in->data());

  if(IsNan(out))
    std::cout << "NaNs or infs spotted in TopKMask\n";
}

// blocks compute rows, threads compute columns within each row
// softmax is applied to each row separately
__global__ void gTopKGrad(float* grad,
                          const float* adj,
                          const float* val, // num x k
                          const float* in,
                          const int rows,
                          const int cols,
                          const size_t numK) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = grad + j * cols;
      const float* sp = in + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          so[id] = 0.0;
          for(int k = numK - 1; k >= 0; --k) {
            if(sp[id] >= val[j * numK + k])
              so[id] += adj[j * numK + k] * val[j * numK + k];
            else
              break;
          }
        }
        __syncthreads();
      }
    }
    __syncthreads();
  }
}

void TopKGrad(Tensor grad, Tensor adj, Tensor val, Tensor in) {
  cudaSetDevice(adj->getDevice().no);

  // grad is m-by-k, val is 1-by-k matrix
  int rows = grad->shape().elements() / grad->shape().back();
  int cols = grad->shape().back();
  int k = val->shape()[-1];

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);
  gTopKGrad<<<blocks, threads>>>(
      grad->data(), adj->data(), val->data(), in->data(), rows, cols, k);
}

// blocks compute rows, threads compute columns within each row
// softmax is applied to each row separately
__global__ void gTopKIndsGrad(float* grad,
                              const int rows,
                              const int cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = grad + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          // so[id] = 1.0;
          }
        __syncthreads();
      }
    }
    __syncthreads();
  }
}

void TopKIndsGrad(Tensor grad, Tensor adj) {
  cudaSetDevice(grad->getDevice().no);

  // grad is m-by-k, val is 1-by-k matrix
  int rows = grad->shape().elements() / grad->shape().back();
  int cols = grad->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);
  gTopKIndsGrad<<<blocks, threads>>>(
      grad->data(), rows, cols);
}

///////////////////////////////////////////////////////

__global__ void gSoftmaxGrad(float* grad,
                             const float* adj,
                             const float* val,
                             const int rows,
                             const int cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;

      float* gradRow = grad + j * cols;
      const float* adjRow = adj + j * cols;
      const float* valRow = val + j * cols;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += valRow[id] * adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float val = valRow[id] * (adjRow[id] - _sum[0]);
          if(val)
            gradRow[id] += val;
        }
      }
    }
  }
}

void SoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
  cudaSetDevice(adj->getDevice().no);
  // grad and val are both m-by-k matrices, passed as input.
  // A weighted average of each row of grad (according to the weights
  // specified in val) is computed and subtracted from Out.
  // adj is multiplied for each element to get backward step in autodiff
  int m = grad->shape().elements() / grad->shape().back();
  int k = grad->shape().back();

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads * 2;
  gSoftmaxGrad<<<blocks, threads, shared>>>(
      grad->data(), adj->data(), val->data(), m, k);
}

__global__ void gLogSoftmaxGrad(float* grad,
                                const float* adj,
                                const float* val,
                                const int rows,
                                const int cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;

      float* gradRow = grad + j * cols;
      const float* adjRow = adj + j * cols;
      const float* valRow = val + j * cols;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols)
          gradRow[id] += adjRow[id] - (expf(valRow[id]) * _sum[0]);
      }
    }
  }
}

void LogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
  cudaSetDevice(adj->getDevice().no);

  // grad and val are both m-by-k matrices, passed as input.
  // A weighted average of each row of grad (according to the weights
  // specified in val) is computed and subtracted from Out.
  // adj is multiplied for each element to get backward step in autodiff
  int m = grad->shape().elements() / grad->shape().back();
  int k = grad->shape().back();

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads * 2;
  gLogSoftmaxGrad<<<blocks, threads, shared>>>(
      grad->data(), adj->data(), val->data(), m, k);
}

///////////////////////////////////////////////////////
__global__ void gArgmax(float* out,
                        const float* data,
                        size_t rows,
                        size_t cols) {
  size_t row = blockIdx.x;
  size_t startInd = row * cols;
  float maxScore = -99999;
  size_t maxInd;
  for(size_t col = 0; col < cols; ++col) {
    size_t ind = startInd + col;
    float score = data[ind];
    if(score > maxScore) {
      maxScore = score;
      maxInd = col;
    }
  }
  out[row] = maxInd;
}

///////////////////////////////////////////////////////

__global__ void gCopyRows(float* out,
                          const float* in,
                          size_t cols,
                          const size_t* sourceRowIdx,
                          size_t rows) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      size_t dstId = j;
      size_t srcId = sourceRowIdx[j];

      float* rowOut = out + dstId * cols;
      const float* rowIn = in + srcId * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i];
      }
    }
  }
}

void CopyRows(Tensor out, const Tensor in, const std::vector<size_t>& indices) {
  cudaSetDevice(out->getDevice().no);

  size_t cols = in->shape().back();
  size_t rowsToCopy = indices.size();

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks = std::min(MAX_BLOCKS, (int)rowsToCopy);

  size_t* d_indices;
  CUDA_CHECK(cudaMalloc(&d_indices, rowsToCopy * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_indices,
                        indices.data(),
                        rowsToCopy * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  gCopyRows<<<blocks, threads>>>(
      out->data(), in->data(), cols, d_indices, rowsToCopy);

  CUDA_CHECK(cudaFree(d_indices));
}

__global__ void gPasteRows(float* out,
                           const float* in,
                           size_t cols,
                           const size_t* targetRowIdx,
                           size_t rows) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      size_t dstId = targetRowIdx[j];
      size_t srcId = j;

      float* rowOut = out + dstId * cols;
      const float* rowIn = in + srcId * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          atomicAdd(rowOut + i, rowIn[i]);
      }
    }
  }
}

void PasteRows(Tensor out,
               const Tensor in,
               const std::vector<size_t>& indices) {
  cudaSetDevice(out->getDevice().no);

  size_t cols = in->shape().back();
  size_t rowsToCopy = indices.size();

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks = std::min(MAX_BLOCKS, (int)rowsToCopy);

  // @TODO: turn into tensor
  size_t* d_indices;
  CUDA_CHECK(cudaMalloc(&d_indices, rowsToCopy * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_indices,
                        indices.data(),
                        rowsToCopy * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  gPasteRows<<<blocks, threads>>>(
      out->data(), in->data(), cols, d_indices, rowsToCopy);
  CUDA_CHECK(cudaFree(d_indices));
}

/////////////

__global__ void gCopyCols(float* out,
                          const float* in,
                          size_t rows,
                          size_t colsIn,
                          const size_t* sourceColIdx,
                          size_t colsOut) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* rowIn = in + j * colsIn;
      float* rowOut = out + j * colsOut;

      for(int tid = 0; tid < colsOut; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < colsOut)
          rowOut[i] = rowIn[sourceColIdx[i]];
      }
    }
  }
}

void CopyCols(Tensor out, const Tensor in, const std::vector<size_t>& indices) {
  cudaSetDevice(out->getDevice().no);

  size_t rows = in->shape().elements() / in->shape().back();
  size_t cols = in->shape().back();

  size_t colsToCopy = indices.size();

  int threads = std::min(MAX_THREADS, (int)colsToCopy);
  int blocks = std::min(MAX_BLOCKS, (int)rows);

  size_t* d_indices;
  CUDA_CHECK(cudaMalloc(&d_indices, colsToCopy * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_indices,
                        indices.data(),
                        colsToCopy * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  gCopyCols<<<blocks, threads>>>(
      out->data(), in->data(), rows, cols, d_indices, colsToCopy);

  CUDA_CHECK(cudaFree(d_indices));
}

__global__ void gPasteCols(float* out,
                           const float* in,
                           size_t rows,
                           size_t colsOut,
                           const size_t* targetColIdx,
                           size_t colsIn) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* rowIn = in + j * colsIn;
      float* rowOut = out + j * colsOut;

      for(int tid = 0; tid < colsIn; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < colsIn)
          rowOut[targetColIdx[i]] += rowIn[i];
      }
    }
  }
}

void PasteCols(Tensor out,
               const Tensor in,
               const std::vector<size_t>& indices) {
  cudaSetDevice(out->getDevice().no);

  size_t rows = in->shape().elements() / in->shape().back();
  size_t cols = in->shape().back();

  size_t colsToCopy = indices.size();

  int threads = std::min(MAX_THREADS, (int)colsToCopy);
  int blocks = std::min(MAX_BLOCKS, (int)rows);

  size_t* d_indices;
  CUDA_CHECK(cudaMalloc(&d_indices, colsToCopy * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_indices,
                        indices.data(),
                        colsToCopy * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  gPasteCols<<<blocks, threads>>>(
      out->data(), in->data(), rows, cols, d_indices, colsToCopy);

  CUDA_CHECK(cudaFree(d_indices));
}

__global__ void gSelect(float* out,
                        functional::Shape outShape,
                        const float* in,
                        const functional::Shape inShape,
                        int axis,
                        size_t* d_indices) {
  int length = outShape.elements();
  functional::Array<int, functional::Shape::size()> dims;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      outShape.dims(index, dims);
      dims[axis] = d_indices[dims[axis]];
      int inIndex = inShape.index(dims);
      out[index] = in[inIndex];
    }
  }
}

__global__ void gInsert(float* out,
                        functional::Shape outShape,
                        const float* in,
                        const functional::Shape inShape,
                        int axis,
                        size_t* d_indices) {
  int length = inShape.elements();
  functional::Array<int, functional::Shape::size()> dims;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      inShape.dims(index, dims);
      dims[axis] = d_indices[dims[index]];
      int outIndex = outShape.index(dims);
      out[outIndex] += in[index];
    }
  }
}

void Select(Tensor out,
            const Tensor in,
            int axis,
            const std::vector<size_t>& indices,
            Ptr<Allocator> allocator) {
  cudaSetDevice(out->getDevice().no);

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  auto mp_indices = allocator->alloc<size_t>(indices.size());
  CudaCopy(indices.data(),
           indices.data() + indices.size(),
           mp_indices->data<size_t>());

  int axisGPU = axis + functional::Shape::size() - out->shape().size();
  gSelect<<<blocks, threads>>>(out->data(),
                               out->shape(),
                               in->data(),
                               in->shape(),
                               axisGPU,
                               mp_indices->data<size_t>());

  allocator->free(mp_indices);
}

void Insert(Tensor out,
            const Tensor in,
            int axis,
            const std::vector<size_t>& indices,
            Ptr<Allocator> allocator) {
  cudaSetDevice(in->getDevice().no);

  int length = in->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  auto mp_indices = allocator->alloc<size_t>(indices.size());
  CudaCopy(indices.data(),
           indices.data() + indices.size(),
           mp_indices->data<size_t>());

  int axisGPU = axis + functional::Shape::size() - out->shape().size();
  gInsert<<<blocks, threads>>>(out->data(),
                               out->shape(),
                               in->data(),
                               in->shape(),
                               axisGPU,
                               mp_indices->data<size_t>());

  allocator->free(mp_indices);
}

__global__ void gGRUFastForward(float* out,
                                const float* state,
                                const float* xW,
                                const float* sU,
                                const float* b,
                                const float* mask,
                                size_t rows,
                                size_t cols,
                                bool final) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];
      float* rowOut = out + j * cols;
      const float* rowState = state + j * cols;

      const float* xWrow = xW + j * cols * 3;
      const float* sUrow = sU + j * cols * 3;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          float r = stableLogit(xWrow[i] + sUrow[i] + b[i]);

          int k = i + cols;

          float z = stableLogit(xWrow[k] + sUrow[k] + b[k]);

          int l = i + 2 * cols;
          float h;
          if(final)
            h = tanhf(xWrow[l] + (sUrow[l] + b[l]) * r);
          else
            h = tanhf(xWrow[l] + sUrow[l] * r + b[l]);

          float out = (1.0f - z) * h + z * rowState[i];
          rowOut[i] = m * out + (1 - m) * rowState[i];
        }
      }
    }
  }
}

void GRUFastForward(Tensor out, std::vector<Tensor> inputs, bool final) {
  cudaSetDevice(out->getDevice().no);

  int rows = out->shape().elements() / out->shape().back();
  int cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gGRUFastForward<<<blocks, threads>>>(
      out->data(),                                // output
      inputs[0]->data(),                          // state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      inputs.size() > 4 ? inputs[4]->data() : 0,  // mask
      rows,
      cols,
      final);
}

__global__ void gGRUFastBackward(float* outState,
                                 float* outXW,
                                 float* outSU,
                                 float* outB,
                                 const float* state,
                                 const float* xW,
                                 const float* sU,
                                 const float* b,
                                 const float* mask,
                                 const float* adj,
                                 size_t rows,
                                 size_t cols,
                                 bool final) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];

      float* rowOutState = outState + j * cols;
      float* rowOutXW = outXW + j * cols * 3;
      float* rowOutSU = outSU + j * cols * 3;

      const float* rowState = state + j * cols;
      const float* rowXW = xW + j * cols * 3;
      const float* rowSU = sU + j * cols * 3;
      const float* rowAdj = adj + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          int k = i + cols;
          int l = i + 2 * cols;

          float r = stableLogit(rowXW[i] + rowSU[i] + b[i]);
          float z = stableLogit(rowXW[k] + rowSU[k] + b[k]);

          float h;
          if(final)
            h = tanhf(rowXW[l] + (rowSU[l] + b[l]) * r);
          else
            h = tanhf(rowXW[l] + rowSU[l] * r + b[l]);

          float adj = rowAdj[i];

          float t = (1 - z) * (1 - h * h);

          // df/ds
          if(outState)
            rowOutState[i] += (m * z - m + 1) * adj;

          // df/d(xW_r) ...
          float dfdxW_r = m * r * (1 - r) * t * adj;
          if(final)
            dfdxW_r *= rowSU[l] + b[l];
          else
            dfdxW_r *= rowSU[l];
          if(outXW)
            rowOutXW[i] += dfdxW_r;
          if(outSU)
            rowOutSU[i] += dfdxW_r;
          if(outB)
            atomicAdd(outB + i, dfdxW_r);

          // df/d(xW_z) ...
          float dfdxW_z = m * (1 - z) * z * (rowState[i] - h) * adj;
          if(outXW)
            rowOutXW[k] += dfdxW_z;
          if(outSU)
            rowOutSU[k] += dfdxW_z;
          if(outB)
            atomicAdd(outB + k, dfdxW_z);

          // df/d(xW_x) ...
          float dfdxW_x = m * t * adj;
          if(outXW)
            rowOutXW[l] += dfdxW_x;
          if(outSU)
            rowOutSU[l] += dfdxW_x * r;
          if(outB)
            if(final)
              atomicAdd(outB + l, dfdxW_x * r);
            else
              atomicAdd(outB + l, dfdxW_x);
        }
      }
    }
  }
}

void GRUFastBackward(std::vector<Tensor> outputs,
                     std::vector<Tensor> inputs,
                     Tensor adj,
                     bool final) {
  cudaSetDevice(adj->getDevice().no);

  int rows = adj->shape().elements() / adj->shape().back();
  int cols = adj->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gGRUFastBackward<<<blocks, threads>>>(
      outputs[0] ? outputs[0]->data() : 0,        // state - adj
      outputs[1] ? outputs[1]->data() : 0,        // xW - adj
      outputs[2] ? outputs[2]->data() : 0,        // sU - adj
      outputs[3] ? outputs[3]->data() : 0,        // b - adj
      inputs[0]->data(),                          // state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      inputs.size() > 4 ? inputs[4]->data() : 0,  // mask
      adj->data(),
      rows,
      cols,
      final);
}

__global__ void gCrossEntropyPick(float* out,
                                  const functional::Shape outShape,
                                  const float* in,
                                  const functional::Shape inShape,
                                  const float* pick) {
  int rows = inShape.elements() / inShape.back();
  int cols = inShape.back();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* sp = in + j * cols;

      extern __shared__ float _share[];
      float* _max = _share + blockDim.x;

      _max[threadIdx.x] = sp[threadIdx.x];
      for(int tid = 1; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          if(sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float max = _max[0];
      __syncthreads();

      float* _sum = _share + blockDim.x;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += __expf(sp[id] - max);
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();

      // cross-entropy
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id == (int)pick[j]) {
          out[j] = __logf(_sum[0]) - sp[id] + max;
        }
      }
    }
  }
}

void CrossEntropyPick(Tensor out, Tensor in, Tensor pick) {
  cudaSetDevice(out->getDevice().no);

  int rows = in->shape().elements() / in->shape().back();
  int cols = in->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  int shared = sizeof(float) * threads * 2;

  gCrossEntropyPick<<<blocks, threads, shared>>>(
      out->data(), out->shape(), in->data(), in->shape(), pick->data());
}

__global__ void gCrossEntropyPickBackward(float* out,
                                          const functional::Shape outShape,
                                          const float* adj,
                                          const float* in,
                                          const float* pick) {
  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* sp = in + j * cols;
      float* so = out + j * cols;

      extern __shared__ float _share[];
      float* _max = _share + blockDim.x;

      _max[threadIdx.x] = sp[threadIdx.x];
      for(int tid = 1; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          if(sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float max = _max[0];
      __syncthreads();

      float* _sum = _share + blockDim.x;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = __expf(sp[id] - max);
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();

      // cross-entropy
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float sub = (float)(id == (int)pick[j]);
          so[id] += adj[j] * (__expf(sp[id] - max) / _sum[0] - sub);
        }
      }
    }
  }
}

void CrossEntropyPickBackward(Tensor out, Tensor adj, Tensor a, Tensor pick) {
  cudaSetDevice(out->getDevice().no);

  int rows = out->shape().elements() / out->shape().back();
  int cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  int shared = sizeof(float) * threads * 2;

  gCrossEntropyPickBackward<<<blocks, threads, shared>>>(
      out->data(), out->shape(), adj->data(), a->data(), pick->data());
}

float L2Norm(Tensor in) {
  cudaSetDevice(in->getDevice().no);

  int size = in->shape().elements();
  int threads = std::min(MAX_THREADS, size);
  int blocks = std::min(MAX_BLOCKS, size / threads + (size % threads != 0));

  uint8_t* data;
  cudaMalloc(&data, blocks * sizeof(float));
  Tensor out(new TensorBase(New<MemoryPiece>(data, blocks * sizeof(float)),
                            {1, blocks},
                            in->getBackend()));

  using namespace functional;
  ReduceAll(_1 * _1, out, in);
  float dataCpu = sqrtf(out->get(0));
  out.reset();
  cudaFree(data);
  return dataCpu;
}

__global__ void gAtt(float* out,
                     const float* va,
                     const float* ctx,
                     const float* state,
                     int m,  // total rows (batch x time x beam)
                     int k,  // depth
                     int b,  // batch size
                     int t   // time of ctx
                     ) {
  int rows = m;
  int cols = k;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* vaRow = va;
      const float* ctxRow = ctx + (j % (b * t)) * cols;
      const float* stateRow = state + ((j / (b * t)) * b + j % b) * cols;

      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float z = ctxRow[id] + stateRow[id];
          float ex = tanhf(z) * vaRow[id];
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      out[j] = _sum[0];
      __syncthreads();
    }
  }
}

void Att(Tensor out, Tensor va, Tensor context, Tensor state) {
  cudaSetDevice(out->getDevice().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t k = context->shape()[-1];
  size_t b = context->shape()[-2];
  size_t t = context->shape()[-3];

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(float) * threads * 2;

  gAtt<<<blocks, threads, shared>>>(
      out->data(), va->data(), context->data(), state->data(), m, k, b, t);
}

__global__ void gAttBack(float* gVa,
                         float* gContext,
                         float* gState,
                         const float* va,
                         const float* context,
                         const float* state,
                         const float* adj,
                         int m,  // rows
                         int k,  // cols
                         int n   // batch size
                         ) {
  int rows = m;
  int cols = k;
  for(int bid = 0; bid < m; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* gcRow = gContext + j * cols;
      float* gsRow = gState + (j % n) * cols;

      const float* cRow = context + j * cols;
      const float* sRow = state + (j % n) * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float z = cRow[id] + sRow[id];

          float t = tanhf(z);
          float r = va[id] * (1.f - t * t);

          gcRow[id] += r * adj[j];
          gsRow[id] += r * adj[j];
          atomicAdd(gVa + id, t * adj[j]);
        }
      }
    }
  }
}

void AttBack(Tensor gVa,
             Tensor gContext,
             Tensor gState,
             Tensor va,
             Tensor context,
             Tensor state,
             Tensor adj) {
  cudaSetDevice(adj->getDevice().no);

  size_t m = adj->shape().elements() / adj->shape()[-1];
  size_t k = context->shape()[-1];
  size_t n = context->shape()[-2];

  int blocks = std::min(MAX_BLOCKS, (int)n);
  int threads = std::min(MAX_THREADS, (int)k);

  gAttBack<<<blocks, threads>>>(gVa->data(),
                                gContext->data(),
                                gState->data(),

                                va->data(),
                                context->data(),
                                state->data(),

                                adj->data(),
                                m,
                                k,
                                n);
}

__global__ void gLNormalization(float* out,
                                const float* in,
                                const float* alpha,
                                const float* beta,
                                int rows,
                                int cols,
                                float eps = 1e-9) {
  extern __shared__ float _share[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;

      float* _sum = _share + blockDim.x;
      _sum[threadIdx.x] = 0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = _sum[0] / cols;
      __syncthreads();

      float* _sqSum = _share + blockDim.x;

      _sqSum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = sp[id] - mean;
          _sqSum[threadIdx.x] += ex * ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sqSum[threadIdx.x] += _sqSum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (_sqSum[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float t = alpha[id] * ((sp[id] - mean) / sigma);
          if(beta != nullptr)
            t += beta[id];
          so[id] = t;
        }
      }
    }
  }
}

void LayerNormalization(Tensor out,
                        Tensor in,
                        Tensor gamma,
                        Tensor beta,
                        float eps) {
  cudaSetDevice(out->getDevice().no);

  int rows = in->shape().elements() / in->shape().back();
  int cols = in->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  int shared = 2 * threads * sizeof(float);

  gLNormalization<<<blocks, threads, shared>>>(out->data(),
                                               in->data(),
                                               gamma->data(),
                                               beta ? beta->data() : nullptr,
                                               rows,
                                               cols,
                                               eps);
}

__global__ void gLayerNormalizationGrad(float* gradX,
                                        float* gradGamma,
                                        float* gradBeta,
                                        float* adj,
                                        float* y,
                                        float* x,
                                        float* gamma,
                                        float* beta,
                                        int rows,
                                        int cols,
                                        float eps = 1e-9) {
  extern __shared__ float shared[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* sum_adj = shared;
      float* sum_adj_x = shared + blockDim.x;
      float* sum_x = shared + 2 * blockDim.x;
      float* sum_sqr = shared + 3 * blockDim.x;

      const float* xRow = x + j * cols;
      const float* yRow = y + j * cols;
      const float* adjRow = adj + j * cols;
      float* gradXRow = gradX + j * cols;

      sum_x[threadIdx.x] = 0.0f;
      sum_adj[threadIdx.x] = 0.0f;
      sum_adj_x[threadIdx.x] = 0.0f;
      sum_sqr[threadIdx.x] = 0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          sum_x[threadIdx.x] += xRow[id];
          sum_adj_x[threadIdx.x]
              += adjRow[id] * (yRow[id] - ((beta) ? beta[id] : 0)) / gamma[id];
          sum_adj[threadIdx.x] += adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          sum_x[threadIdx.x] += sum_x[threadIdx.x + skip];
          sum_adj[threadIdx.x] += sum_adj[threadIdx.x + skip];
          sum_adj_x[threadIdx.x] += sum_adj_x[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = sum_x[0] / cols;
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = xRow[id] - mean;
          sum_sqr[threadIdx.x] += ex * ex;
        }
      }

      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          sum_sqr[threadIdx.x] += sum_sqr[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (sum_sqr[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float grad_x = 0.0f;
          float x_hat = (yRow[id] - ((beta) ? beta[id] : 0)) / gamma[id];
          grad_x += cols * adjRow[id];
          grad_x -= sum_adj[0];
          grad_x -= sum_adj_x[0] * x_hat;
          grad_x /= (cols * sigma);

          float valX = gamma[id] * grad_x;
          float sign = (0.f < valX) - (valX < 0.f);
          valX = fabs(valX) > 1000 ? sign * 1000 : valX;

          gradXRow[id] += valX;
          atomicAdd(gradGamma + id, adjRow[id] * x_hat);
          if(beta) {
            atomicAdd(gradBeta + id, adjRow[id]);
          }
        }
      }
    }
  }
}

void LayerNormalizationGrad(Tensor gradX,
                            Tensor gradGamma,
                            Tensor gradBeta,
                            Tensor adj,
                            Tensor y,
                            Tensor x,
                            Tensor gamma,
                            Tensor beta,
                            float eps) {
  cudaSetDevice(adj->getDevice().no);
  int rows = y->shape().elements() / y->shape()[-1];
  int cols = y->shape()[-1];

  int threads = std::min(MAX_THREADS, cols);
  int blocks = std::min(MAX_BLOCKS, rows);
  int shared = sizeof(float) * threads * 4;

  gLayerNormalizationGrad<<<blocks, threads, shared>>>(
      gradX->data(),
      gradGamma->data(),
      (gradBeta) ? gradBeta->data() : nullptr,
      adj->data(),
      y->data(),
      x->data(),
      gamma->data(),
      (beta) ? beta->data() : nullptr,
      rows,
      cols,
      eps);
}

__global__ void gShift(float* out, const float* in, int length, int offset, float beta) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      if(index - offset < 0 || index - offset >= length)
        out[index] = 0;
      else
        out[index] = in[index - offset] + beta * out[index];
    }
  }
}

void Shift(Tensor out, Tensor in, marian::Shape shift, bool invert, float beta) {
  ABORT_IF(in->shape().size() != shift.size(), "bad dimensions");

  // BUGBUG: This can only shift along the first axis. Shifting, e.g., along the last axis cannot be implemented this way.
  int offset = 0;
  for(int i = 0; i < shift.size(); ++i)
    offset += in->shape().stride(i) * shift[i];

  if(invert)
    offset = -offset;

  cudaSetDevice(out->getDevice().no);

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  gShift<<<blocks, threads>>>(out->data(), in->data(), length, offset, beta);
}

__global__ void gSetSparse(float* out,
                           const size_t* indices,
                           const float* values,
                           int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out[indices[index]] = values[index];
    }
  }
}

void SetSparse(float* out,
               const std::vector<size_t>& indices,
               const std::vector<float>& values) {
  int length = indices.size();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  size_t* d_indices;
  CUDA_CHECK(cudaMalloc(&d_indices, length * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_indices,
                        indices.data(),
                        length * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  float* d_values;
  CUDA_CHECK(cudaMalloc(&d_values, length * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(
      d_values, values.data(), length * sizeof(float), cudaMemcpyHostToDevice));

  gSetSparse<<<blocks, threads>>>(out, d_indices, d_values, length);

  cudaFree(d_indices);
  cudaFree(d_values);
}

/******************************************************************************/

__global__ void gLSTMCellForward(float* out,
                                 const float* cell,
                                 const float* xW,
                                 const float* sU,
                                 const float* b,
                                 const float* mask,
                                 size_t rows,
                                 size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];

      float* rowOut = out + j * cols;
      const float* rowCell = cell + j * cols;

      const float* xWrow = xW + j * cols * 4;
      const float* sUrow = sU + j * cols * 4;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          float gf = stableLogit(xWrow[i] + sUrow[i] + b[i]);

          int k = i + cols;
          float gi = stableLogit(xWrow[k] + sUrow[k] + b[k]);

          int l = i + 2 * cols;
          float gc = tanhf(xWrow[l] + sUrow[l] + b[l]);

          float cout = gf * rowCell[i] + gi * gc;
          rowOut[i] = m * cout + (1 - m) * rowCell[i];
        }
      }
    }
  }
}

void LSTMCellForward(Tensor out, std::vector<Tensor> inputs) {
  cudaSetDevice(out->getDevice().no);

  int rows = out->shape().elements() / out->shape().back();
  int cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gLSTMCellForward<<<blocks, threads>>>(
      out->data(),                                // output
      inputs[0]->data(),                          // cell state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      inputs.size() > 4 ? inputs[4]->data() : 0,  // mask
      rows,
      cols);
}

__global__ void gLSTMOutputForward(float* out,
                                   const float* cell,
                                   const float* xW,
                                   const float* sU,
                                   const float* b,
                                   size_t rows,
                                   size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowCell = cell + j * cols;

      const float* xWrow = xW + j * cols * 4;
      const float* sUrow = sU + j * cols * 4;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          int k = i + 3 * cols;
          float go = stableLogit(xWrow[k] + sUrow[k] + b[k]);

          rowOut[i] = go * tanhf(rowCell[i]);
        }
      }
    }
  }
}

void LSTMOutputForward(Tensor out, std::vector<Tensor> inputs) {
  cudaSetDevice(out->getDevice().no);

  int rows = out->shape().elements() / out->shape().back();
  int cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gLSTMOutputForward<<<blocks, threads>>>(out->data(),        // output
                                          inputs[0]->data(),  // cell state
                                          inputs[1]->data(),  // xW
                                          inputs[2]->data(),  // sU
                                          inputs[3]->data(),  // b
                                          rows,
                                          cols);
}

__global__ void gLSTMCellBackward(float* outCell,
                                  float* outXW,
                                  float* outSU,
                                  float* outB,
                                  const float* cell,
                                  const float* xW,
                                  const float* sU,
                                  const float* b,
                                  const float* mask,
                                  const float* adj,
                                  size_t rows,
                                  size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];

      float* rowOutCell = outCell + j * cols;
      float* rowOutXW = outXW + j * cols * 4;
      float* rowOutSU = outSU + j * cols * 4;

      const float* rowCell = cell + j * cols;
      const float* xWrow = xW + j * cols * 4;
      const float* sUrow = sU + j * cols * 4;

      const float* rowAdj = adj + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          float gf = stableLogit(xWrow[i] + sUrow[i] + b[i]);

          int k = i + cols;
          float gi = stableLogit(xWrow[k] + sUrow[k] + b[k]);

          int l = i + 2 * cols;
          float gc = tanhf(xWrow[l] + sUrow[l] + b[l]);

          float adj = rowAdj[i];

          // dc/dc_{t-1}
          if(outCell)
            rowOutCell[i] += (m * gf - m + 1) * adj;

          // dc/d(b_f) = dc/d(xW_f) ...
          float dcdxf = m * rowCell[i] * gf * (1 - gf) * adj;
          if(outXW)
            rowOutXW[i] += dcdxf;
          if(outSU)
            rowOutSU[i] += dcdxf;
          if(outB)
            atomicAdd(outB + i, dcdxf);

          // dc/d(b_i) ...
          float dcdb_i = m * gc * gi * (1 - gi) * adj;
          if(outXW)
            rowOutXW[k] += dcdb_i;
          if(outSU)
            rowOutSU[k] += dcdb_i;
          if(outB)
            atomicAdd(outB + k, dcdb_i);

          // dc/d(b_c) ...
          float dcdxc = m * gi * (1 - gc * gc) * adj;
          if(outXW)
            rowOutXW[l] += dcdxc;
          if(outSU)
            rowOutSU[l] += dcdxc;
          if(outB)
            atomicAdd(outB + l, dcdxc);
        }
      }
    }
  }
}

void LSTMCellBackward(std::vector<Tensor> outputs,
                      std::vector<Tensor> inputs,
                      Tensor adj) {
  cudaSetDevice(adj->getDevice().no);

  int rows = adj->shape().elements() / adj->shape().back();
  int cols = adj->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gLSTMCellBackward<<<blocks, threads>>>(
      outputs[0] ? outputs[0]->data() : 0,        // state - adj
      outputs[1] ? outputs[1]->data() : 0,        // xW - adj
      outputs[2] ? outputs[2]->data() : 0,        // sU - adj
      outputs[3] ? outputs[3]->data() : 0,        // b - adj
      inputs[0]->data(),                          // state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      inputs.size() > 4 ? inputs[4]->data() : 0,  // mask
      adj->data(),
      rows,
      cols);
}

__global__ void gLSTMOutputBackward(float* outCell,
                                    float* outXW,
                                    float* outSU,
                                    float* outB,
                                    const float* cell,
                                    const float* xW,
                                    const float* sU,
                                    const float* b,
                                    const float* adj,
                                    size_t rows,
                                    size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOutCell = outCell + j * cols;
      float* rowOutXW = outXW + j * cols * 4;
      float* rowOutSU = outSU + j * cols * 4;

      const float* rowCell = cell + j * cols;
      const float* xWrow = xW + j * cols * 4;
      const float* sUrow = sU + j * cols * 4;

      const float* rowAdj = adj + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          int k = i + 3 * cols;
          float go = stableLogit(xWrow[k] + sUrow[k] + b[k]);

          float t = tanhf(rowCell[i]);

          float adj = rowAdj[i];

          // dc/dc_{t-1}
          if(outCell)
            rowOutCell[i] += go * (1 - t * t) * adj;

          // dc/d(b_o) = dc/d(xW_f) ...
          float dcdxo = t * go * (1 - go) * adj;
          if(outXW)
            rowOutXW[k] += dcdxo;
          if(outSU)
            rowOutSU[k] += dcdxo;
          if(outB)
            atomicAdd(outB + k, dcdxo);
        }
      }
    }
  }
}

void LSTMOutputBackward(std::vector<Tensor> outputs,
                        std::vector<Tensor> inputs,
                        Tensor adj) {
  cudaSetDevice(adj->getDevice().no);

  int rows = adj->shape().elements() / adj->shape().back();
  int cols = adj->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gLSTMOutputBackward<<<blocks, threads>>>(
      outputs[0] ? outputs[0]->data() : 0,  // state - adj
      outputs[1] ? outputs[1]->data() : 0,  // xW - adj
      outputs[2] ? outputs[2]->data() : 0,  // sU - adj
      outputs[3] ? outputs[3]->data() : 0,  // b - adj
      inputs[0]->data(),                    // state
      inputs[1]->data(),                    // xW
      inputs[2]->data(),                    // sU
      inputs[3]->data(),                    // b
      adj->data(),
      rows,
      cols);
}

__global__ void gHighwayForward(float* out,
                                const float* in1,
                                const float* in2,
                                const float* t,
                                size_t length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      float sigma = stableLogit(t[index]);
      out[index] = in1[index] * sigma + in2[index] * (1.f - sigma);
    }
  }
}

void HighwayForward(Tensor out,
                    const Tensor in1,
                    const Tensor in2,
                    const Tensor t) {
  cudaSetDevice(out->getDevice().no);

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  gHighwayForward<<<blocks, threads>>>(
      out->data(), in1->data(), in2->data(), t->data(), length);
}

__global__ void gHighwayBackward(float* out1,
                                 float* out2,
                                 float* outt,
                                 const float* in1,
                                 const float* in2,
                                 const float* t,
                                 const float* adj,
                                 size_t length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      float sigma = stableLogit(t[index]);
      out1[index] = sigma * adj[index];
      out2[index] = (1.f - sigma) * adj[index];
      outt[index]
          = sigma * (1.f - sigma) * (in1[index] - in2[index]) * adj[index];
    }
  }
}

void HighwayBackward(Tensor out1,
                     Tensor out2,
                     Tensor outt,
                     const Tensor in1,
                     const Tensor in2,
                     const Tensor t,
                     const Tensor adj) {
  cudaSetDevice(out1->getDevice().no);

  int length = out1->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  gHighwayBackward<<<blocks, threads>>>(out1->data(),
                                        out2->data(),
                                        outt->data(),
                                        in1->data(),
                                        in2->data(),
                                        t->data(),
                                        adj->data(),
                                        length);
}

__global__ void gMaxPoolingForward(float* out,
                                   int outRows,
                                   int outCols,
                                   float* in,
                                   int inRows,
                                   int inCols,
                                   float* mask,
                                   int numKernels,
                                   int maskCols,
                                   int width,
                                   int lastWidth) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if(tid >= outRows * outCols)
    return;

  int rowId = tid / outRows;
  int colId = tid % outRows;

  float* b = in + (rowId * inCols) + (colId * width);
  float* localMask = mask + (rowId / numKernels) * maskCols + colId * width;

  if(colId == outRows - 1) {
    width = lastWidth;
  }

  float currentMax = b[0] * localMask[0];
  for(int i = 1; i < width; ++i) {
    if(b[i] * localMask[i] > currentMax) {
      currentMax = b[i] * localMask[i];
    }
  }

  out[rowId + (colId * outCols)] = currentMax;
}

void PoolingWithMaskingForward(Tensor out,
                               Tensor in,
                               Tensor mask,
                               int width,
                               bool isEven) {
  int n = out->shape().elements();
  int threads = std::min(n, MAX_THREADS);
  int blocks = n / threads + (n % threads != 0);

  auto& inShape = in->shape();
  int inRows = inShape[0] * inShape[1];
  int inCols = inShape[2];

  auto& outShape = out->shape();
  int outRows = outShape[2];
  int outCols = outShape[0] * outShape[1];

  int lastWidth
      = ((inCols - isEven) % width == 0) ? width : (inCols - isEven) % width;

  gMaxPoolingForward<<<blocks, threads>>>(out->data(),
                                          outRows,
                                          outCols,
                                          in->data(),
                                          inRows,
                                          inCols,
                                          mask->data(),
                                          outShape[1],
                                          mask->shape()[2],
                                          width,
                                          lastWidth);
}

__global__ void gMaxPoolingBackward(float* adj,
                                    int adjRows,
                                    int adjCols,
                                    float* in,
                                    float* adjIn,
                                    int inRows,
                                    int inCols,
                                    float* mask,
                                    int numKernels,
                                    int maskCols,
                                    int width,
                                    int lastWidth) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if(tid >= adjRows * adjCols)
    return;

  int rowId = tid / adjRows;
  int colId = tid % adjRows;

  float* b = in + (rowId * inCols) + (colId * width);

  if(colId == adjRows - 1) {
    width = lastWidth;
  }

  float* localMask = mask + (rowId / numKernels) * maskCols + colId * width;
  size_t currentMaxIdx = 0;
  for(int i = 1; i < width; ++i) {
    if(b[i] * localMask[i] > b[currentMaxIdx] * localMask[currentMaxIdx]) {
      currentMaxIdx = i;
    }
  }

  adjIn[(rowId * inCols) + (colId * width) + currentMaxIdx]
      += adj[rowId + (colId * adjCols)];
}

void PoolingWithMaskingBackward(Tensor adj,
                                Tensor adjIn,
                                Tensor in,
                                Tensor mask,
                                int width,
                                bool isEven) {
  int n = adj->shape().elements();
  int threads = std::min(n, 512);
  int blocks = n / threads + (n % threads != 0);

  auto& inShape = in->shape();
  int inRows = inShape[0] * inShape[1];
  int inCols = inShape[2];

  auto& adjShape = adj->shape();
  int adjRows = adjShape[2];
  int adjCols = adjShape[0] * adjShape[1];

  int lastWidth
      = ((inCols - isEven) % width == 0) ? width : (inCols - isEven) % width;

  gMaxPoolingBackward<<<blocks, threads>>>(adj->data(),
                                           adjRows,
                                           adjCols,
                                           in->data(),
                                           adjIn->data(),
                                           inRows,
                                           inCols,
                                           mask->data(),
                                           adjShape[1],
                                           mask->shape()[2],
                                           width,
                                           lastWidth);
}

__global__ void gBalancedMoeSlicerWithMask(float* out,
                                           const int numTokens,
                                           const int dim,
                                           const int exp,
                                           const int m,
                                           const float* in,
                                           const float* mask) {
  // out (EXPERTS, M, DIM)
  for(int bid = 0; bid < exp; bid += gridDim.x) {
    int e = bid + blockIdx.x;
    if(e < exp) {

      // Zero output matrix
      for(int m_ = 0; m_ < m; ++m_) {
        for(int tid = 0; tid < dim; tid += blockDim.x) {
          out[(e * m + m_) * + (tid % dim)] = 0.0f;
        }
      }
      __syncthreads();

      for(int t = 0; t < numTokens; ++t) {
        int m_ = (int)mask[t * exp + e];
        if(m_ >= 0) {
          const float* src = &in[t * dim];
          float* tgt = &out[(e * m + m_) * dim];
          for(int tid = 0; tid < dim; tid += blockDim.x) {
            int d = tid + threadIdx.x;
            if(d < dim) {
              tgt[d] = src[d];
            }
          }
        }
      }
    }
    __syncthreads();
  }
}

void BalancedMoeSlicerWithMask(Tensor out, Tensor in, Tensor mask) {
  cudaSetDevice(out->getDevice().no);

  size_t numTokens = in->shape().elements() / in->shape().back();
  size_t dim = in->shape().back();
  size_t exp = out->shape()[-3];
  size_t m = out->shape()[-2];

  int blocks = std::min(MAX_BLOCKS, (int)exp);
  int threads = std::min(MAX_THREADS, (int)dim);

  gBalancedMoeSlicerWithMask<<<blocks, threads>>>(
      out->data(), numTokens, dim, exp, m, in->data(), mask->data());

  if(IsNan(out))
    std::cout << "NaNs or infs spotted in Slicer\n";
}

__global__ void gBalancedMoeSlicerWithMaskGrad(float* grad,
                                               const int numTokens,
                                               const int dim,
                                               const int exp,
                                               const int m,
                                               const float* mask,
                                               const float* adj) {
  // grad (TOKENS, DIM)
  for(int bid = 0; bid < exp; bid += gridDim.x) {
    int e = bid + blockIdx.x;
    if(e < exp) {
      for(int t = 0; t < numTokens; ++t) {
        int m_ = (int)mask[t * exp + e];
        if(m_ >= 0) {

          float* tgt = &grad[t * dim];
          const float* adjSrc = &adj[(e * m + m_) * dim];

          for(int tid = 0; tid < dim; tid += blockDim.x) {
            int d = tid + threadIdx.x;
            if(d < dim) {
              atomicAdd(tgt + d, adjSrc[d]);
            }
          }

        }
      }
    }
    __syncthreads();
  }
}

void BalancedMoeSlicerWithMaskGrad(Tensor grad, Tensor val, Tensor mask, Tensor adj) {
  cudaSetDevice(grad->getDevice().no);

  size_t numTokens = grad->shape().elements() / grad->shape().back();
  size_t dim = grad->shape().back();
  size_t exp = adj->shape()[-3];
  size_t m = adj->shape()[-2];

  int blocks = std::min(MAX_BLOCKS, (int)exp);
  int threads = std::min(MAX_THREADS, (int)dim);

  gBalancedMoeSlicerWithMaskGrad<<<blocks, threads>>>(
      grad->data(), numTokens, dim, exp, m, mask->data(), adj->data());

  if(IsNan(grad))
    std::cout << "NaNs or infs spotted in SlicerGrad\n";
}

__global__ void gBalancedMoeSlicer(float* out,
                                   const int exp,
                                   const int m,
                                   const int dim,
                                   const float* in,
                                   const float* inds) {
  for(int bid = 0; bid < m; bid += gridDim.x) {
    int m_ = bid + blockIdx.x;
    if(m_ >= m)
      continue;

    for(int tid = 0; tid < exp; tid += blockDim.x) {
      int exp_ = tid + threadIdx.x;
      if(exp_ >= exp)
        continue;

      int t_ = (int)inds[exp_ * m + m_];
      const float* src = &in[t_ * dim];  // in[t,:]
      float* tgt = &out[(exp_ * m + m_) * dim]; // out[e,m,:]
      // cublasScopy(cublasHandle, dim, src, 0, tgt, 0);
      // auto cublasHandle = std::static_pointer_cast<gpu::Backend>(out->getBackend())
      //                         ->getCublasHandle();
      for (int i = 0; i < dim; ++i)
        tgt[i] = src[i];
    }
    __syncthreads();
  }
}

void BalancedMoeSlicer(Tensor out, Tensor in, Tensor inds) {
  cudaSetDevice(out->getDevice().no);

  size_t exp = inds->shape().elements() / inds->shape().back();
  size_t m = inds->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)exp);

  int _exp = out->shape()[-3];
  int _m = out->shape()[-2];
  int _dim = out->shape()[-1];

  gBalancedMoeSlicer<<<blocks, threads>>>(
      out->data(), _exp, _m, _dim, in->data(), inds->data());
}

__global__ void gBalancedMoeSlicerGrad(float* grad,
                                       const int exp,
                                       const int m,
                                       const int dim,
                                       const float* val, // XXX unused
                                       const float* inds,
                                       const float* adj) {
  // TODO Zero the grad matrix?
  for(int pos = 0; pos < m * exp; ++pos) {

    int32_t t = (int32_t)inds[pos];
    if(t % gridDim.x == blockIdx.x) {

      float* tgt = &grad[t * dim];  // in[t,:]
      const float* adjSrc = &adj[pos * dim]; // out[e,m,:]

      for(int tid = 0; tid < dim; tid += blockDim.x) {
        int dim_ = tid + threadIdx.x;
        if(dim_ < dim)
          tgt[dim_] += adjSrc[dim_];
      }
    }
    __syncthreads();
    // cublasScopy(cublasHandle, dim, src, 0, tgt, 0);
    // auto cublasHandle = std::static_pointer_cast<gpu::Backend>(out->getBackend())
    //                         ->getCublasHandle();
  }
}

void BalancedMoeSlicerGrad(Tensor grad, Tensor val, Tensor inds, Tensor adj) {
  cudaSetDevice(grad->getDevice().no);

  size_t exp = inds->shape().elements() / inds->shape().back();
  size_t m = inds->shape().back();

  int blocks = 1; // XXX std::min(MAX_BLOCKS, (int)m);
  int threads = 1; // XXX std::min(MAX_THREADS, (int)exp);

  int _exp = adj->shape()[-3];
  int _m = adj->shape()[-2];
  int _dim = adj->shape()[-1];

  gBalancedMoeSlicerGrad<<<blocks, threads>>>(
      grad->data(), _exp, _m, _dim, val->data(), inds->data(), adj->data());
}

__global__ void gBalancedMoeStitcher(float* out,
                                     const int exp,
                                     const int m,
                                     const int dim,
                                     const float* in,
                                     const float* inds,
                                     const size_t numTokens) {
  // out (TOKENS, DIM)
  // TODO Zero the output matrix?

  for(int pos = 0; pos < m * exp; ++pos) {

    int32_t t = (int32_t)inds[pos];
    if(t % gridDim.x == blockIdx.x) {

      const float* src = &in[pos * dim];
      float* tgt = &out[t * dim];

      for(int tid = 0; tid < dim; tid += blockDim.x) {
        int dim_ = tid + threadIdx.x;
        if(dim_ < dim)
          tgt[dim_] += src[dim_];
      }
    }
    __syncthreads();
    // cublasScopy(cublasHandle, dim, src, 0, tgt, 0);
    // auto cublasHandle = std::static_pointer_cast<gpu::Backend>(out->getBackend())
    //                         ->getCublasHandle();
  }
}

void BalancedMoeStitcher(
    Tensor out, Tensor in, Tensor inds, const size_t numTokens) {
  cudaSetDevice(out->getDevice().no);

  size_t exp = inds->shape().elements() / inds->shape().back();
  size_t m = inds->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)exp);

  int _exp = in->shape()[-3];
  int _m = in->shape()[-2];
  int _dim = in->shape()[-1];

  gBalancedMoeStitcher<<<blocks, threads>>>(
      out->data(), _exp, _m, _dim, in->data(), inds->data(), numTokens);
}

__global__ void gBalancedMoeStitcherWithMask(float* out,
                                             const int exp,
                                             const int m,
                                             const int dim,
                                             const int numTokens,
                                             const float* in,
                                             const float* mask) {
  // out (TOKENS, DIM)
  for(int bid = 0; bid < numTokens; bid += gridDim.x) {
    int t = bid + blockIdx.x;
    if(t < numTokens) {

      // Zero output matrix
      for(int tid = 0; tid < dim; tid += blockDim.x) {
        out[t * dim + (tid % dim)] = 0.0f;
      }
      __syncthreads();

      for(int tid = 0; tid < exp; tid += blockDim.x) {
        int e = tid + threadIdx.x;
        if(e < exp) {
          int m_ = (int)mask[t * exp + e];
          if(m_ >= 0) {
            const float* src = &in[(e * m + m_) * dim];
            for(int d = 0; d < dim; ++d)
              atomicAdd(out + t * dim + d, src[d]);
          }
        }
      }

    }
  }
}

void BalancedMoeStitcherWithMask(Tensor out, Tensor in, Tensor mask) {
  cudaSetDevice(out->getDevice().no);

  size_t exp = in->shape()[-3];
  size_t m = in->shape()[-2];
  size_t dim = in->shape()[-1];
  size_t numTokens = out->shape().elements() / out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)numTokens);
  int threads = std::min(MAX_THREADS, (int)exp);

  gBalancedMoeStitcherWithMask<<<blocks, threads>>>(
      out->data(), exp, m, dim, numTokens, in->data(), mask->data());

  if(IsNan(out))
    std::cout << "NaNs or infs spotted in Stitcher\n";
}

__global__ void gBalancedMoeStitcherWithMaskGrad(float* grad,
                                                 const int exp,
                                                 const int m,
                                                 const int dim,
                                                 const int numTokens,
                                                 const float* val,
                                                 const float* mask,
                                                 const float* adj) {
  // grad (TOKENS, EXPERTS)
  for(int bid = 0; bid < numTokens; bid += gridDim.x) {
    int t = bid + blockIdx.x;
    if(t < numTokens) {

      for(int tid = 0; tid < exp; tid += blockDim.x) {
        int e = tid + threadIdx.x;
        if(e < exp) {
          int m_ = (int)mask[t * exp + e];
          if(m_ >= 0) {
            const float* src = &adj[t * dim];
            float* tgt = &grad[(e * m + m_) * dim];

            for(int d = 0; d < dim; ++d)
              atomicAdd(tgt + d, src[d]);
          }
        }
      }

    }
  }
}

void BalancedMoeStitcherWithMaskGrad(
    Tensor grad, Tensor val, Tensor mask, Tensor adj) {
  cudaSetDevice(grad->getDevice().no);

  size_t exp = val->shape()[-3];
  size_t m = val->shape()[-2];
  size_t dim = val->shape()[-1];
  size_t numTokens = mask->shape().elements() / mask->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)numTokens);
  int threads = std::min(MAX_THREADS, (int)exp);

  gBalancedMoeStitcherWithMaskGrad<<<blocks, threads>>>(
      grad->data(), exp, m, dim, numTokens, val->data(), mask->data(), adj->data());

  if(IsNan(grad))
    std::cout << "NaNs or infs spotted in StitcherGrad\n";
}

__global__ void gBalancedMoeStitcherGrad(float* grad,
                                         const int exp,
                                         const int m,
                                         const int dim,
                                         const float* val,
                                         const float* inds,
                                         const float* adj) {
  for(int bid = 0; bid < m; bid += gridDim.x) {
    int m_ = bid + blockIdx.x;
    if(m_ >= m)
      continue;

    for(int tid = 0; tid < exp; tid += blockDim.x) {
      int exp_ = tid + threadIdx.x;
      if(exp_ >= exp)
        continue;

      size_t t_ = (size_t)inds[exp_ * m + m_];
      const float* src = &adj[t_ * dim];  // in[t,:]
      float* tgt = &grad[(exp_ * m + m_) * dim]; // out[e,m,:]
      for (int i = 0; i < dim; ++i)
        tgt[i] += src[i];
    }
    __syncthreads();
  }
}

void BalancedMoeStitcherGrad(
    Tensor grad, Tensor val, Tensor inds, Tensor adj) {
  cudaSetDevice(grad->getDevice().no);

  size_t exp = inds->shape().elements() / inds->shape().back();
  size_t m = inds->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)exp);

  int _exp = grad->shape()[-3];
  int _m = grad->shape()[-2];
  int _dim = grad->shape()[-1];

  gBalancedMoeStitcherGrad<<<blocks, threads>>>(
      grad->data(), _exp, _m, _dim, val->data(), inds->data(), adj->data());
}

__global__ void gBalancedMoeNormalizeGateWithMask(float *out,
                                                  const int exp,
                                                  const int m,
                                                  const float* in,
                                                  const float* mask,
                                                  const int numTokens) {
  // in (NUM_TOKENS, EXPERTS)
  // mask (NUM_TOKENS, EXPERTS)
  // out (EXPERTS, M)

  // Initialize shared memory
  extern __shared__ float shared[];
  float* norm = shared;
  for(int bid = 0; bid < numTokens; bid += gridDim.x) {
    int t = bid + blockIdx.x;
    if(t < numTokens) {

      // Accumulate normalizing constants
      norm[t] = 0.0;
      __syncthreads();
      for(int tid = 0; tid < exp; tid += blockDim.x) {
        int e = tid + threadIdx.x;
        if(e < exp) {
          if(mask[t * exp + e] > -0.5f) {
            atomicAdd(norm + t, in[t * exp + e]);
          }
        }
      }
      __syncthreads();

      if(threadIdx.x == 0) {
        for(int ee1 = 0; ee1 < exp; ++ee1) {
          int m_ = (int)mask[t * exp + ee1];
          if(m_ >= 0 && fabs(norm[t]) < 1e-20) {
            printf("(%d) norm[%d] == %f\n", blockIdx.x, t, norm[t]);
            printf("(%d) exp %d ERROR FOR i == %d\n", blockIdx.x, exp, t);
            // for(int tt = 0; tt < numTokens; ++tt) {
              for(int ee = 0; ee < exp; ++ee)
                printf("(%d) %f\n", blockIdx.x, in[t * exp + ee]);
              printf("(%d) mask:\n", blockIdx.x);
              for(int ee = 0; ee < exp; ++ee)
                printf("(%d) %d ", blockIdx.x, mask[t * exp + ee]);
              printf("\n");
            // }
            return;
          }
        }
      }

      for(int tid = 0; tid < exp; tid += blockDim.x) {
        int e = tid + threadIdx.x;
        if(e < exp) {
          if(mask[t * exp + e] > -0.5f) {
            int m_ = (int)mask[t * exp + e];
            // MYASSERT(fabs(norm[t]) < 1e-20, "Gate: norm[t] == 0\n");
            out[e * m + m_] = in[t * exp + e] / norm[t];
          }
        }
      }

    }
  }
}

void BalancedMoeNormalizeGateWithMask(Tensor out, Tensor in, Tensor mask, const int UNUSED) {
  cudaSetDevice(out->getDevice().no);

  size_t m = out->shape().back();
  size_t exp = mask->shape().back();
  size_t numTokens = mask->shape().elements() / mask->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)numTokens);
  int threads = std::min(MAX_THREADS, (int)m);
  int shared = sizeof(float) * numTokens * 2;

  gBalancedMoeNormalizeGateWithMask<<<blocks, threads, shared>>>(
      out->data(), exp, m, in->data(), mask->data(), numTokens);

  if(IsNan(in))
    std::cout << "NaNs or infs spotted in Gate (input)\n";
  if(IsNan(out))
    std::cout << "NaNs or infs spotted in Gate\n";
}

__global__ void gBalancedMoeNormalizeGateWithMaskGrad(float* grad,
                                                      const int exp,
                                                      const int m,
                                                      const int numTokens,
                                                      const float* in,
                                                      const float* val,
                                                      const float* mask,
                                                      const float* adj) {
  // grad (TOKENS, EXPERTS)

  // Initialize shared memory
  extern __shared__ float shared[];
  float* norm = shared;
  for(int bid = 0; bid < numTokens; bid += gridDim.x) {
    int t = bid + blockIdx.x;
    if(t < numTokens) {

      // Accumulate normalizing constants
      norm[t] = 0.0;
      __syncthreads();
      for(int tid = 0; tid < exp; tid += blockDim.x) {
        int e = tid + threadIdx.x;
        if(e < exp) {
          if(mask[t * exp + e] > -0.5f) {
            atomicAdd(norm + t, in[t * exp + e]);
          }
        }
      }
      __syncthreads();

      // Compute the gradients
      for(int tid = 0; tid < exp; tid += blockDim.x) {
        int e = tid + threadIdx.x;
        if(e < exp) {
          if (mask[t * exp + e] > -0.5f) {
            int m_ = (int)mask[t * exp + e];
            for(int egrad = 0; egrad < exp; ++egrad) {
              // MYASSERT(fabs(norm[t]) < 1e-20, "GateGrad: norm[t] == 0\n");
              float partial = ((float)(e == egrad) - val[e * m + m_]) / norm[t];
              float adj_ = adj[e * m + m_];
              float flag = ((float)(mask[t * exp + egrad] > -0.5f));
              atomicAdd(grad + (t * exp + egrad), flag * adj_ * partial);
            }
          }
        }
      } // finished computing gradients

    }
  }
}

void BalancedMoeNormalizeGateWithMaskGrad(
    Tensor grad, Tensor in, Tensor val, Tensor mask, Tensor adj) {
  cudaSetDevice(grad->getDevice().no);

  size_t m = adj->shape().back();
  size_t exp = mask->shape().back();
  size_t numTokens = mask->shape().elements() / mask->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)numTokens);
  int threads = std::min(MAX_THREADS, (int)exp);
  int shared = sizeof(float) * numTokens * 2;

  gBalancedMoeNormalizeGateWithMaskGrad<<<blocks, threads, shared>>>(
      grad->data(), exp, m, numTokens, in->data(), val->data(), mask->data(), adj->data());

  // bool nan_ = false;
  // gIsNan<<<blocks, threads>>>(grad->data(), &nan_, numTokens * exp);
  if(IsNan(grad))
    std::cout << "NaNs or infs spotted in GateGrad\n";
}

/**************************************************/

__global__ void gBalancedMoeNormalizeGate(float *out,
                                          const int exp,
                                          const int m,
                                          const float* in,
                                          const float* inds,
                                          const int numTokens) {
  // in (NUM_TOKENS, EXPERTS)
  // inds (EXPERTS, M)
  // out (EXPERTS, M)

  // Initialize shared memory
  extern __shared__ float shared[];
  float* norm = shared;
  for(int btid = 0; btid < numTokens; btid += (gridDim.x * blockDim.x)) {
    int pos = btid + (blockIdx.x * blockDim.x + threadIdx.x);
    if(pos < numTokens)
      norm[pos] = 0.0;
  }
  __syncthreads();

  if(blockIdx.x == 0 && threadIdx.x == 0) {
    printf("NUM TOKENS: %d\n", numTokens);
    for(int i = 0; i < numTokens; ++i)
      printf("%f ", norm[i]);
    printf("\n");
  }
  __syncthreads();

  // Go over `inds` and aggregate normalizations in shared mem

  // `inds` are guaranteed to be distinct in each row
  // To avoid race conditions, we will process each row with all blocks/threads,
  // along the `m` dimension
  for(int exp_ = 0; exp_ < exp; ++exp_) {
    for(int btid = 0; btid < m; btid += (gridDim.x * blockDim.x)) {
      int m_ = btid + (blockIdx.x * blockDim.x + threadIdx.x);
      if(m_ < m) {
        int pos = exp_ * m + m_;
        int t = (int)inds[pos];
        norm[t] += in[t * exp + exp_];
      }
    }
    __syncthreads();
  }

  if(blockIdx.x == 0 && threadIdx.x == 0) {
    for(int i = 0; i < numTokens; ++i)
      printf("%f ", norm[i]);
    printf("\n");
  }
  __syncthreads();

  // Copy and normalize in to out
  for(int bid = 0; bid < exp; bid += gridDim.x) {
    int exp_ = bid + blockIdx.x;
    if (exp_ < exp) {
      for(int tid = 0; tid < m; tid += blockDim.x) {
        int m_ = tid + threadIdx.x;
        if(m_ < m) {
          int pos = exp_ * m + m_;
          int t = (int)inds[pos];
          out[pos] = (fabs(norm[t]) < 1e-10f ? 0.0f : in[t * exp + exp_] / norm[t]);
          printf("t:%d (%d %d)\t%f --> %f (norm %f)\n", t, exp_, m_, in[t*exp+exp_], out[pos], norm[t]);
        }
      }
    }
    __syncthreads();
  }
}

void BalancedMoeNormalizeGate(Tensor out, Tensor in, Tensor inds, const int numTokens) {
  cudaSetDevice(out->getDevice().no);

  size_t exp = inds->shape().elements() / inds->shape().back();
  size_t m = inds->shape().back();

  int blocks = 1; // XXX XXX XXX std::min(MAX_BLOCKS, (int)exp);
  int threads = std::min(MAX_THREADS, (int)m);
  int shared = sizeof(float) * numTokens * 2;

  gBalancedMoeNormalizeGate<<<blocks, threads, shared>>>(
      out->data(), exp, m, in->data(), inds->data(), numTokens);
}

__global__ void gBalancedMoeNormalizeGateGrad(float* grad,
                                              const int exp,
                                              const int m,
                                              const int numTokens,
                                              const float* in,
                                              const float* val,
                                              const float* inds,
                                              const float* adj) {

  // grad (TOKENS, EXPERTS)

  // 1. Initialize shared memory
  // 2. Set `grad` to +infty
  // 3. Walk over `inds` and set `grad` to 0.0 for participating fields
  // 4. Walk over `inds` and compute partial derivatives
  //   - assume the whole column of `in` (and thus `grad`) participates in normalization
  //   - so update a whole column of `grad`
  //   - not participating fields are easily distinguishable by being +infty
  // 5. Zero +infty fields of `grad`

  // Initialize shared memory
  extern __shared__ float shared[];
  float* norm = shared;
  for(int btid = 0; btid < numTokens; btid += (gridDim.x * blockDim.x)) {
    int pos = btid + (blockIdx.x * blockDim.x + threadIdx.x);
    if(pos < numTokens)
      norm[pos] = 0.0;
  }
  __syncthreads();

  // 1. a) compute normalization vector
  for(int exp_ = 0; exp_ < exp; ++exp_) {
    for(int btid = 0; btid < m; btid += (gridDim.x * blockDim.x)) {
      int m_ = btid + (blockIdx.x * blockDim.x + threadIdx.x);
      if(m_ < m) {
        int pos = exp_ * m + m_;
        int t = (int)inds[pos];
        norm[t] += in[t * exp + exp_];
      }
    }
    __syncthreads();
  }

  if(blockIdx.x == 0 && threadIdx.x == 0) {
    printf("NUM TOKENS: %d\n", numTokens);
    for(int i = 0; i < numTokens; ++i)
      printf("%f ", norm[i]);
    printf("\n");
  }
  __syncthreads();

  // 2. Set `grad` to +infty
  const float infty = 1e20;
  for(int btid = 0; btid < exp * numTokens; btid += (gridDim.x * blockDim.x)) {
    int pos = btid + (blockIdx.x * blockDim.x + threadIdx.x);
    if(pos < exp * numTokens)
      grad[pos] = infty;
  }
  __syncthreads();

  // 3. Walk over `inds` and set `grad` to 0.0 for participating fields
  for(int exp_ = 0; exp_ < exp; ++exp_) {
    for(int btid = 0; btid < m; btid += (gridDim.x * blockDim.x)) {
      int m_ = btid + (blockIdx.x * blockDim.x + threadIdx.x);
      if(m_ < m) {
        int pos = exp_ * m + m_;
        int t = (int)inds[pos];
        grad[t * exp + exp_] = 0.0f;
      }
    }
    __syncthreads();
  }

  printf("ADJ\n");
  for(int exp_ = 0; exp_ < exp; ++exp_) {
    for(int btid = 0; btid < m; btid += (gridDim.x * blockDim.x)) {
      int m_ = btid + (blockIdx.x * blockDim.x + threadIdx.x);
      if(m_ < m) {
        printf("%f ", adj[exp_*m+m_]);
      }
    }
    printf("\n");
  }

  // 4. Walk over `inds` and compute partial derivatives
  //   - assume the whole column of `in` (and thus `grad`) participates in normalization
  //   - so update a whole column of `grad`
  //   - not participating fields are easily distinguishable by being +infty
  for(int exp_ = 0; exp_ < exp; ++exp_) {
    for(int btid = 0; btid < m; btid += (gridDim.x * blockDim.x)) {
      int m_ = btid + (blockIdx.x * blockDim.x + threadIdx.x);
      if(m_ < m) {
        int pos = exp_ * m + m_;
        int t = (int)inds[pos];

        for(int grad_exp_ = 0; grad_exp_ < exp; ++grad_exp_) {
          if(fabs(norm[t]) > 1e-10) { // norm[t] != 0.0
            float partial;
            partial = (grad_exp_ != exp_) ? - val[pos] / norm[t] : (1.0f - val[pos]) / norm[t];
            // XXX Watch out for race conditions
            grad[t * exp + grad_exp_] += adj[pos] * partial;
          }
        }

      }
    }
    __syncthreads();
  }

  // 5. Zero +infty fields of `grad`
  for(int btid = 0; btid < exp * numTokens; btid += (gridDim.x * blockDim.x)) {
    int pos = btid + (blockIdx.x * blockDim.x + threadIdx.x);
    if(pos < exp * numTokens) {
      // XXX XXX XXX XXX XXX
      if(grad[pos] > 1e18)
        grad[pos] = 0.0f;
    }
  }
  __syncthreads();
}

void BalancedMoeNormalizeGateGrad(
    Tensor grad, Tensor in, Tensor val, Tensor inds, Tensor adj) {
  cudaSetDevice(grad->getDevice().no);

  size_t exp = inds->shape().elements() / inds->shape().back();
  size_t m = inds->shape().back();
  int numTokens = grad->shape()[0];

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)exp);
  int shared = sizeof(float) * numTokens * 2;

  gBalancedMoeNormalizeGateGrad<<<blocks, threads, shared>>>(
      grad->data(), exp, m, numTokens, in->data(), val->data(), inds->data(), adj->data());
}

}  // namespace marian::gpu
}  // namespace marian
