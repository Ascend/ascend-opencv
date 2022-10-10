#include "precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::acl;
namespace cv {
namespace acl {

static int merge_type(int depth, int channels) {
  switch (depth) {
    case CV_8U:
      return CV_8UC(channels);
    case CV_8S:
      return CV_8SC(channels);
    case CV_32F:
      return CV_32FC(channels);
    case CV_32S:
      return CV_32SC(channels);
    case CV_64F:
      return CV_64FC(channels);
    default:
      return -1;
  }
}

void merge(const vector<aclMat> &mv, aclMat &dest, int stream_id) {
  vector<aclDataBuffer *> inputBuffers_;
  vector<aclDataBuffer *> outputBuffers_;

  OperatorDesc opDesc("Concat");
  aclDataType dataType = type_transition(mv[0].depth());

  vector<int64_t> inputShape {};
  opDesc.AddInputTensorDesc(ACL_INT32, inputShape.size(), inputShape.data(),
                            ACL_FORMAT_ND);

  for (size_t i = 0; i < mv.size(); ++i) {
    int cols = mv[i].step / mv[i].elemSize();
    vector<int64_t> inputShape{1, mv[i].rows, cols, mv[i].channels()};
    opDesc.AddInputTensorDesc(dataType, inputShape.size(), inputShape.data(),
                              ACL_FORMAT_NHWC);
  }

  int cols = mv[0].step / mv[0].elemSize();
  int channels = mv.size();
  vector<int64_t> outputShape{1, mv[0].rows, cols, channels};
  opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(),
                             ACL_FORMAT_NHWC);

  ino64_t N = mv.size();
  constexpr int index2 = 2;
  constexpr int index3 = 3;
  constexpr int index4 = 4;
  constexpr int merge_size3 = 3;
  constexpr int merge_size4 = 4;
  aclopSetAttrInt(opDesc.opAttr, "N", N);

  aclSetTensorDescName(opDesc.inputDesc[0], "concat_dim");

  aclSetTensorDescName(opDesc.inputDesc[1], "x0");
  aclSetTensorDescName(opDesc.inputDesc[index2], "x1");
  if (mv.size() == merge_size3)
    aclSetTensorDescName(opDesc.inputDesc[index3], "x2");
  else if (mv.size() == merge_size4)
    aclSetTensorDescName(opDesc.inputDesc[index4], "x3");
  aclSetTensorDescName(opDesc.outputDesc[0], "y");

  void *dev;
  int64_t concat_dim = 3;
  size_t size = aclGetTensorDescSize(opDesc.inputDesc[0]);
  aclrtMalloc(&dev, size, ACL_MEM_MALLOC_NORMAL_ONLY);
  aclrtMemcpy(dev, size, &concat_dim, size, ACL_MEMCPY_HOST_TO_DEVICE);
  inputBuffers_.emplace_back(aclCreateDataBuffer(dev, size));

  for (size_t i = 0; i < mv.size(); ++i)
    inputBuffers_.emplace_back(
        aclCreateDataBuffer(mv[i].data, mv[i].totalSize));

  constexpr int false_type_flag = -1;
  int type = merge_type(mv[0].depth(), channels);
  CV_Assert(type != false_type_flag);
  aclMat temp(mv[0].rows, mv[0].cols, type, mv[0].acl_context);
  dest = temp;
  outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));

  compileAndRunop(opDesc, inputBuffers_, outputBuffers_, dest.acl_context,
                  stream_id);

  for (size_t i = 0; i < inputBuffers_.size(); i++)
    AclSafeCall(aclDestroyDataBuffer(inputBuffers_[i]));
  for (size_t i = 0; i < outputBuffers_.size(); i++)
    AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));

  aclrtFree(dev);
}

/**
 * @brief : Dynamic shape reasoning
 *
 */

void transpose(const aclMat &src, aclMat &dest, int stream_id) {
  vector<aclDataBuffer *> inputBuffers_;
  vector<aclDataBuffer *> outputBuffers_;
  vector<aclDataBuffer *> inputBuffers_host;

  OperatorDesc opDesc("Transpose");
  aclDataType dataType = type_transition(src.depth());

  vector<int64_t> inputShape1{1, src.rows, src.cols, src.channels()};
  opDesc.AddInputTensorDesc(dataType, inputShape1.size(), inputShape1.data(),
                            ACL_FORMAT_ND);

  vector<int64_t> inputShape2{4};
  opDesc.AddInputTensorDesc(ACL_INT32, inputShape2.size(), inputShape2.data(),
                            ACL_FORMAT_ND);

  vector<int64_t> outputShape{-1, -1, -1, -1};
  opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(),
                             ACL_FORMAT_ND);

  inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));

  void *dev;
  void *perm;
  constexpr int dim0_t = 0;
  constexpr int dim1_t = 1;
  constexpr int dim2_t = 2;
  constexpr int dim3_t = 3;

  size_t size = aclGetTensorDescSize(opDesc.inputDesc[1]);
  aclrtMalloc(&dev, size, ACL_MEM_MALLOC_NORMAL_ONLY);
  aclrtMallocHost(&perm, aclGetTensorDescSize(opDesc.inputDesc.data()[1]));
  ((int *)perm)[0] = dim0_t;
  ((int *)perm)[1] = dim2_t;
  ((int *)perm)[2] = dim1_t;
  ((int *)perm)[3] = dim3_t;
  aclrtMemcpy(dev, size, perm, size, ACL_MEMCPY_HOST_TO_DEVICE);
  inputBuffers_.emplace_back(aclCreateDataBuffer(dev, size));

  AclSafeCall(aclopCompile(opDesc.opType.c_str(), opDesc.inputDesc.size(),
                           opDesc.inputDesc.data(), opDesc.outputDesc.size(),
                           opDesc.outputDesc.data(), opDesc.opAttr,
                           ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr));

  void *host_data;
  size_t host_size = src.totalSize;
  aclrtMallocHost(&host_data, host_size);
  aclrtMemcpy(host_data, host_size, src.data, host_size,
              ACL_MEMCPY_DEVICE_TO_HOST);
  inputBuffers_host.emplace_back(aclCreateDataBuffer(host_data, host_size));
  inputBuffers_host.emplace_back(aclCreateDataBuffer(perm, size));

  AclSafeCall(aclopInferShape("Transpose", opDesc.inputDesc.size(),
                              opDesc.inputDesc.data(), inputBuffers_host.data(),
                              opDesc.outputDesc.size(),
                              opDesc.outputDesc.data(), opDesc.opAttr));
  outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));

  AclSafeCall(aclopExecuteV2(opDesc.opType.c_str(), inputBuffers_.size(),
                             opDesc.inputDesc.data(), inputBuffers_.data(),
                             outputBuffers_.size(), opDesc.outputDesc.data(),
                             outputBuffers_.data(), opDesc.opAttr,
                             dest.acl_context->get_stream(stream_id)));

  AclSafeCall(aclDestroyDataBuffer(inputBuffers_[0]));
  AclSafeCall(aclDestroyDataBuffer(inputBuffers_[1]));
  AclSafeCall(aclDestroyDataBuffer(inputBuffers_host[0]));
  AclSafeCall(aclDestroyDataBuffer(inputBuffers_host[1]));
  AclSafeCall(aclDestroyDataBuffer(outputBuffers_[0]));
  AclSafeCall(aclrtFree(dev));
  AclSafeCall(aclrtFreeHost(perm));
  AclSafeCall(aclrtFreeHost(host_data));
}

static int split_type(int depth) {
  switch (depth) {
    case CV_8U:
      return CV_8UC1;
    case CV_8S:
      return CV_8SC1;
    case CV_32F:
      return CV_32FC1;
    case CV_32S:
      return CV_32SC1;
    case CV_64F:
      return CV_64FC1;
    default:
      return -1;
  }
}

void split(const aclMat &src, vector<aclMat> &mv, int stream_id) {
  vector<aclDataBuffer *> inputBuffers_;
  vector<aclDataBuffer *> outputBuffers_;
  int split_dim = 3;
  int num_split = src.channels();

  OperatorDesc opDesc("SplitD");
  aclDataType dataType = type_transition(src.depth());

  int cols = src.step / src.elemSize();
  vector<int64_t> inputShape1{1, src.rows, cols, src.channels()};
  opDesc.AddInputTensorDesc(dataType, inputShape1.size(), inputShape1.data(),
                            ACL_FORMAT_ND);

  for (int i = 0; i < num_split; ++i) {
    vector<int64_t> outputShape{1, src.rows, cols, 1};
    opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(),
                               ACL_FORMAT_ND);
  }

  auto opAttr = opDesc.opAttr;
  aclopSetAttrInt(opAttr, "split_dim", split_dim);
  aclopSetAttrInt(opAttr, "num_split", num_split);

  inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));

  constexpr int false_type_flag = -1;
  int type = split_type(src.depth());
  CV_Assert(type != false_type_flag);
  for (int i = 0; i < num_split; ++i) {
    aclMat tmp(src.rows, src.cols, type, src.acl_context);
    mv[i] = tmp;
    outputBuffers_.emplace_back(
        aclCreateDataBuffer(mv[i].data, mv[i].totalSize));
  }

  compileAndRunop(opDesc, inputBuffers_, outputBuffers_, src.acl_context,
                  stream_id);

  AclSafeCall(aclDestroyDataBuffer(inputBuffers_[0]));
  for (int i = 0; i < num_split; ++i)
    AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));
}

static void flip_(const aclMat &src, aclMat &dest, int axis, int stream_id) {
  vector<aclDataBuffer *> inputBuffers_;
  vector<aclDataBuffer *> outputBuffers_;

  OperatorDesc opDesc("ReverseV2");
  aclDataType dataType = type_transition(src.depth());

  vector<int64_t> inputShape1{1, src.rows, src.cols, src.channels()};
  opDesc.AddInputTensorDesc(dataType, inputShape1.size(), inputShape1.data(),
                            ACL_FORMAT_ND);

  vector<int64_t> inputShape2{1};
  opDesc.AddInputTensorDesc(ACL_INT32, inputShape2.size(), inputShape2.data(),
                            ACL_FORMAT_ND);

  vector<int64_t> outputShape{1, dest.rows, dest.cols, dest.channels()};
  opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(),
                             ACL_FORMAT_ND);

  inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));

  void *dev;
  size_t size = aclGetTensorDescSize(opDesc.inputDesc[1]);
  aclrtMalloc(&dev, size, ACL_MEM_MALLOC_NORMAL_ONLY);
  aclrtMemcpy(dev, size, &axis, size, ACL_MEMCPY_HOST_TO_DEVICE);
  inputBuffers_.emplace_back(aclCreateDataBuffer(dev, size));

  outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));

  compileAndRunop(opDesc, inputBuffers_, outputBuffers_, dest.acl_context,
                  stream_id);

  AclSafeCall(aclDestroyDataBuffer(inputBuffers_[0]));
  AclSafeCall(aclDestroyDataBuffer(inputBuffers_[1]));
  AclSafeCall(aclDestroyDataBuffer(outputBuffers_[0]));
  AclSafeCall(aclrtFree(dev));
}

void flip(const aclMat &src, aclMat &dest, int filpCode, int stream_id) {
  constexpr int axis1 = 1;
  constexpr int axis2 = 2;
  if (filpCode == 0) {
    flip_(src, dest, axis1, stream_id);
  } else if (filpCode > 0) {
    flip_(src, dest, axis2, stream_id);
  } else {
    flip_(src, dest, axis2, stream_id);
    aclMat tmp(dest.rows, dest.cols, dest.type(), dest.acl_context);
    aclrtMemcpy(tmp.data, dest.totalSize, dest.data, dest.totalSize,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
    flip_(tmp, dest, axis1, stream_id);
  }
}
} /* end of namespace acl */
} /* end of namespace cv */