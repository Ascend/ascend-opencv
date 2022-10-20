/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::acl;
namespace cv {
namespace acl {
void merge(const vector<aclMat> &mv, aclMat &dest, int stream_id) {
  vector<aclDataBuffer *> inputBuffers_;
  vector<aclDataBuffer *> outputBuffers_;

  OperatorDesc opDesc("ConcatD");
  aclDataType dataType = type_transition(mv[0].depth());

  for (size_t i = 0; i < mv.size(); ++i) {
    int cols = mv[i].step / mv[i].elemSize();
    vector<int64_t> inputShape {1, mv[i].rows, cols, mv[i].channels()};
    opDesc.AddInputTensorDesc(dataType, inputShape.size(), inputShape.data(),
                              ACL_FORMAT_ND);
  }
  int cols = dest.step / dest.elemSize();
  vector<int64_t> outputShape {1, dest.rows, cols, dest.channels()};
  opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(),
                             ACL_FORMAT_ND);

  for (size_t i = 0; i < opDesc.inputDesc.size(); ++i) {
    inputBuffers_.emplace_back(
        aclCreateDataBuffer(mv[i].data, mv[i].totalSize));
  }
  outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));
  constexpr int c_dim = 3;
  aclopSetAttrInt(opDesc.opAttr, "N", mv.size());
  aclopSetAttrInt(opDesc.opAttr, "concat_dim", c_dim);

  compileAndRunop(opDesc, inputBuffers_, outputBuffers_, dest.acl_context,
                  stream_id);

  for (size_t i = 0; i < inputBuffers_.size(); i++)
    AclSafeCall(aclDestroyDataBuffer(inputBuffers_[i]));
  for (size_t i = 0; i < outputBuffers_.size(); i++)
    AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));
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

  vector<int64_t> inputShape1 {1, src.rows, src.cols, src.channels()};
  opDesc.AddInputTensorDesc(dataType, inputShape1.size(), inputShape1.data(),
                            ACL_FORMAT_ND);

  vector<int64_t> inputShape2 {4};
  opDesc.AddInputTensorDesc(ACL_INT32, inputShape2.size(), inputShape2.data(),
                            ACL_FORMAT_ND);

  vector<int64_t> outputShape {-1, -1, -1, -1};
  opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(),
                             ACL_FORMAT_ND);

  inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));

  void *dev;
  void *perm;
  constexpr int dim0_t = 0, dim1_t = 1;
  constexpr int dim2_t = 2, dim3_t = 3;
  constexpr int index0 = 0, index1 = 1;
  constexpr int index2 = 2, index3 = 3;

  size_t size = aclGetTensorDescSize(opDesc.inputDesc[1]);
  aclrtMalloc(&dev, size, ACL_MEM_MALLOC_NORMAL_ONLY);
  aclrtMallocHost(&perm, aclGetTensorDescSize(opDesc.inputDesc.data()[1]));
  ((int *)perm)[index0] = dim0_t;
  ((int *)perm)[index1] = dim2_t;
  ((int *)perm)[index2] = dim1_t;
  ((int *)perm)[index3] = dim3_t;
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
  vector<int64_t> inputShape1 {1, src.rows, cols, src.channels()};
  opDesc.AddInputTensorDesc(dataType, inputShape1.size(), inputShape1.data(),
                            ACL_FORMAT_ND);

  for (int i = 0; i < num_split; ++i) {
    vector<int64_t> outputShape {1, src.rows, cols, 1};
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

  vector<int64_t> inputShape1 {1, src.rows, src.cols, src.channels()};
  opDesc.AddInputTensorDesc(dataType, inputShape1.size(), inputShape1.data(),
                            ACL_FORMAT_ND);

  vector<int64_t> inputShape2 {1};
  opDesc.AddInputTensorDesc(ACL_INT32, inputShape2.size(), inputShape2.data(),
                            ACL_FORMAT_ND);

  vector<int64_t> outputShape {1, dest.rows, dest.cols, dest.channels()};
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