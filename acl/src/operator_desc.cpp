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
OperatorDesc::OperatorDesc(std::string opType) : opType(std::move(opType)) {
  opAttr = aclopCreateAttr();
}

OperatorDesc::~OperatorDesc() {
  for (auto* desc : inputDesc) {
    aclDestroyTensorDesc(desc);
  }

  for (auto* desc : outputDesc) {
    aclDestroyTensorDesc(desc);
  }

  aclopDestroyAttr(opAttr);
}

OperatorDesc& OperatorDesc::AddInputTensorDesc(aclDataType dataType,
                                               int numDims, const int64_t* dims,
                                               aclFormat format) {
  aclTensorDesc* desc = aclCreateTensorDesc(dataType, numDims, dims, format);
  CV_Assert(desc);
  inputDesc.emplace_back(desc);
  return *this;
}

OperatorDesc& OperatorDesc::AddOutputTensorDesc(aclDataType dataType,
                                                int numDims,
                                                const int64_t* dims,
                                                aclFormat format) {
  aclTensorDesc* desc = aclCreateTensorDesc(dataType, numDims, dims, format);
  CV_Assert(desc);
  outputDesc.emplace_back(desc);
  return *this;
}

/**
 * @brief create operator describe
 *
 */
OperatorDesc CreateOpDesc(const string opType, const vector<aclMat>& input_Mat,
                          vector<aclMat>& output_Mat, aclFormat format,
                          Opdims config) {
  CV_Assert(config == TWO_DIMS || config == FOUR_DIMS);

  size_t i;
  aclDataType dataType = type_transition(input_Mat[0].depth());

  OperatorDesc opDesc(opType);
  for (i = 0; i < input_Mat.size(); ++i) {
    if (config == TWO_DIMS) {
      int cols = input_Mat[i].step / input_Mat[i].elemSize();
      vector<int64_t> shape {input_Mat[i].rows, cols};
      opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
    } else if (config == FOUR_DIMS) {
      int cols = input_Mat[i].step / input_Mat[i].elemSize();
      vector<int64_t> shape {1, input_Mat[i].rows, cols,
                            input_Mat[i].channels()};
      opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
    }
  }

  for (i = 0; i < output_Mat.size(); ++i) {
    if (config == TWO_DIMS) {
      int cols = output_Mat[i].step / output_Mat[i].elemSize();
      vector<int64_t> shape {output_Mat[i].rows, cols};
      opDesc.AddOutputTensorDesc(dataType, shape.size(), shape.data(), format);
    } else if (config == FOUR_DIMS) {
      int cols = output_Mat[i].step / output_Mat[i].elemSize();
      vector<int64_t> shape {1, output_Mat[i].rows, cols,
                            output_Mat[i].channels()};
      opDesc.AddOutputTensorDesc(dataType, shape.size(), shape.data(), format);
    }
  }

  return opDesc;
}

/**
 * @brief compile and run operator
 *
 */
void compileAndRunop(OperatorDesc& opDesc,
                     vector<aclDataBuffer*>& inputBuffers_,
                     vector<aclDataBuffer*>& outputBuffers_,
                     aclCxt* acl_context, int stream_id) {
  AclSafeCall(aclopCompile(opDesc.opType.c_str(), opDesc.inputDesc.size(),
                           opDesc.inputDesc.data(), opDesc.outputDesc.size(),
                           opDesc.outputDesc.data(), opDesc.opAttr,
                           ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr));

  AclSafeCall(aclopExecuteV2(opDesc.opType.c_str(), inputBuffers_.size(),
                             opDesc.inputDesc.data(), inputBuffers_.data(),
                             outputBuffers_.size(), opDesc.outputDesc.data(),
                             outputBuffers_.data(), opDesc.opAttr,
                             acl_context->get_stream(stream_id)));
}

void Runop(vector<aclMat>& input, vector<aclMat>& output, OperatorDesc& opDesc,
           int stream_id) {
  size_t i;

  vector<aclDataBuffer*> inputBuffers_;
  vector<aclDataBuffer*> outputBuffers_;

  for (i = 0; i < input.size(); ++i)
    inputBuffers_.emplace_back(
        aclCreateDataBuffer(input[i].data, input[i].totalSize));
  for (i = 0; i < output.size(); ++i)
    outputBuffers_.emplace_back(
        aclCreateDataBuffer(output[i].data, output[i].totalSize));

  compileAndRunop(opDesc, inputBuffers_, outputBuffers_, output[0].acl_context,
                  stream_id);

  for (i = 0; i < input.size(); ++i)
    AclSafeCall(aclDestroyDataBuffer(inputBuffers_[i]));
  for (i = 0; i < output.size(); ++i)
    AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));
}

void OneInAndOneOut(const aclMat& inputMat, aclMat& outputMat,
                    const string opType, int stream_id) {
  vector<aclMat> input_Mat;
  vector<aclMat> output_Mat;

  input_Mat.emplace_back(inputMat);
  output_Mat.emplace_back(outputMat);

  OperatorDesc opDesc = CreateOpDesc(opType, input_Mat, output_Mat);
  Runop(input_Mat, output_Mat, opDesc, stream_id);
}

void TwoInAndOneOut(const aclMat& inputMat, const aclMat& inputMatOther,
                    aclMat& outputMat, const string opType, int stream_id) {
  vector<aclMat> input_Mat;
  vector<aclMat> output_Mat;

  input_Mat.emplace_back(inputMat);
  input_Mat.emplace_back(inputMatOther);
  output_Mat.emplace_back(outputMat);

  OperatorDesc opDesc = CreateOpDesc(opType, input_Mat, output_Mat);
  Runop(input_Mat, output_Mat, opDesc, stream_id);
}
} /* end of namespace acl */
} /* end of namespace cv */
