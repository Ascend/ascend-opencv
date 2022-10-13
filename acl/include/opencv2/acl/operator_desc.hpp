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
#ifndef OPENCV_OPERATOR_DESC_HPP
#define OPENCV_OPERATOR_DESC_HPP

#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl_init.hpp"
#include "acl_mat.hpp"
#include "acl_type.hpp"

namespace cv {
namespace acl {
class CV_EXPORTS OperatorDesc {
 public:
  /**
   * Constructor
   * @param [in] opType: op type
   */
  OperatorDesc(std::string opType);

  /**
   * Destructor
   */
  virtual ~OperatorDesc();

  /**
   * Add an input tensor description
   * @param [in] dataType: data type
   * @param [in] numDims: number of dims
   * @param [in] dims: dims
   * @param [in] format: format
   * @return OperatorDesc
   */
  OperatorDesc &AddInputTensorDesc(aclDataType dataType, int numDims,
                                   const int64_t *dims, aclFormat format);

  /**
   * Add an output tensor description
   * @param [in] dataType: data type
   * @param [in] numDims: number of dims
   * @param [in] dims: dims
   * @param [in] format: format
   * @return OperatorDesc
   */
  OperatorDesc &AddOutputTensorDesc(aclDataType dataType, int numDims,
                                    const int64_t *dims, aclFormat format);

  template <typename T>
  bool AddTensorAttr(const char *attrName, AttrType type, T vaule) {
    if (opAttr == nullptr) return false;
    switch (type) {
      case OP_BOOL:
        aclopSetAttrBool(opAttr, attrName, vaule);
        break;
      case OP_INT:
        aclopSetAttrInt(opAttr, attrName, vaule);
        break;
      case OP_FLOAT:
        aclopSetAttrFloat(opAttr, attrName, vaule);
        break;
      default:
        break;
    }
    return true;
  }
  std::string opType;
  std::vector<aclTensorDesc *> inputDesc;
  std::vector<aclTensorDesc *> outputDesc;
  aclopAttr *opAttr;
};

// Create operator description
CV_EXPORTS OperatorDesc CreateOpDesc(const std::string opType,
                                     const std::vector<aclMat> &input_Mat,
                                     std::vector<aclMat> &output_Mat,
                                     aclFormat format = ACL_FORMAT_NHWC,
                                     Opdims config = FOUR_DIMS);
// Compile and run the operator
CV_EXPORTS void compileAndRunop(OperatorDesc &opDesc,
                                std::vector<aclDataBuffer *> &inputBuffers_,
                                std::vector<aclDataBuffer *> &outputBuffers_,
                                aclCxt *acl_context, int stream_id);
// Suitable for one input and one output
CV_EXPORTS void OneInAndOneOut(const aclMat &input, aclMat &output,
                               const std::string opType, int stream_id = 0);
// Suitable for tow input and one output
CV_EXPORTS void TwoInAndOneOut(const aclMat &inputMat,
                               const aclMat &inputMatOther, aclMat &outputMat,
                               const std::string opType, int stream_id = 0);
// run the operator
CV_EXPORTS void Runop(std::vector<aclMat> &input, std::vector<aclMat> &output,
                      OperatorDesc &opDesc, int stream_id);
} /* end of namespace acl */
} /* end of namespace cv */

#endif  // OPERATOR_DESC_HPP
