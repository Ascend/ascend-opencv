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
#ifndef OPENCV_GEMM_HPP
#define OPENCV_GEMM_HPP

#include "acl_mat.hpp"

namespace cv {
namespace acl {
// matrix multiplication
CV_EXPORTS void MatMul(const aclMat &src1, const aclMat &src2, aclMat &dest,
                       int stream_id = 0);
// convolution
CV_EXPORTS void Convolution(
    const aclMat &src, const aclMat &kernel, aclMat &dest,
    const std::vector<int64_t> &stridesList = std::vector<int64_t> {1, 1, 1, 1},
    const std::vector<int64_t> &padsList = std::vector<int64_t> {0, 0, 0, 0},
    int stream_id = 0);
} /* end of namespace acl */
} /* end of namespace cv */

#endif