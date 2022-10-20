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
#ifndef OPENCV_MATRICES_HPP
#define OPENCV_MATRICES_HPP

#include "acl_mat.hpp"

namespace cv {
namespace acl {
// Multiple channel merge
CV_EXPORTS void merge(const std::vector<aclMat> &mv, aclMat &dst, int stream_id = 0);
// Split into channels
CV_EXPORTS void split(const aclMat &src, std::vector<aclMat> &mv, int stream_id = 0);
// Matrix transpose
CV_EXPORTS void transpose(const aclMat &src, aclMat &dest, int stream_id = 0);
CV_EXPORTS void flip(const aclMat &src, aclMat &dest, int flipCode = 0,
                     int stream_id = 0);
} /* end of namespace acl */
} /* end of namespace cv */

#endif