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
#ifndef OPENCV_MATHFUNCS_HPP
#define OPENCV_MATHFUNCS_HPP

#include "acl_mat.hpp"

/**
 *   mathfunctions;
 */

namespace cv {
namespace acl {
CV_EXPORTS aclMat abs(const aclMat &src, int stream_id = 0);
CV_EXPORTS void pow(const aclMat &src, double power, aclMat &dest,
                    int stream_id = 0);
CV_EXPORTS void sqrt(const aclMat &src, aclMat &dest, int stream_id = 0);
CV_EXPORTS void add(const aclMat &src, const aclMat &other_src, aclMat &dest,
                    int stream_id = 0);
CV_EXPORTS void divide(const aclMat &src, const aclMat &other_src, aclMat &dest,
                       int stream_id = 0);
CV_EXPORTS void exp(const aclMat &src, aclMat &dest, int stream_id = 0);
CV_EXPORTS void log(const aclMat &src, aclMat &dest, int stream_id = 0);
CV_EXPORTS void max(const aclMat &src, const aclMat &other_src, aclMat &dest,
                    int stream_id = 0);
CV_EXPORTS void min(const aclMat &src, const aclMat &other_src, aclMat &dest,
                    int stream_id = 0);
} /* end of namespace acl */
} /* end of namespace cv */

#endif