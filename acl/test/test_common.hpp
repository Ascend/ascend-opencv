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
#ifndef __OPENCV_TEST_COMMON_HPP__
#define __OPENCV_TEST_COMMON_HPP__

#include "test_precomp.hpp"

using TestDatatype = enum TestDatatype { INT = 1, FLOAT };

class CV_EXPORTS Common_Test {
public:
  Common_Test();
  ~Common_Test();
  CV_EXPORTS bool Test_Diff(
      const cv::acl::aclMat& aclmat, const cv::Mat& mat,
      cv::acl::ALIGNMENT config = cv::acl::ALIGNMENT::MEMORY_UNALIGNED);
  CV_EXPORTS bool Test_Diff(const cv::acl::aclMat& aclmat,
                            const cv::acl::aclMat& aclmat_other);
  CV_EXPORTS bool Test_Diff(const cv::Mat& mat, const cv::Mat& mat_other);
  CV_EXPORTS void MatShow(cv::Mat& m, std::string str);
  CV_EXPORTS void StatShow(cv::Mat& mat_src, cv::acl::aclMat& aclmat_dst);
  CV_EXPORTS void PrintLog(const std::string& funcname, int type);

  CV_EXPORTS size_t RandDom_(int config = 0xff);
  CV_EXPORTS bool SetDataRange(cv::Mat& src, int dataRange = 0xff);
};

#endif