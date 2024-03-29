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
#include "test_common.hpp"
#include "test_perf.hpp"

using namespace cv;
using namespace cv::acl;
using namespace cvtest;
using namespace testing;
using namespace std;

void PERF_TEST::Test_Merge(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  constexpr int base = 2;
  constexpr int start_val = 8;
  constexpr int rand_data_range = 32;
  constexpr int min_format_flag = 128;

  vector<int> srcType {CV_8UC1, CV_32FC1, CV_32SC1};
  vector<int> destType {CV_8UC3, CV_32FC3, CV_32SC3};

  for (size_t i = 0; i < srcType.size(); ++i) {
    test.PrintLog("Perf test : Function: merge()", srcType[i]);
    for (val = start_val; val <= valmax; val *= base) {
      n = cycle_index;
      Mat mat_src1(val, val, srcType[i]);
      Mat mat_src2(val, val, srcType[i]);
      Mat mat_src3(val, val, srcType[i]);
      Mat mat_dest(val, val, destType[i]);
      Mat mat_dest1(val, val, destType[i]);

      test.SetDataRange(mat_src1, rand_data_range);
      test.SetDataRange(mat_src2, rand_data_range);
      test.SetDataRange(mat_src3, rand_data_range);

      aclMat aclmat_src1(val, val, srcType[i], mat_src1.data, acl_context);
      aclMat aclmat_src2(val, val, srcType[i], mat_src2.data, acl_context);
      aclMat aclmat_src3(val, val, srcType[i], mat_src3.data, acl_context);
      aclMat aclmat_dest(val, val, destType[i], mat_dest.data, acl_context);

      vector<Mat> src;
      src.emplace_back(mat_src1);
      src.emplace_back(mat_src2);
      src.emplace_back(mat_src3);

      vector<aclMat> acl_src;
      acl_src.emplace_back(aclmat_src1);
      acl_src.emplace_back(aclmat_src2);
      acl_src.emplace_back(aclmat_src3);

      begin = static_cast<double>(getTickCount());
      while (n--) merge(src, mat_dest);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      merge(acl_src, aclmat_dest);
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) merge(acl_src, aclmat_dest, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);
      aclmat_dest.download(mat_dest1);
      bool ret = test.Test_Diff(mat_dest, mat_dest1);
      ASSERT_TRUE(ret);
      if (val < min_format_flag)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_Transpose(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  constexpr int base = 2;
  constexpr int start_val = 8;
  constexpr int rand_data_range = 32;
  constexpr int min_format_flag = 128;

  vector<int> type {CV_32FC1, CV_32SC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Perf test : Function: transpose()", type[i]);
    for (val = start_val; val <= valmax; val *= base) {
      n = cycle_index;
      Mat mat_src(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src, rand_data_range);

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) transpose(mat_src, mat_dest);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      transpose(aclmat_src, aclmat_dest);
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) transpose(aclmat_src, aclmat_dest, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      bool ret = test.Test_Diff(mat_dest, mat_dest1);
      ASSERT_TRUE(ret);
      if (val < min_format_flag)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_Split(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  constexpr int base = 2;
  constexpr int start_val = 8;
  constexpr int rand_data_range = 32;
  constexpr int min_format_flag = 128;
  constexpr int index0 = 0, index1 = 1, index2 = 2;

  vector<int> srcType {CV_8UC3, CV_32FC3, CV_32SC3};
  vector<int> destType {CV_8UC1, CV_32FC1, CV_32SC1};
  for (size_t i = 0; i < srcType.size(); ++i) {
    test.PrintLog("Perf test : Function: split()", srcType[i]);
    for (val = start_val; val <= valmax; val *= base) {
      n = cycle_index;
      Mat mat_src(val, val, srcType[i]);
      Mat mat_dest1(val, val, destType[i]);
      Mat mat_dest2(val, val, destType[i]);
      Mat mat_dest3(val, val, destType[i]);

      test.SetDataRange(mat_src, rand_data_range);

      aclMat aclmat_src(val, val, srcType[i], mat_src.data, acl_context);
      aclMat aclmat_dest1;
      aclMat aclmat_dest2;
      aclMat aclmat_dest3;

      vector<Mat> dest;
      dest.emplace_back(mat_dest1);
      dest.emplace_back(mat_dest2);
      dest.emplace_back(mat_dest3);

      vector<aclMat> acl_dest;
      acl_dest.emplace_back(aclmat_dest1);
      acl_dest.emplace_back(aclmat_dest2);
      acl_dest.emplace_back(aclmat_dest3);

      begin = static_cast<double>(getTickCount());
      while (n--) split(mat_src, dest);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      split(aclmat_src, acl_dest);
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) split(aclmat_src, acl_dest, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      (acl_dest.data())[index0].download(mat_dest1);
      (acl_dest.data())[index1].download(mat_dest2);
      (acl_dest.data())[index2].download(mat_dest3);

      bool ret = test.Test_Diff((dest.data())[index0], mat_dest1);
      ret &= test.Test_Diff((dest.data())[index1], mat_dest2);
      ret &= test.Test_Diff((dest.data())[index2], mat_dest3);
      ASSERT_TRUE(ret);
      if (val < min_format_flag)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_Flip(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 100;
  double begin, end, time, acltime;
  Common_Test test;
  constexpr int base = 2;
  constexpr int start_val = 8;
  constexpr int rand_data_range = 32;
  constexpr int min_format_flag = 128;

  vector<int> type {CV_8UC1, CV_32FC1, CV_32SC1, CV_64FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Perf test : Function: flip()", type[i]);
    for (val = start_val; val <= valmax; val *= base) {
      n = cycle_index;
      Mat mat_src(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src, rand_data_range);

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) flip(mat_src, mat_dest, 0);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      flip(aclmat_src, aclmat_dest, 0);
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) flip(aclmat_src, aclmat_dest, 0, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      bool ret = test.Test_Diff(mat_dest, mat_dest1);
      ASSERT_TRUE(ret);
      if (val < min_format_flag)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}