#include "test_common.hpp"
#include "test_perf.hpp"

using namespace cv;
using namespace cv::acl;
using namespace cvtest;
using namespace testing;
using namespace std;

void PERF_TEST::Test_operator_add_perf(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  constexpr int start_val = 8;
  constexpr int rand_data_range = 1;
  constexpr int min_format_flag = 128;

  vector<int> type{CV_8UC1, CV_32FC1, CV_32SC1, CV_64FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Perf test : Function: operator+=()", type[i]);
    for (val = start_val; val <= valmax; val *= 2) {
      n = cycle_index;
      Mat mat_src(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src, rand_data_range);
      test.SetDataRange(mat_dest, rand_data_range);

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) mat_dest += mat_src;
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      aclmat_dest += aclmat_src;
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) aclmat_dest += aclmat_src;
      wait_stream(acl_context);
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

void PERF_TEST::Test_operator_sub_perf(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  constexpr int start_val = 8;
  constexpr int rand_data_range1 = 4;
  constexpr int rand_data_range2 = 32;
  constexpr int min_format_flag = 128;

  vector<int> type{CV_8UC1, CV_32FC1, CV_32SC1, CV_64FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Perf test : Function: operator-=()", type[i]);
    for (val = start_val; val <= valmax; val *= 2) {
      n = cycle_index;
      Mat mat_src(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src, rand_data_range1);
      test.SetDataRange(mat_dest, rand_data_range2);

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) mat_dest -= mat_src;
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      aclmat_dest -= aclmat_src;
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) aclmat_dest -= aclmat_src;
      wait_stream(acl_context);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      if (val < min_format_flag)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_operator_div_perf(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  constexpr int start_val = 8;
  constexpr int s_val1 = 1;
  constexpr int s_val2 = 2;
  constexpr int s_val4 = 4;
  constexpr int s_val8 = 8;
  constexpr int min_format_flag = 128;

  vector<int> type{CV_8UC1, CV_32FC1, CV_32SC1, CV_64FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Perf test : Function: operator/=()", type[i]);
    for (val = start_val; val <= valmax; val *= 2) {
      n = cycle_index;
      Mat mat_src(val, val, type[i], Scalar(s_val1, s_val2, s_val4));
      Mat mat_dest(val, val, type[i], Scalar(s_val2, s_val4, s_val8));
      Mat mat_dest1(val, val, type[i]);

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) mat_dest /= mat_src;
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      aclmat_dest /= aclmat_src;
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) aclmat_dest /= aclmat_src;
      wait_stream(acl_context);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      if (val < min_format_flag)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_operator_mul_perf(aclCxt *acl_context) {
  int val, n;
  int valmax = 4096;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  constexpr int start_val = 8;
  constexpr int rand_data_range = 1;
  constexpr int min_format_flag = 128;

  vector<int> type{CV_32FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    for (val = start_val; val <= valmax; val *= 2) {
      n = cycle_index;
      Mat mat_src(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src, rand_data_range);
      test.SetDataRange(mat_dest, rand_data_range);

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) mat_dest *= mat_src;
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      aclmat_dest *= aclmat_src;
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) aclmat_dest *= aclmat_src;
      wait_stream(acl_context);
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
