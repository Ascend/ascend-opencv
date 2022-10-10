#include "test_common.hpp"
#include "test_perf.hpp"

using namespace cv;
using namespace cv::acl;
using namespace cvtest;
using namespace testing;
using namespace std;

void PERF_TEST::Test_MatMul(aclCxt *acl_context) {
  int val, n;
  int valmax = 4096;
  int cycle_index = 10;  // 100;
  double begin, end, time, acltime;
  Common_Test test;
  vector<int> type{CV_32FC1};
  constexpr int base = 2;
  constexpr int start_val = 8;
  constexpr int rand_data_range = 32;
  constexpr int min_format_flag = 128;

  for (size_t i = 0; i < type.size(); ++i) {
    for (val = start_val; val <= valmax; val *= base) {
      Mat mat_src(val, val, type[i]);
      Mat mat_src1(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src, rand_data_range);
      test.SetDataRange(mat_src1, rand_data_range);
      test.SetDataRange(mat_dest, rand_data_range);

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_src1(val, val, type[i], mat_src1.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      n = cycle_index;
      begin = static_cast<double>(getTickCount());
      while (n--) mat_dest = mat_src * mat_src1;
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      MatMul(aclmat_src1, aclmat_src, aclmat_dest, 0);
      wait_stream(acl_context, 0);
      begin = static_cast<double>(getTickCount());
      while (n--) MatMul(aclmat_src1, aclmat_src, aclmat_dest, 1);
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

void PERF_TEST::Test_Convolution(aclCxt *acl_context) {
  int val, n;
  int valmax = 4096;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  vector<int> type{CV_32FC1};
  constexpr int base = 2;
  constexpr int start_val = 8;
  constexpr int min_format_flag = 128;
  constexpr int s_val1 = 1, s_val2 = 2;
  constexpr int s_val4 = 4, s_val6 = 6;
  constexpr int kernel_val = 3;

  for (size_t i = 0; i < type.size(); ++i) {
    for (val = start_val; val <= valmax; val *= base) {
      Mat mat_src(val, val, type[i], Scalar{s_val1, s_val2});
      Mat mat_kernel(kernel_val, kernel_val, type[i], Scalar(s_val1, s_val4));
      Mat mat_dest(val, val, type[i], Scalar{s_val6});

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_kernel(kernel_val, kernel_val, type[i], mat_kernel.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      n = cycle_index;
      begin = static_cast<double>(getTickCount());
      while (n--) filter2D(mat_src, mat_dest, -1, mat_kernel);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      vector<int64_t> strides{1, 1, 1, 1};
      vector<int64_t> pads{1, 1, 1, 1};
      n = (cycle_index - 1);
      Convolution(aclmat_src, aclmat_kernel, aclmat_dest, strides, pads, 0);
      wait_stream(acl_context, 0);
      begin = static_cast<double>(getTickCount());
      while (n--)
        Convolution(aclmat_src, aclmat_kernel, aclmat_dest, strides, pads, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      Mat mat_dest1(aclmat_dest.rows, aclmat_dest.cols, type[i]);
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