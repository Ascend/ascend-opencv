#include "test_common.hpp"
#include "test_perf.hpp"

using namespace cv;
using namespace cv::acl;
using namespace cvtest;
using namespace testing;
using namespace std;

void PERF_TEST::Test_Abs(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;

  // vector<int> type{CV_32FC1};
  vector<int> type{CV_32FC1, CV_32SC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Perf test : Function: Abs()", type[i]);
    for (val = 8; val <= valmax; val *= 2) {
      n = cycle_index;
      Mat mat_src(val, val, type[i], Scalar{-2});
      Mat mat_dest(val, val, type[i], Scalar{-4});
      Mat mat_dest1(val, val, type[i], Scalar{-6});

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) mat_dest = abs(mat_src);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      aclmat_dest = abs(aclmat_src, 0);
      wait_stream(acl_context, 0);
      begin = static_cast<double>(getTickCount());
      while (n--) aclmat_dest = abs(aclmat_src, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      bool ret = test.Test_Diff(mat_dest, mat_dest1);
      ASSERT_TRUE(ret);
      if (val < 128)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_Pow(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;

  // vector<int> type{CV_32FC1};
  vector<int> type{CV_8UC1, CV_32FC1, CV_32SC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Perf test : Function: Pow()", type[i]);
    for (val = 8; val <= valmax; val *= 2) {
      n = cycle_index;
      int power = test.RandDom_(6);
      Mat mat_src(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src, 32);

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) pow(mat_src, power, mat_dest);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      pow(aclmat_src, power, aclmat_dest, 0);
      wait_stream(acl_context, 0);
      begin = static_cast<double>(getTickCount());
      while (n--) pow(aclmat_src, power, aclmat_dest, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      if (val < 128)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_Sqrt(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  // vector<int> type{CV_32FC1};
  vector<int> type{CV_32FC1, CV_64FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    for (val = 8; val <= valmax; val *= 2) {
      n = cycle_index;
      Mat mat_src(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src, 32);
      test.SetDataRange(mat_dest, 32);

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) sqrt(mat_src, mat_dest);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      sqrt(aclmat_src, aclmat_dest);
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) sqrt(aclmat_src, aclmat_dest, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      if (val < 128)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_Add(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  // vector<int> type{CV_32FC1};
  vector<int> type{CV_8UC1, CV_32FC1, CV_32SC1, CV_64FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    for (val = 8; val <= valmax; val *= 2) {
      n = cycle_index;
      Mat mat_src1(val, val, type[i]);
      Mat mat_src2(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src1, 32);
      test.SetDataRange(mat_src2, 32);
      test.SetDataRange(mat_dest, 32);

      aclMat aclmat_src1(val, val, type[i], mat_src1.data, acl_context);
      aclMat aclmat_src2(val, val, type[i], mat_src2.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) add(mat_src1, mat_src2, mat_dest);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      add(aclmat_src1, aclmat_src2, aclmat_dest);
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) add(aclmat_src1, aclmat_src2, aclmat_dest, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      bool ret = test.Test_Diff(mat_dest, mat_dest1);
      ASSERT_TRUE(ret);
      if (val < 128)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_Divide(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  // vector<int> type{CV_32FC1};
  vector<int> type{CV_8UC1, CV_32FC1, CV_32SC1};
  for (size_t i = 0; i < type.size(); ++i) {
    for (val = 8; val <= valmax; val *= 2) {
      n = cycle_index;
      Mat mat_src1(val, val, type[i]);
      Mat mat_src2(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src1, 32);
      test.SetDataRange(mat_src2, 4);
      test.SetDataRange(mat_dest, 32);

      aclMat aclmat_src1(val, val, type[i], mat_src1.data, acl_context);
      aclMat aclmat_src2(val, val, type[i], mat_src2.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) divide(mat_src1, mat_src2, mat_dest);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      divide(aclmat_src1, aclmat_src2, aclmat_dest);
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) divide(aclmat_src1, aclmat_src2, aclmat_dest, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      // bool ret = test.Test_Diff(mat_dest, mat_dest1);
      // ASSERT_TRUE(ret);
      if (val < 128)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_Exp(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;

  // vector<int> type{CV_32FC1};
  vector<int> type{CV_32FC1, CV_64FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    for (val = 8; val <= valmax; val *= 2) {
      n = cycle_index;
      Mat mat_src(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src, 32);
      test.SetDataRange(mat_dest, 2);

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) exp(mat_src, mat_dest);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      exp(aclmat_src, aclmat_dest);
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) exp(aclmat_src, aclmat_dest, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      if (val < 128)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_Log(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  // vector<int> type{CV_32FC1};
  vector<int> type{CV_32FC1, CV_64FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    for (val = 8; val <= valmax; val *= 2) {
      n = cycle_index;
      Mat mat_src(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src, 32);
      test.SetDataRange(mat_dest, 32);

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) log(mat_src, mat_dest);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      log(aclmat_src, aclmat_dest, 1);
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) log(aclmat_src, aclmat_dest, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      if (val < 128)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_Max(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  // vector<int> type{CV_32FC2};
  vector<int> type{CV_32FC2, CV_32SC2, CV_64FC2};
  for (size_t i = 0; i < type.size(); ++i) {
    for (val = 8; val <= valmax; val *= 2) {
      n = cycle_index;
      Mat mat_src1(val, val, type[i]);
      Mat mat_src2(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src1, 32);
      test.SetDataRange(mat_src2, 32);
      test.SetDataRange(mat_dest, 32);

      aclMat aclmat_src1(val, val, type[i], mat_src2.data, acl_context);
      aclMat aclmat_src2(val, val, type[i], mat_src1.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) cv::max(mat_src1, mat_src2, mat_dest);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      cv::acl::max(aclmat_src1, aclmat_src2, aclmat_dest);
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) cv::acl::max(aclmat_src1, aclmat_src2, aclmat_dest, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      bool ret = test.Test_Diff(mat_dest, mat_dest1);
      ASSERT_TRUE(ret);
      if (val < 128)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}

void PERF_TEST::Test_Min(aclCxt *acl_context) {
  int val, n;
  int valmax = 8192;
  int cycle_index = 10;
  double begin, end, time, acltime;
  Common_Test test;
  // vector<int> type{CV_32FC3};
  vector<int> type{CV_32FC3, CV_32SC3, CV_64FC3};
  for (size_t i = 0; i < type.size(); ++i) {
    for (val = 8; val <= valmax; val *= 2) {
      n = cycle_index;
      Mat mat_src1(val, val, type[i]);
      Mat mat_src2(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src1, 32);
      test.SetDataRange(mat_src2, 32);
      test.SetDataRange(mat_dest, 32);

      aclMat aclmat_src1(val, val, type[i], mat_src2.data, acl_context);
      aclMat aclmat_src2(val, val, type[i], mat_src1.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      begin = static_cast<double>(getTickCount());
      while (n--) cv::min(mat_src1, mat_src2, mat_dest);
      end = static_cast<double>(getTickCount());
      time = (end - begin) / getTickFrequency() / cycle_index;

      n = (cycle_index - 1);
      cv::acl::min(aclmat_src1, aclmat_src2, aclmat_dest);
      wait_stream(acl_context);
      begin = static_cast<double>(getTickCount());
      while (n--) cv::acl::min(aclmat_src1, aclmat_src2, aclmat_dest, 1);
      wait_stream(acl_context, 1);
      end = static_cast<double>(getTickCount());
      acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

      aclmat_dest.download(mat_dest1);
      bool ret = test.Test_Diff(mat_dest, mat_dest1);
      ASSERT_TRUE(ret);
      if (val < 128)
        cout << "Shape: " << val << " x " << val << "\t\t";
      else
        cout << "Shape: " << val << " x " << val << "\t";
      cout << "CpuTimes: " << time << "\tAclTimes: " << acltime
           << "\tRate: " << time / acltime << endl;
    }
  }
}