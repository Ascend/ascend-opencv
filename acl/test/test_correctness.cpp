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
#include "test_correctness.hpp"

#include "test_common.hpp"

using namespace cv;
using namespace cv::acl;
using namespace cvtest;
using namespace testing;
using namespace std;

AclMat_Test::AclMat_Test() {}

AclMat_Test::~AclMat_Test() {}

/* thread function */
void thread_handler(void) {
  aclCxt *acl_context_0 =
      set_device("/home/perfxlab4/OpenCV_ACL/modules/acl/test/acl.json", 0, 1);
  release_device(acl_context_0);
}

void AclMat_Test::Test_set_device() {
  /* Current thread */
  aclCxt *acl_context_0 =
      set_device("/home/perfxlab4/OpenCV_ACL/modules/acl/test/acl.json", 0, 1);

  /* Different scope */
  {
    aclCxt *acl_context_1 = set_device(
        "/home/perfxlab4/OpenCV_ACL/modules/acl/test/acl.json", 2, 3);
    release_device(acl_context_1);
  }

  release_device(acl_context_0);
  /* Different thread */
  thread t(thread_handler);
  t.join();
}

void AclMat_Test::Test_constructor_UNALIGNED(aclCxt *acl_context) {
  Common_Test test;
  int rows, cols, type;
  bool ret;
  constexpr int rand_data_range = 32;
  const int rowsMax = 128;
  const int colsMax = 128;
  const int typeMax = 7;

  for (type = 0; type < typeMax; type++) {
    for (rows = 1; rows < rowsMax; rows++) {
      for (cols = 1; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type);
        aclMat aclmat_src(rows, cols, type, acl_context);
        test.SetDataRange(mat_src, rand_data_range);
        aclmat_src.upload(mat_src);
        ret = test.Test_Diff(aclmat_src, mat_src);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_constructor_UNALIGNED: -> aclMat(rows, cols, type, "
          "acl_context, config, policy) <- is success"
       << endl;

  for (type = 0; type < typeMax; type++) {
    for (rows = 1; rows < rowsMax; rows++) {
      for (cols = 1; cols < colsMax; cols++) {
        Mat mat_src(cv::Size(cols, rows), type);
        test.SetDataRange(mat_src, rand_data_range);
        aclMat aclmat_src(cv::Size(cols, rows), type, acl_context);
        aclmat_src.upload(mat_src);
        ret = test.Test_Diff(aclmat_src, mat_src);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_constructor_UNALIGNED: -> aclMat(size, type, acl_context, "
          "config, policy) <- is success"
       << endl;
}

void AclMat_Test::Test_constructor_ALIGN(aclCxt *acl_context) {
  Common_Test test;
  int rows, cols, type;
  bool ret;
  constexpr int rand_data_range = 32;
  const int rowsMax = 128;
  const int colsMax = 128;
  const int typeMax = 7;

  for (type = 0; type < typeMax; type++) {
    for (rows = 1; rows < rowsMax; rows++) {
      for (cols = 1; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type);
        test.SetDataRange(mat_src, rand_data_range);
        aclMat aclmat_src(rows, cols, type, acl_context, MEMORY_ALIGN);
        aclmat_src.upload(mat_src, MEMORY_ALIGN);
        ret = test.Test_Diff(aclmat_src, mat_src, MEMORY_ALIGN);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_constructor_ALIGN: -> aclMat(rows, cols, type, acl_context, "
          "config, policy) <- is success"
       << endl;

  for (type = 0; type < typeMax; type++) {
    for (rows = 1; rows < rowsMax; rows++) {
      for (cols = 1; cols < colsMax; cols++) {
        Mat mat_src(cv::Size(cols, rows), type);
        test.SetDataRange(mat_src, rand_data_range);
        aclMat aclmat_src(cv::Size(cols, rows), type, acl_context,
                          MEMORY_ALIGN);
        aclmat_src.upload(mat_src, MEMORY_ALIGN);
        ret = test.Test_Diff(aclmat_src, mat_src, MEMORY_ALIGN);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_constructor_ALIGN: -> aclMat(size, type, acl_context, config, "
          "policy) <- is success"
       << endl;
}

void AclMat_Test::Test_constructor(aclCxt *acl_context_0) {
  Common_Test test;
  int rows, cols, type;
  bool ret;
  const int rowsMax = 128;
  const int colsMax = 128;
  const int typeMax = 7;

  for (type = 0; type < typeMax; type++) {
    for (rows = 1; rows < rowsMax; rows++) {
      for (cols = 1; cols < colsMax; cols++) {
        aclMat aclmat_src(rows, cols, type, acl_context_0);
        aclMat aclmat_dest(aclmat_src);
        ret = test.Test_Diff(aclmat_src, aclmat_dest);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_constructor: -> aclMat(aclmat_src) <- is success" << endl;

  for (type = 0; type < typeMax; type++) {
    for (rows = 1; rows < rowsMax; rows++) {
      for (cols = 1; cols < colsMax; cols++) {
        aclMat aclmat_src(cv::Size(cols, rows), type, acl_context_0,
                          MEMORY_ALIGN);
        aclMat aclmat_dest(aclmat_src);
        ret = test.Test_Diff(aclmat_src, aclmat_dest);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_constructor: -> aclMat(const aclMat& other) <- is success"
       << endl;
}

void AclMat_Test::Test_constructor_DATA(aclCxt *acl_context_0) {
  Common_Test test;
  int rows, cols, type;
  bool ret;
  const int rowsMax = 128;
  const int colsMax = 128;
  const int typeMax = 7;

  for (type = 0; type < typeMax; type++) {
    for (rows = 1; rows < rowsMax; rows++) {
      for (cols = 1; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type);
        Mat mat_dest(rows, cols, type);
        test.SetDataRange(mat_src);

        aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
        aclmat_src.download(mat_dest);
        ret = test.Test_Diff(mat_src, mat_dest);
        ASSERT_TRUE(ret);
      }
    }
  }
  cerr << "Test_constructor_DATA: -> aclMat(rows, cols, type, data, "
          "acl_context)) <- is success"
       << endl;

  for (type = 0; type < typeMax; type++) {
    for (rows = 1; rows < rowsMax; rows++) {
      for (cols = 1; cols < colsMax; cols++) {
        Mat mat_src(cv::Size(cols, rows), type);
        Mat mat_dest(cv::Size(cols, rows), type);
        test.SetDataRange(mat_src);

        aclMat aclmat_src(cv::Size(cols, rows), type, mat_src.data,
                          acl_context_0);
        aclmat_src.download(mat_dest);
        ret = test.Test_Diff(mat_src, mat_dest);
        ASSERT_TRUE(ret);
      }
    }
  }

  cerr << "Test_constructor_DATA: -> aclMat(size, type, data, acl_context)) <- "
          "is success"
       << endl;
}

void AclMat_Test::Test_constructor_RANGE(aclCxt *acl_context_0) {
  Common_Test test;
  int type;
  bool ret;
  int rangerows, rangecols;
  int rows = 64, cols = 64;
  const int rangerowsMax = 64;
  const int rangecolsMax = 64;
  const int typeMax = 7;
  constexpr int large_mat_range = 4;
  constexpr int small_mat_range = 2;

  for (type = 0; type < typeMax; type++) {
    for (rangerows = large_mat_range; rangerows < rangerowsMax; rangerows++) {
      for (rangecols = large_mat_range; rangecols < rangecolsMax; rangecols++) {
        Mat mat_src(rows, cols, type);
        Mat mat_dest(rows, cols, type);
        test.SetDataRange(mat_src);
        test.SetDataRange(mat_dest);

        Mat mat_rangesrc(mat_src, cv::Range(small_mat_range, rangerows),
                         cv::Range(small_mat_range, rangecols));
        Mat mat_rangedest(mat_dest, cv::Range(small_mat_range, rangerows),
                          cv::Range(small_mat_range, rangecols));
        aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
        aclMat aclmat_range(aclmat_src, cv::Range(small_mat_range, rangerows),
                            cv::Range(small_mat_range, rangecols));
        aclmat_range.download(mat_rangedest);
        ret = test.Test_Diff(mat_rangesrc, mat_rangedest);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_constructor_RANGE: -> aclMat(aclmat_src, rowragne, colrange)) "
          "<- is success"
       << endl;
}

void AclMat_Test::Test_constructor_ROI(aclCxt *acl_context_0) {
  Common_Test test;
  {
    int rows = 6, cols = 8;
    int type = CV_8UC1;
    constexpr int test_val_1 = 1;
    constexpr int test_val_2 = 2;
    cv::Rect roi(test_val_2, test_val_2, test_val_1, test_val_1);
    bool ret;
    Mat mat_src(rows, cols, type);
    Mat mat_dest(rows, cols, type);

    test.SetDataRange(mat_src);
    test.SetDataRange(mat_dest);

    Mat mat_roi1(mat_src, roi);
    Mat mat_roi(mat_dest, roi);

    aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
    aclMat aclmat_roi(aclmat_src, roi);
    aclmat_roi.download(mat_roi);
    ret = test.Test_Diff(mat_roi1, mat_roi);
    ASSERT_TRUE(ret);
  }

  {
    int rows = 12, cols = 61;
    constexpr int test_val_2 = 2;
    constexpr int test_val_8 = 8;
    int type = CV_16UC3;
    cv::Rect roi(test_val_8, test_val_8, test_val_2, test_val_2);
    bool ret;
    Mat mat_src(rows, cols, type);
    Mat mat_dest(rows, cols, type);

    test.SetDataRange(mat_src);
    test.SetDataRange(mat_dest);

    Mat mat_roi1(mat_src, roi);
    Mat mat_roi(mat_dest, roi);

    aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
    aclMat aclmat_roi(aclmat_src, roi);
    aclmat_roi.download(mat_roi);
    ret = test.Test_Diff(mat_roi1, mat_roi);
    ASSERT_TRUE(ret);
  }

  {
    int rows = 16, cols = 80;
    int type = CV_32FC3;
    constexpr int test_val_1 = 1;
    constexpr int test_val_3 = 3;
    constexpr int test_val_4 = 4;
    constexpr int test_val_8 = 8;
    cv::Rect roi(test_val_8, test_val_4, test_val_1, test_val_3);
    bool ret;
    Mat mat_src(rows, cols, type);
    Mat mat_dest(rows, cols, type);

    test.SetDataRange(mat_src);
    test.SetDataRange(mat_dest);

    Mat mat_roi1(mat_src, roi);
    Mat mat_roi(mat_dest, roi);

    aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
    aclMat aclmat_roi(aclmat_src, roi);
    aclmat_roi.download(mat_roi);
    ret = test.Test_Diff(mat_roi1, mat_roi);
    ASSERT_TRUE(ret);
  }

  clog << "Test_constructor_ROI: -> aclMat(aclmat_src, roi)) <- is success"
       << endl;
}

void AclMat_Test::Test_constructor_MAT(aclCxt *acl_context_0) {
  Common_Test test;
  int rows, cols, type;
  bool ret;
  const int rowsMax = 1048;
  const int colsMax = 1048;
  const int typeMax = 7;
  constexpr int lval = 1000;

  for (type = 0; type < typeMax; type++) {
    for (rows = lval; rows < rowsMax; rows++) {
      for (cols = lval; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type);
        Mat mat_dest(rows, cols, type);
        test.SetDataRange(mat_src);

        aclMat aclmat_src(mat_src, acl_context_0);
        aclmat_src.download(mat_dest);
        ret = test.Test_Diff(mat_src, mat_dest);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_constructor_MAT: -> aclMat(mat_src, acl_context_0)) <- is "
          "success"
       << endl;
}

void AclMat_Test::Test_DATA_TRANSFER(aclCxt *acl_context_0) {
  Common_Test test;
  int rows, cols, type;
  bool ret;
  const int rowsMax = 1048;
  const int colsMax = 1048;
  constexpr int lval = 1024;
  const int typeMax = 7;

  for (type = 0; type < typeMax; type++) {
    for (rows = lval; rows < rowsMax; rows++) {
      for (cols = lval; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type);
        Mat mat_dest(rows, cols, type);

        test.SetDataRange(mat_src);
        test.SetDataRange(mat_dest);

        aclMat aclmat_src(rows, cols, type, acl_context_0);
        aclmat_src.upload(mat_src);
        aclmat_src.download(mat_dest);
        ret = test.Test_Diff(mat_src, mat_dest);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_DATA_TRANSFER_UNALIGNED: -> upload(), download() <- is success"
       << endl;

  for (type = 0; type < typeMax; type++) {
    for (rows = lval; rows < rowsMax; rows++) {
      for (cols = lval; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type);
        Mat mat_dest(rows, cols, type);

        test.SetDataRange(mat_src);
        test.SetDataRange(mat_dest);

        aclMat aclmat_src(rows, cols, type, acl_context_0, MEMORY_ALIGN);
        aclmat_src.upload(mat_src, MEMORY_ALIGN);
        aclmat_src.download(mat_dest, MEMORY_ALIGN);
        ret = test.Test_Diff(mat_src, mat_dest);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_DATA_TRANSFER_ALIGN: -> upload(), download() <- is success"
       << endl;
}

void AclMat_Test::Test_DATA_TRANSFERASYNC(aclCxt *acl_context_0) {
  Common_Test test;
  int rows, cols, type;
  bool ret;
  const int rowsMax = 1048;
  const int colsMax = 1048;
  const int typeMax = 7;
  constexpr int lval = 1024;

  for (type = 0; type < typeMax; type++) {
    for (rows = lval; rows < rowsMax; rows++) {
      for (cols = lval; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type);
        Mat mat_dest(rows, cols, type);

        test.SetDataRange(mat_src);
        test.SetDataRange(mat_dest);

        aclMat aclmat_src(rows, cols, type, acl_context_0);
        aclmat_src.upload(mat_src, aclmat_src.acl_context->get_stream(0));
        aclmat_src.download(mat_dest, aclmat_src.acl_context->get_stream(0));
        ret = test.Test_Diff(mat_src, mat_dest);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_DATA_TRANSFERASYNC_UNALIGNED: -> upload(), download() <- is "
          "success"
       << endl;

  for (type = 0; type < typeMax; type++) {
    for (rows = lval; rows < rowsMax; rows++) {
      for (cols = lval; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type);
        Mat mat_dest(rows, cols, type);

        test.SetDataRange(mat_src);
        test.SetDataRange(mat_dest);

        aclMat aclmat_src(rows, cols, type, acl_context_0, MEMORY_ALIGN);
        aclmat_src.upload(mat_src, aclmat_src.acl_context->get_stream(0),
                          MEMORY_ALIGN);
        aclmat_src.download(mat_dest, aclmat_src.acl_context->get_stream(0),
                            MEMORY_ALIGN);
        ret = test.Test_Diff(mat_src, mat_dest);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_DATA_TRANSFERASYNC_ALIGN: -> upload(), download() <- is success"
       << endl;
}

static inline void dataSwap(int &data1, int &data2) {
  Common_Test test;
  int temp;
  if (data1 < data2) {
    temp = data1;
    data1 = data2;
    data2 = temp;
  }
}

void AclMat_Test::Test_locateROI(aclCxt *acl_context_0) {
  Common_Test test;
  int rows = 256, cols = 256;
  int type = CV_8UC1;
  int rangex, rangey;
  int rangex1, rangey1;
  cv::Size size, size1;
  cv::Point ofs, ofs1;

  for (int x = 0; x < rows * cols; ++x) {
    rangex = (rangex = test.RandDom_()) > 0 ? rangex : 1;
    rangey = (rangey = test.RandDom_()) > 0 ? rangey : 1;
    rangex1 = (rangex1 = test.RandDom_()) > 0 ? rangex1 : 1;
    rangey1 = (rangey1 = test.RandDom_()) > 0 ? rangey1 : 1;

    dataSwap(rangex, rangex1);
    dataSwap(rangey, rangey1);

    Mat mat_src(rows, cols, type);
    Mat mat_range(mat_src, cv::Range(rangex1, rangex + 1),
                  cv::Range(rangey1, rangey + 1));
    mat_range.locateROI(size, ofs);

    aclMat aclmat_src(rows, cols, type, acl_context_0);
    aclMat aclmat_range(aclmat_src, cv::Range(rangex1, rangex + 1),
                        cv::Range(rangey1, rangey + 1));
    aclmat_range.locateROI(size1, ofs1);

    ASSERT_EQ(size.height, size1.height);
    ASSERT_EQ(size.width, size1.width);
    ASSERT_EQ(ofs.x, ofs1.x);
    ASSERT_EQ(ofs.y, ofs1.y);
  }
  clog << "Test_loacteROI: -> locateROI() <- is success" << endl;
}

void AclMat_Test::Test_swap(aclCxt *acl_context_0) {
  Common_Test test;
  int rows, cols, type;
  bool ret;
  const int rowsMax = 1048;
  const int colsMax = 1048;
  const int typeMax = 7;
  constexpr int lval = 1024;

  for (type = 0; type < typeMax; type++) {
    for (rows = lval; rows < rowsMax; rows++) {
      for (cols = lval; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type);
        Mat mat_dest(rows, cols, type);

        test.SetDataRange(mat_src);
        test.SetDataRange(mat_dest);

        Mat mat_dest1(rows, cols, type);
        Mat mat_dest2(rows, cols, type);

        aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
        aclMat aclmat_src1(rows, cols, type, mat_dest.data, acl_context_0);
        aclmat_src.swap(aclmat_src1);

        aclmat_src.download(mat_dest1);
        aclmat_src1.download(mat_dest2);

        ret = test.Test_Diff(mat_dest1, mat_dest);
        ASSERT_TRUE(ret);

        ret = test.Test_Diff(mat_dest2, mat_src);
        ASSERT_TRUE(ret);
      }
    }
  }
  clog << "Test_Swap: -> swap() <- is success" << endl;
}

void AclMat_Test::Test_operator_add(aclCxt *acl_context) {
  Common_Test test;
  int rows, cols;
  bool ret;
  constexpr int rand_data_range = 32;
  constexpr int lval = 1024;
  const int rowsMax = 1048;
  const int colsMax = 1048;

  vector<int> type {CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3, CV_32SC1, CV_32SC3};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Correctness test: Functoin: operator+=()", type[i]);
    for (rows = lval; rows < rowsMax; rows++) {
      for (cols = lval; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type[i]);
        Mat mat_dest(rows, cols, type[i]);
        Mat mat_dest1(rows, cols, type[i]);

        test.SetDataRange(mat_src, rand_data_range);
        test.SetDataRange(mat_dest, rand_data_range);

        aclMat aclmat_src(rows, cols, type[i], mat_src.data, acl_context,
                          MEMORY_ALIGN);
        aclMat aclmat_dest(rows, cols, type[i], mat_dest.data, acl_context,
                           MEMORY_ALIGN);

        mat_dest += mat_src;

        aclmat_dest += aclmat_src;
        wait_stream(acl_context);
        aclmat_dest.download(mat_dest1, MEMORY_ALIGN);

        ret = test.Test_Diff(mat_dest, mat_dest1);
        ASSERT_TRUE(ret);
      }
    }
  }
}

void AclMat_Test::Test_operator_sub(aclCxt *acl_context) {
  Common_Test test;
  int rows, cols;
  bool ret;
  const int rowsMax = 1048;
  const int colsMax = 1048;
  constexpr int lval = 1024;
  constexpr int s_val1 = 1, s_val2 = 2, s_val3 = 3;
  constexpr int s_val4 = 4, s_val6 = 6, s_val8 = 8;

  vector<int> type {CV_8UC1,  CV_8UC3,  CV_32FC1, CV_32FC3,
                   CV_32SC1, CV_32SC3, CV_64FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Correctness test: Functoin: operator-=()", type[i]);
    for (rows = lval; rows < rowsMax; rows++) {
      for (cols = lval; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type[i], Scalar(s_val1, s_val2, s_val3));
        Mat mat_dest(rows, cols, type[i], Scalar(s_val4, s_val6, s_val8));
        Mat mat_dest1(rows, cols, type[i]);

        aclMat aclmat_src(rows, cols, type[i], mat_src.data, acl_context,
                          MEMORY_ALIGN);
        aclMat aclmat_dest(rows, cols, type[i], mat_dest.data, acl_context,
                           MEMORY_ALIGN);

        mat_dest -= mat_src;

        aclmat_dest -= aclmat_src;
        wait_stream(acl_context);
        aclmat_dest.download(mat_dest1, MEMORY_ALIGN);

        ret = test.Test_Diff(mat_dest, mat_dest1);
        ASSERT_TRUE(ret);
      }
    }
  }
}

void AclMat_Test::Test_operator_div(aclCxt *acl_context) {
  Common_Test test;
  int rows, cols;
  bool ret;
  const int rowsMax = 1048;
  const int colsMax = 1048;
  constexpr int lval = 1024;
  constexpr int s_val1 = 1, s_val2 = 2;
  constexpr int s_val4 = 4, s_val6 = 6, s_val8 = 8;

  vector<int> type {CV_8UC1,  CV_8UC3,  CV_32FC1, CV_32FC3,
                   CV_32SC1, CV_32SC3, CV_64FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Correctness test: Functoin: operator/=()", type[i]);
    for (rows = lval; rows < rowsMax; rows++) {
      for (cols = lval; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type[i], Scalar(s_val1, s_val2, s_val4));
        Mat mat_dest(rows, cols, type[i], Scalar(s_val4, s_val6, s_val8));
        Mat mat_dest1(rows, cols, type[i]);

        aclMat aclmat_src(rows, cols, type[i], mat_src.data, acl_context,
                          MEMORY_ALIGN);
        aclMat aclmat_dest(rows, cols, type[i], mat_dest.data, acl_context,
                           MEMORY_ALIGN);

        mat_dest /= mat_src;

        aclmat_dest /= aclmat_src;
        wait_stream(acl_context);
        aclmat_dest.download(mat_dest1, MEMORY_ALIGN);

        ret = test.Test_Diff(mat_dest, mat_dest1);
        ASSERT_TRUE(ret);
      }
    }
  }
}

void AclMat_Test::Test_operator_mul(aclCxt *acl_context) {
  Common_Test test;
  int val;
  bool ret;
  constexpr int rand_data_range = 32;
  const int valMax = 1048;
  constexpr int lval = 1024;

  vector<int> type {CV_32FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Correctness test: Functoin: operator*=()", type[i]);
    for (val = lval; val < valMax; val++) {
      Mat mat_src(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src, rand_data_range);
      test.SetDataRange(mat_dest, rand_data_range);

      aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
      aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

      mat_dest *= mat_src;

      aclmat_dest *= aclmat_src;
      wait_stream(acl_context);
      aclmat_dest.download(mat_dest1);

      ret = test.Test_Diff(mat_dest, mat_dest1);
      ASSERT_TRUE(ret);
    }
  }
}
