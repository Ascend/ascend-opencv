/* M/////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
//  license. If you do not agree to this license, do not download, install, copy
//  or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science,
// all rights reserved. Copyright (C) 2010-2012, Advanced Micro Devices, Inc.,
// all rights reserved. Copyright (C) 2010-2012, Multicoreware, Inc., all rights
// reserved. Third party copyrights are property of their respective owners.
//
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
//   notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
//   products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or business
// interruption) however caused and on any theory of liability, whether in
// contract, strict liability, or tort (including negligence or otherwise)
// arising in any way out of the use of this software, even if advised of the
// possibility of such damage.
//
// M*/

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
  const int rowsMax = 128;
  const int colsMax = 128;
  const int typeMax = 7;

  for (type = 0; type < typeMax; type++) {
    for (rows = 1; rows < rowsMax; rows++) {
      for (cols = 1; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type);
        aclMat aclmat_src(rows, cols, type, acl_context);
        test.SetDataRange(mat_src, 32);
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
        test.SetDataRange(mat_src, 32);
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
  const int rowsMax = 128;
  const int colsMax = 128;
  const int typeMax = 7;

  for (type = 0; type < typeMax; type++) {
    for (rows = 1; rows < rowsMax; rows++) {
      for (cols = 1; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type);
        test.SetDataRange(mat_src, 32);
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
        test.SetDataRange(mat_src, 32);
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

  for (type = 0; type < typeMax; type++) {
    for (rangerows = 4; rangerows < rangerowsMax; rangerows++) {
      for (rangecols = 4; rangecols < rangecolsMax; rangecols++) {
        Mat mat_src(rows, cols, type);
        Mat mat_dest(rows, cols, type);
        test.SetDataRange(mat_src);
        test.SetDataRange(mat_dest);

        Mat mat_rangesrc(mat_src, cv::Range(2, rangerows),
                         cv::Range(2, rangecols));
        Mat mat_rangedest(mat_dest, cv::Range(2, rangerows),
                          cv::Range(2, rangecols));
        aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
        aclMat aclmat_range(aclmat_src, cv::Range(2, rangerows),
                            cv::Range(2, rangecols));
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
    cv::Rect roi(2, 2, 1, 1);
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
    int type = CV_16UC3;
    cv::Rect roi(8, 8, 2, 2);
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
    cv::Rect roi(8, 4, 1, 3);
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

  for (type = 0; type < typeMax; type++) {
    for (rows = 1000; rows < rowsMax; rows++) {
      for (cols = 1000; cols < colsMax; cols++) {
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
  const int typeMax = 7;

  for (type = 0; type < typeMax; type++) {
    for (rows = 1000; rows < rowsMax; rows++) {
      for (cols = 1000; cols < colsMax; cols++) {
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
    for (rows = 1000; rows < rowsMax; rows++) {
      for (cols = 1000; cols < colsMax; cols++) {
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

  for (type = 0; type < typeMax; type++) {
    for (rows = 1000; rows < rowsMax; rows++) {
      for (cols = 1000; cols < colsMax; cols++) {
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
    for (rows = 1000; rows < rowsMax; rows++) {
      for (cols = 1000; cols < colsMax; cols++) {
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

  for (type = 0; type < typeMax; type++) {
    for (rows = 1024; rows < rowsMax; rows++) {
      for (cols = 1024; cols < colsMax; cols++) {
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
  const int rowsMax = 1048;
  const int colsMax = 1048;

  vector<int> type{CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3, CV_32SC1, CV_32SC3};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Correctness test: Functoin: operator+=()", type[i]);
    for (rows = 1024; rows < rowsMax; rows++) {
      for (cols = 1024; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type[i]);
        Mat mat_dest(rows, cols, type[i]);
        Mat mat_dest1(rows, cols, type[i]);

        test.SetDataRange(mat_src, 32);
        test.SetDataRange(mat_dest, 32);

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

  vector<int> type{CV_8UC1,  CV_8UC3,  CV_32FC1, CV_32FC3,
                   CV_32SC1, CV_32SC3, CV_64FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Correctness test: Functoin: operator-=()", type[i]);
    for (rows = 1024; rows < rowsMax; rows++) {
      for (cols = 1024; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type[i], Scalar(1, 2, 3));
        Mat mat_dest(rows, cols, type[i], Scalar(4, 6, 8));
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

  vector<int> type{CV_8UC1,  CV_8UC3,  CV_32FC1, CV_32FC3,
                   CV_32SC1, CV_32SC3, CV_64FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Correctness test: Functoin: operator/=()", type[i]);
    for (rows = 1024; rows < rowsMax; rows++) {
      for (cols = 1024; cols < colsMax; cols++) {
        Mat mat_src(rows, cols, type[i], Scalar(1, 2, 4));
        Mat mat_dest(rows, cols, type[i], Scalar(4, 6, 8));
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
  const int valMax = 1048;

  vector<int> type{CV_32FC1};
  for (size_t i = 0; i < type.size(); ++i) {
    test.PrintLog("Correctness test: Functoin: operator*=()", type[i]);
    for (val = 1024; val < valMax; val++) {
      Mat mat_src(val, val, type[i]);
      Mat mat_dest(val, val, type[i]);
      Mat mat_dest1(val, val, type[i]);

      test.SetDataRange(mat_src, 32);
      test.SetDataRange(mat_dest, 32);

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
