#ifndef __OPENCV_TEST_COMMON_HPP__
#define __OPENCV_TEST_COMMON_HPP__

#include "test_precomp.hpp"

typedef enum TestDatatype {
    INT = 1,
    FLOAT
} TestDatatype;

class CV_EXPORTS Common_Test {
    public:
        Common_Test();
        ~Common_Test();
        CV_EXPORTS bool Test_Diff(const aclMat& aclmat, const Mat& mat, ALIGNMENT config = ALIGNMENT::MEMORY_UNALIGNED);
        CV_EXPORTS bool Test_Diff(const aclMat& aclmat, const aclMat& aclmat_other);
        CV_EXPORTS bool Test_Diff(const Mat& mat, const Mat& mat_other);
        CV_EXPORTS void MatShow(Mat &m, string str);
        CV_EXPORTS void StatShow(Mat &mat_src, aclMat &aclmat_dst);
        CV_EXPORTS void PrintLog(const string& funcname, int type);

        CV_EXPORTS size_t RandDom_(int config = 0xff);
        CV_EXPORTS bool SetDataRange(Mat &src, int dataRange = 0xff);
};


#endif