#ifndef __OPENCV_TEST_COMMON_HPP__
#define __OPENCV_TEST_COMMON_HPP__

#include "test_precomp.hpp"



class CV_EXPORTS AclMat_Test {
    public:
        AclMat_Test();
        ~AclMat_Test();
        /* test set_device() */
        CV_EXPORTS void Test_set_device();
        /* test aclMat(int rows, int cols, int type, aclCxt *acl_context, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST) */
        CV_EXPORTS void Test_constructor_UNALIGNED(aclCxt *acl_context);
        CV_EXPORTS void Test_constructor_ALIGN(aclCxt *acl_context);

        /* test aclMat(const aclMat &m) */
        CV_EXPORTS void Test_constructor(aclCxt *acl_context);
        /* test aclMat(int rows, int cols, int type, void *data, aclCxt* acl_context, size_t step = Mat::AUTO_STEP) */
        CV_EXPORTS void Test_constructor_DATA(aclCxt *acl_context);
        /* test aclMat(const aclMat &m, const Range &rowRange, const Range &colRange = Range::all()) */
        CV_EXPORTS void Test_constructor_RANGE(aclCxt *acl_context);
        /* test aclMat(const aclMat &m, const Rect &roi) */
        CV_EXPORTS void Test_constructor_ROI(aclCxt *acl_context);
        /* test aclMat (const Mat &m, aclCxt* acl_context, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST) */
        CV_EXPORTS void Test_constructor_MAT(aclCxt *acl_context);
        /* test upload download*/
        CV_EXPORTS void Test_DATA_TRANSFER(aclCxt *acl_context);
        /* test upload_2d download_2d */
        CV_EXPORTS void Test_DATA_TRANSFERASYNC(aclCxt *acl_context);
        /* test locateROI adjustROI */
        CV_EXPORTS void Test_locateROI(aclCxt *acl_context);
        /* test swap */
        CV_EXPORTS void Test_swap(aclCxt *acl_context);
        CV_EXPORTS void Test_operator(aclCxt *acl_context);
        CV_EXPORTS void Test_operator_perf(aclCxt *acl_context);
        CV_EXPORTS void Test_Abs(aclCxt *acl_context);

        /* powerful function below */
        CV_EXPORTS bool Test_Diff(const aclMat& aclmat, const Mat& mat, ALIGNMENT config = ALIGNMENT::MEMORY_UNALIGNED);
        CV_EXPORTS bool Test_Diff(const aclMat& aclmat, const aclMat& aclmat_other);
        CV_EXPORTS bool Test_Diff(const Mat& mat, const Mat& mat_other);
        CV_EXPORTS void MatShow(Mat &m, string str);
        CV_EXPORTS void StatShow(Mat &mat_src, aclMat &aclmat_dst);
        CV_EXPORTS size_t RandDom_(int config = 0xff);
};

void thread_handler(void);


#endif