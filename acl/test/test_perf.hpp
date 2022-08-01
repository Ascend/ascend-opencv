#ifndef __OPENCV_TEST_PERF_HPP__
#define __OPENCV_TEST_PERF_HPP__

#include "test_precomp.hpp"

class PERF_TEST
{
public:
    CV_EXPORTS void Test_operator_add_perf(aclCxt *acl_context);
    CV_EXPORTS void Test_operator_sub_perf(aclCxt *acl_context);
    CV_EXPORTS void Test_operator_div_perf(aclCxt *acl_context);
    CV_EXPORTS void Test_operator_mul_perf(aclCxt *acl_context);
    CV_EXPORTS void Test_Abs(aclCxt *acl_context);
    CV_EXPORTS void Test_Pow(aclCxt *acl_context);
    CV_EXPORTS void Test_Sqrt(aclCxt *acl_context);
    CV_EXPORTS void Test_Add(aclCxt *acl_context);
    CV_EXPORTS void Test_Divide(aclCxt *acl_context);
    CV_EXPORTS void Test_Exp(aclCxt *acl_context);
    CV_EXPORTS void Test_Log(aclCxt *acl_context);
    CV_EXPORTS void Test_Max(aclCxt *acl_context);
    CV_EXPORTS void Test_Min(aclCxt *acl_context);

    CV_EXPORTS void Test_MatMul(aclCxt *acl_context);
    CV_EXPORTS void Test_Convolution(aclCxt *acl_context);

    CV_EXPORTS void Test_Lookuptable(aclCxt *acl_context);
    CV_EXPORTS void Test_Merge(aclCxt *acl_context);
    CV_EXPORTS void Test_Split(aclCxt *acl_context);
    CV_EXPORTS void Test_Transpose(aclCxt *acl_context);
    CV_EXPORTS void Test_Flip(aclCxt *acl_context);

    CV_EXPORTS void Test_other(aclCxt *acl_context);
    CV_EXPORTS void Test_other1(aclCxt *acl_context);
    CV_EXPORTS void Test_other2();
};

#endif