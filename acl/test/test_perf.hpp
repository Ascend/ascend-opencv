#ifndef __OPENCV_TEST_PERF_HPP__
#define __OPENCV_TEST_PERF_HPP__

#include "test_precomp.hpp"

class PERF_TEST {
public:
  CV_EXPORTS void Test_operator_add_perf(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_operator_sub_perf(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_operator_div_perf(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_operator_mul_perf(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Abs(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Pow(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Sqrt(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Add(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Divide(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Exp(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Log(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Max(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Min(cv::acl::aclCxt *acl_context);

  CV_EXPORTS void Test_MatMul(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Convolution(cv::acl::aclCxt *acl_context);

  CV_EXPORTS void Test_Lookuptable(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Merge(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Split(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Transpose(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_Flip(cv::acl::aclCxt *acl_context);
};

#endif