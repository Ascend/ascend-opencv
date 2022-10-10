#ifndef OPENCV_CORRECTNESS_HPP__
#define OPENCV_CORRECTNESS_HPP__

#include "test_precomp.hpp"

class CV_EXPORTS AclMat_Test {
 public:
  AclMat_Test();
  ~AclMat_Test();
  /* test set_device() */
  CV_EXPORTS void Test_set_device();
  /* test aclMat(int rows, int cols, int type, cv::acl::aclCxt *acl_context,
   * aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST) */
  CV_EXPORTS void Test_constructor_UNALIGNED(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_constructor_ALIGN(cv::acl::aclCxt *acl_context);

  /* test aclMat(const aclMat &m) */
  CV_EXPORTS void Test_constructor(cv::acl::aclCxt *acl_context);
  /* test aclMat(int rows, int cols, int type, void *data, cv::acl::aclCxt* acl_context,
   * size_t step = Mat::AUTO_STEP) */
  CV_EXPORTS void Test_constructor_DATA(cv::acl::aclCxt *acl_context);
  /* test aclMat(const aclMat &m, const Range &rowRange, const Range &colRange =
   * Range::all()) */
  CV_EXPORTS void Test_constructor_RANGE(cv::acl::aclCxt *acl_context);
  /* test aclMat(const aclMat &m, const Rect &roi) */
  CV_EXPORTS void Test_constructor_ROI(cv::acl::aclCxt *acl_context);
  /* test aclMat (const Mat &m, cv::acl::aclCxt* acl_context, aclrtMemMallocPolicy policy
   * = ACL_MEM_MALLOC_HUGE_FIRST) */
  CV_EXPORTS void Test_constructor_MAT(cv::acl::aclCxt *acl_context);
  /* test upload download*/
  CV_EXPORTS void Test_DATA_TRANSFER(cv::acl::aclCxt *acl_context);
  /* test upload_2d download_2d */
  CV_EXPORTS void Test_DATA_TRANSFERASYNC(cv::acl::aclCxt *acl_context);
  /* test locateROI adjustROI */
  CV_EXPORTS void Test_locateROI(cv::acl::aclCxt *acl_context);
  /* test swap */
  CV_EXPORTS void Test_swap(cv::acl::aclCxt *acl_context);

  CV_EXPORTS void Test_operator_add(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_operator_sub(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_operator_mul(cv::acl::aclCxt *acl_context);
  CV_EXPORTS void Test_operator_div(cv::acl::aclCxt *acl_context);
};

void thread_handler(void);

#endif