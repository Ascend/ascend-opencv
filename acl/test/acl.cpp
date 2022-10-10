#include "test_common.hpp"
#include "test_correctness.hpp"
#include "test_perf.hpp"

using namespace cv;
using namespace cv::acl;
using namespace cvtest;
using namespace testing;
using namespace std;

namespace opencv_test {
namespace {
aclCxt *acl_context_0 = set_device("../../modules/acl/test/acl.json", 2, 3);
////////////////////////////////////////////////////Correctness_test////////////////////////////////////////////////////////
/* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7
 * test function:
 * config: MEMORY_ALIGN
 * aclMat(int rows, int cols, int type, aclCxt *acl_context, ALIGNMENT config =
 * MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
 * aclMat(Size size, int type, aclCxt *acl_context, ALIGNMENT config =
 * MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
 * aclMat(const aclMat &m);
 *
 */
TEST(ACLMAT_CONSTRUCTOR, MEMORY_ALIGN) {
  AclMat_Test test;
  test.Test_constructor_ALIGN(acl_context_0);
}

/* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7
 * test function:
 * config: MEMORY_UNALIGNED
 * aclMat(int rows, int cols, int type, aclCxt *acl_context, ALIGNMENT config =
 * MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
 * aclMat(Size size, int type, aclCxt *acl_context, ALIGNMENT config =
 * MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
 *
 */
TEST(ACLMAT_CONSTRUCTOR, MEMORY_UNALIGNED) {
  AclMat_Test test;
  test.Test_constructor_UNALIGNED(acl_context_0);
}

/* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7
 * test function:
 * aclMat(const aclMat &m);
 */
TEST(ACLMAT_CONSTRUCTOR, COPY_CONSTRUCTOR) {
  AclMat_Test test;
  test.Test_constructor(acl_context_0);
}

/* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7
 * test function:
 * aclMat(int rows, int cols, int type, void *data, aclCxt* acl_context,
 * ALIGNMENT config = MEMORY_UNALIGNED, size_t step = Mat::AUTO_STEP);
 * aclMat(Size size, int type, void *data, aclCxt* acl_context, ALIGNMENT config
 * = MEMORY_UNALIGNED, size_t step = Mat::AUTO_STEP);
 */
TEST(ACLMAT_CONSTRUCTOR, DATA) {
  AclMat_Test test;
  test.Test_constructor_DATA(acl_context_0);
}

/* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7
 * test function:
 * aclMat(const aclMat &m, const Range &rowRange, const Range &colRange =
 * Range::all());
 *
 */
TEST(ACLMAT_CONSTRUCTOR, RANGE) {
  AclMat_Test test;
  test.Test_constructor_RANGE(acl_context_0);
}

/*
 * test function:
 * aclMat(const aclMat &m, const Rect &roi);
 *
 */
TEST(ACLMAT_CONSTRUCTOR, ROI) {
  AclMat_Test test;
  test.Test_constructor_ROI(acl_context_0);
}

/*
 * test function:
 * aclMat (const Mat &m, aclCxt* acl_context, ALIGNMENT config =
 * MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
 */
TEST(ACLMAT_CONSTRUCTOR, MAT) {
  AclMat_Test test;
  test.Test_constructor_MAT(acl_context_0);
}

/* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7
 * test function:
 * CV_EXPORTS void upload(const Mat &m, ALIGNMENT config = MEMORY_UNALIGNED);
 * CV_EXPORTS void upload(const Mat &m, aclStream stream, ALIGNMENT config =
 * MEMORY_UNALIGNED);
 *
 */
TEST(ACLMAT_FUNCTION, DATA_TRANSFER) {
  AclMat_Test test;
  test.Test_DATA_TRANSFER(acl_context_0);
}

/* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7
 * test function:
 * CV_EXPORTS void download(Mat &m, ALIGNMENT config = MEMORY_UNALIGNED) const;
 * CV_EXPORTS void download(Mat &m, aclStream stream, ALIGNMENT config =
 * MEMORY_UNALIGNED) const;
 *
 */
TEST(ACLMAT_FUNCTION, DATA_TRANSFERASYNC) {
  AclMat_Test test;
  test.Test_DATA_TRANSFERASYNC(acl_context_0);
}

/*
 * test function:
 * void locateROI(Size &wholeSize, Point &ofs) const;
 */
TEST(ACLMAT_FUNCTION, LOCATEROI) {
  AclMat_Test test;
  test.Test_locateROI(acl_context_0);
}

/*
 * test function:
 * void swap(aclMat &mat);
 *
 */
TEST(ACLMAT_FUNCTION, SWAP) {
  AclMat_Test test;
  test.Test_swap(acl_context_0);
}

/*
 * test function:
 * operator+=()
 *
 */
TEST(ACLMAT_FUNCTION, OPERATOR_ADD) {
  AclMat_Test test;
  test.Test_operator_add(acl_context_0);
}

/*
 * test function:
 * operator-=()
 *
 */
TEST(ACLMAT_FUNCTION, OPERATOR_SUB) {
  AclMat_Test test;
  test.Test_operator_sub(acl_context_0);
}

/*
 * test function:
 * operator*=()
 *
 */
TEST(ACLMAT_FUNCTION, OPERATOR_MUL) {
  AclMat_Test test;
  test.Test_operator_mul(acl_context_0);
}

/*
 * test function:
 * operator/=()
 *
 */
TEST(ACLMAT_FUNCTION, OPERATOR_DIV) {
  AclMat_Test test;
  test.Test_operator_div(acl_context_0);
}

////////////////////////////////////////////////////Perf_test////////////////////////////////////////////////////////

TEST(Operator, add) {
  PERF_TEST test;
  test.Test_operator_add_perf(acl_context_0);
}

TEST(Operator, sub) {
  PERF_TEST test;
  test.Test_operator_sub_perf(acl_context_0);
}

TEST(Operator, div) {
  PERF_TEST test;
  test.Test_operator_div_perf(acl_context_0);
}

TEST(Operator, mul) {
  PERF_TEST test;
  test.Test_operator_mul_perf(acl_context_0);
}

TEST(Mathfunction, abs) {
  PERF_TEST test;
  test.Test_Abs(acl_context_0);
}

TEST(Mathfunction, pow) {
  PERF_TEST test;
  test.Test_Pow(acl_context_0);
}

TEST(Mathfunction, sqrt) {
  PERF_TEST test;
  test.Test_Sqrt(acl_context_0);
}

TEST(Mathfunction, add) {
  PERF_TEST test;
  test.Test_Add(acl_context_0);
}

TEST(Mathfunction, divide) {
  PERF_TEST test;
  test.Test_Divide(acl_context_0);
}

TEST(Mathfunction, exp) {
  PERF_TEST test;
  test.Test_Exp(acl_context_0);
}

TEST(Mathfunction, log) {
  PERF_TEST test;
  test.Test_Log(acl_context_0);
}

TEST(Mathfunction, max) {
  PERF_TEST test;
  test.Test_Max(acl_context_0);
}

TEST(Mathfunction, min) {
  PERF_TEST test;
  test.Test_Min(acl_context_0);
}

TEST(Gemm, MatMul) {
  PERF_TEST test;
  test.Test_MatMul(acl_context_0);
}

TEST(Gemm, Convolution) {
  PERF_TEST test;
  test.Test_Convolution(acl_context_0);
}

TEST(Matrices, merge) {
  PERF_TEST test;
  test.Test_Merge(acl_context_0);
}

TEST(Matrices, split) {
  PERF_TEST test;
  test.Test_Split(acl_context_0);
}

TEST(Matrices, transpose) {
  PERF_TEST test;
  test.Test_Transpose(acl_context_0);
}

TEST(Matrices, flip) {
  PERF_TEST test;
  test.Test_Flip(acl_context_0);
  release_device(acl_context_0);
}
}  // namespace
}  // namespace opencv_test