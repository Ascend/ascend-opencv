#ifndef OPENCV_GEMM_HPP
#define OPENCV_GEMM_HPP

#include "acl_mat.hpp"

namespace cv {
namespace acl {
// matrix multiplication
CV_EXPORTS void MatMul(const aclMat &src1, const aclMat &src2, aclMat &dest,
                       int stream_id = 0);
// convolution
CV_EXPORTS void Convolution(
    const aclMat &src, const aclMat &kernel, aclMat &dest,
    const std::vector<int64_t> &stridesList = std::vector<int64_t>{1, 1, 1, 1},
    const std::vector<int64_t> &padsList = std::vector<int64_t>{0, 0, 0, 0},
    int stream_id = 0);

} /* end of namespace acl */

} /* end of namespace cv */

#endif