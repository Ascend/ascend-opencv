#ifndef OPENCV_GEMM_HPP
#define OPENCV_GEMM_HPP

#include "acl_mat.hpp"

namespace cv
{
    namespace acl
    {
        // matrix multiplication
        CV_EXPORTS void MatMul(const aclMat& src1, const aclMat& src2, aclMat& dest);
        // convolution
        CV_EXPORTS void Convolution(const aclMat& src, const aclMat& kernel, aclMat& dest, \
            const vector<int64_t>& stridesList = vector<int64_t> {1, 1, 1, 1}, const vector<int64_t>& padsList = vector<int64_t> {0, 0, 0, 0});

    } /* end of namespace acl */

} /* end of namespace cv */


#endif 