#ifndef OPENCV_MATRICES_HPP
#define OPENCV_MATRICES_HPP

#include "acl_mat.hpp"

namespace cv {
namespace acl {
// Multiple channel merge
CV_EXPORTS void merge(const std::vector<aclMat> &mv, aclMat &dst, int stream_id = 0);
// Split into channels
CV_EXPORTS void split(const aclMat &src, std::vector<aclMat> &mv, int stream_id = 0);
// Matrix transpose
CV_EXPORTS void transpose(const aclMat &src, aclMat &dest, int stream_id = 0);
CV_EXPORTS void flip(const aclMat &src, aclMat &dest, int flipCode = 0,
                     int stream_id = 0);
} /* end of namespace acl */

} /* end of namespace cv */

#endif