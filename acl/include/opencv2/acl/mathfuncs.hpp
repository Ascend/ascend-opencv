#ifndef OPENCV_MATHFUNCS_HPP
#define OPENCV_MATHFUNCS_HPP

#include "acl_mat.hpp"

/**
 *   mathfunctions;
 */

namespace cv {
namespace acl {
CV_EXPORTS aclMat abs(const aclMat &src, int stream_id = 0);
CV_EXPORTS void pow(const aclMat &src, double power, aclMat &dest,
                    int stream_id = 0);
CV_EXPORTS void sqrt(const aclMat &src, aclMat &dest, int stream_id = 0);
CV_EXPORTS void add(const aclMat &src, const aclMat &other_src, aclMat &dest,
                    int stream_id = 0);
CV_EXPORTS void divide(const aclMat &src, const aclMat &other_src, aclMat &dest,
                       int stream_id = 0);
CV_EXPORTS void exp(const aclMat &src, aclMat &dest, int stream_id = 0);
CV_EXPORTS void log(const aclMat &src, aclMat &dest, int stream_id = 0);
CV_EXPORTS void max(const aclMat &src, const aclMat &other_src, aclMat &dest,
                    int stream_id = 0);
CV_EXPORTS void min(const aclMat &src, const aclMat &other_src, aclMat &dest,
                    int stream_id = 0);
} /* end of namespace acl */

} /* end of namespace cv */

#endif