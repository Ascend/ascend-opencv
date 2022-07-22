#ifndef OPENCV_MATRICES_HPP   
#define OPENCV_MATRICES_HPP

#include "acl_mat.hpp"

namespace cv
{
    namespace acl
    {
        // Matrix lookup table
        //CV_EXPORTS void lookUpTable(const aclMat& src, const aclMat& lut, aclMat& dst);
        // Multiple channel merge
        CV_EXPORTS void merge(const vector<aclMat>& mv, aclMat& dst);
        // Split into channels 
        CV_EXPORTS void split(const aclMat& src, vector<aclMat>& mv);
        // Matrix transpose
        CV_EXPORTS void transpose(const aclMat& src, aclMat& dest);
        CV_EXPORTS void flip(const aclMat& src, aclMat& dest, int flipCode = 0);
    } /* end of namespace acl */

} /* end of namespace cv */

#endif