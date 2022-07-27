#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/acl/acl.hpp"
#include "acl/acl.h"
#include <iostream>

using namespace cv;
using namespace cv::acl;
using namespace std;


int main()
{
    // 初始化
    aclCxt *acl_context_0 = set_device("../acl.json", 1, 2);

    Mat src = imread("../cat1.jpg");
    if (src.empty()) {
        cerr << "could not image !" << endl;
        return -1;
    }

    // 上传数据到aclMat对象中
    aclMat acl_src(src, acl_context_0); 
    aclMat acl_dest1;
    aclMat acl_dest2;
    aclMat acl_dest3;

    vector<aclMat> mv;
    mv.emplace_back(acl_dest1);
    mv.emplace_back(acl_dest2);
    mv.emplace_back(acl_dest3);


    imshow("src", src);
    split(acl_src, mv);

    //下载aclMat数据到Mat类中 
    Mat dest1 = mv.data()[0].operator cv::Mat();
    Mat dest2 = mv.data()[1].operator cv::Mat();
    Mat dest3 = mv.data()[2].operator cv::Mat();

    imshow("dest1", dest1);
    imshow("dest2", dest2);
    imshow("dest3", dest3);

    aclMat acl_imgdest; 

    merge(mv, acl_imgdest);
    Mat imgdest = acl_imgdest.operator cv::Mat(); 
    imshow("imgdest", imgdest);

    // 去初始化
    release_device(acl_context_0);    
	waitKey(0);

    return 0;
}

