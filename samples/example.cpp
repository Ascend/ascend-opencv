#include <iostream>

#include "acl/acl.h"
#include "opencv2/acl/acl.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace cv::acl;
using namespace std;

/**
 * @brief A simple example of the split and merge functions, Using a different
 * Stream
 */
int test_func1(aclCxt *acl_context_0) {
  Mat src = imread("../cat1.jpg");
  if (src.empty()) {
    cerr << "could not image !" << endl;
    return -1;
  }

  // 上传数据到aclMat对象中
  aclMat acl_src (src, acl_context_0);
  aclMat acl_dest1;
  aclMat acl_dest2;
  aclMat acl_dest3;

  vector<aclMat> mv;
  mv.emplace_back(acl_dest1);
  mv.emplace_back(acl_dest2);
  mv.emplace_back(acl_dest3);

  imshow("src", src);
  split(acl_src, mv, 0);
  wait_stream(acl_context_0, 0);

  // 下载aclMat数据到Mat类中
  Mat dest1 = mv.data()[0].operator cv::Mat();
  Mat dest2 = mv.data()[1].operator cv::Mat();
  Mat dest3 = mv.data()[2].operator cv::Mat();

  imshow("dest1", dest1);
  imshow("dest2", dest2);
  imshow("dest3", dest3);

  aclMat acl_imgdest;

  merge(mv, acl_imgdest, 1);
  wait_stream(acl_context_0, 1);
  Mat imgdest = acl_imgdest.operator cv::Mat();
  imshow("imgdest", imgdest);
}

/**
 * @brief A demo of use a stream to synchronize multiple functions
 */
int test_func2(aclCxt *acl_context_0) {
  Mat src = imread("../cat1.jpg");
  if (src.empty()) {
    cerr << "src could not image !" << endl;
    return -1;
  }

  Mat src1 = imread("../cat1.jpg");
  if (src1.empty()) {
    cerr << "src1 could not image !" << endl;
    return -1;
  }

  // 待split操作数据上传数据到aclMat对象中
  aclMat acl_src(src, acl_context_0);
  aclMat acl_dest1;
  aclMat acl_dest2;
  aclMat acl_dest3;
  vector<aclMat> mv;
  mv.emplace_back(acl_dest1);
  mv.emplace_back(acl_dest2);
  mv.emplace_back(acl_dest3);

  // 待flip数据上传到aclMat对象中
  aclMat acl_flip_src(src1, acl_context_0);
  aclMat acl_flip_dest(src1.rows, src1.cols, src1.type(), acl_context_0);

  imshow("src", src);
  imshow("src1", src1);

  // 将函数挂载到1号stream上
  split(acl_src, mv, 1);
  flip(acl_flip_src, acl_flip_dest, 0, 1);

  // 等待1号stream中的任务执行完毕,stream内部按序执行任务
  wait_stream(acl_context_0, 1);

  // 下载split操作完成的数据
  Mat dest1 = mv.data()[0].operator cv::Mat();
  Mat dest2 = mv.data()[1].operator cv::Mat();
  Mat dest3 = mv.data()[2].operator cv::Mat();

  Mat flip_dest = acl_flip_dest.operator cv::Mat();

  imshow("split_dest1", dest1);
  imshow("split_dest2", dest2);
  imshow("split_dest3", dest3);

  imshow("flip_dest", flip_dest);
}

int main() {
  // 初始化
  aclCxt *acl_context_0 = set_device("../acl.json", 1, 2);

  test_func2(acl_context_0);

  // 去初始化
  release_device (acl_context_0);
  waitKey(0);

  return 0;
}
