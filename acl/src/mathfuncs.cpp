#include "precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::acl;
namespace cv {
namespace acl {
aclMat abs(const aclMat &a, int stream_id) {
  aclMat dest(a.rows, a.cols, a.type(), a.acl_context);
  OneInAndOneOut(a, dest, "Abs", stream_id);
  return dest;
}

static void *power_data(double power, aclDataType type, size_t powersize) {
  void *dev_ptr;

  switch (type) {
    case ACL_UINT8: {
      aclrtMalloc(&dev_ptr, powersize, ACL_MEM_MALLOC_NORMAL_ONLY);
      uchar power_8u = uchar(power);
      aclrtMemcpy(dev_ptr, powersize, static_cast<void *>(&power_8u), powersize,
                  ACL_MEMCPY_HOST_TO_DEVICE);
      return dev_ptr;
    }
    case ACL_INT8: {
      aclrtMalloc(&dev_ptr, powersize, ACL_MEM_MALLOC_NORMAL_ONLY);
      char power_8s = char(power);
      aclrtMemcpy(dev_ptr, powersize, static_cast<void *>(&power_8s), powersize,
                  ACL_MEMCPY_HOST_TO_DEVICE);
      return dev_ptr;
    }
    case ACL_FLOAT16: {
      aclrtMalloc(&dev_ptr, powersize, ACL_MEM_MALLOC_NORMAL_ONLY);
      float16_t power_16f = float16_t(power);
      aclrtMemcpy(dev_ptr, powersize, static_cast<void *>(&power_16f),
                  powersize, ACL_MEMCPY_HOST_TO_DEVICE);
      return dev_ptr;
    }
    case ACL_INT32: {
      aclrtMalloc(&dev_ptr, powersize, ACL_MEM_MALLOC_NORMAL_ONLY);
      int power_32s = int(power);
      aclrtMemcpy(dev_ptr, powersize, static_cast<void *>(&power_32s),
                  powersize, ACL_MEMCPY_HOST_TO_DEVICE);
      return dev_ptr;
    }
    case ACL_FLOAT: {
      aclrtMalloc(&dev_ptr, powersize, ACL_MEM_MALLOC_NORMAL_ONLY);
      float power_32f = float(power);
      aclrtMemcpy(dev_ptr, powersize, static_cast<void *>(&power_32f),
                  powersize, ACL_MEMCPY_HOST_TO_DEVICE);
      return dev_ptr;
    }
    case ACL_DOUBLE: {
      aclrtMalloc(&dev_ptr, powersize, ACL_MEM_MALLOC_NORMAL_ONLY);
      double power_64f = double(power);
      aclrtMemcpy(dev_ptr, powersize, static_cast<void *>(&power_64f),
                  powersize, ACL_MEMCPY_HOST_TO_DEVICE);
      return dev_ptr;
    }
    default:
      return nullptr;
  }
}

void pow(const aclMat &src, double power, aclMat &dest, int stream_id) {
  vector<aclMat> input_Mat;
  vector<aclMat> output_Mat;
  vector<aclDataBuffer *> inputBuffers_;
  vector<aclDataBuffer *> outputBuffers_;

  aclDataType dataType = type_transition(src.depth());

  input_Mat.emplace_back(src);
  output_Mat.emplace_back(dest);

  OperatorDesc opDesc = CreateOpDesc("Pow", input_Mat, output_Mat);
  vector<int64_t> shape2 {1};
  opDesc.AddInputTensorDesc(dataType, shape2.size(), shape2.data(),
                            ACL_FORMAT_NHWC);

  size_t size = aclGetTensorDescSize(opDesc.inputDesc[1]);
  void *power_dev = power_data(power, dataType, size);

  inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));
  inputBuffers_.emplace_back(aclCreateDataBuffer(power_dev, size));

  outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));

  compileAndRunop(opDesc, inputBuffers_, outputBuffers_, dest.acl_context,
                  stream_id);

  aclrtFree(power_dev);
  for (size_t i = 0; i < inputBuffers_.size(); i++)
    AclSafeCall(aclDestroyDataBuffer(inputBuffers_[i]));
  for (size_t i = 0; i < outputBuffers_.size(); i++)
    AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));
}

void add(const aclMat &src, const aclMat &other_src, aclMat &dest,
         int stream_id) {
  bool is_correct;

  is_correct = (src.rows == other_src.rows);
  is_correct &= (src.rows == dest.rows);
  is_correct &= (src.cols == other_src.cols);
  is_correct &= (src.cols == dest.cols);
  is_correct &= (src.type() == other_src.type());
  is_correct &= (src.type() == dest.type());
  CV_Assert(is_correct);

  TwoInAndOneOut(src, other_src, dest, "Add", stream_id);
}

void divide(const aclMat &src, const aclMat &other_src, aclMat &dest,
            int stream_id) {
  bool is_correct;

  is_correct = (src.rows == other_src.rows);
  is_correct &= (src.rows == dest.rows);
  is_correct &= (src.cols == other_src.cols);
  is_correct &= (src.cols == dest.cols);
  is_correct &= (src.type() == other_src.type());
  is_correct &= (src.type() == dest.type());
  CV_Assert(is_correct);

  TwoInAndOneOut(src, other_src, dest, "Div", stream_id);
}

void exp(const aclMat &src, aclMat &dest, int stream_id) {
  CV_Assert(src.rows == dest.rows && src.cols == dest.cols &&
            src.type() == dest.type());

  vector<aclMat> input_Mat;
  vector<aclMat> output_Mat;

  vector<aclDataBuffer *> inputBuffers_;
  vector<aclDataBuffer *> outputBuffers_;

  input_Mat.emplace_back(src);
  output_Mat.emplace_back(dest);

  inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));
  outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));

  OperatorDesc opDesc = CreateOpDesc("Exp", input_Mat, output_Mat);
  opDesc.AddTensorAttr("base", OP_FLOAT, -1.0);
  opDesc.AddTensorAttr("scale", OP_FLOAT, 1.0);
  opDesc.AddTensorAttr("shift", OP_FLOAT, 0.0);

  compileAndRunop(opDesc, inputBuffers_, outputBuffers_, dest.acl_context,
                  stream_id);

  AclSafeCall(aclDestroyDataBuffer(inputBuffers_[0]));
  AclSafeCall(aclDestroyDataBuffer(outputBuffers_[0]));
}

void log(const aclMat &src, aclMat &dest, int stream_id) {
  CV_Assert(src.rows == dest.rows && src.cols == dest.cols &&
            src.type() == dest.type());

  vector<aclMat> input_Mat;
  vector<aclMat> output_Mat;

  vector<aclDataBuffer *> inputBuffers_;
  vector<aclDataBuffer *> outputBuffers_;

  input_Mat.emplace_back(src);
  output_Mat.emplace_back(dest);

  inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));
  outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));

  OperatorDesc opDesc = CreateOpDesc("Log", input_Mat, output_Mat);
  opDesc.AddTensorAttr("base", OP_FLOAT, -1.0);
  opDesc.AddTensorAttr("scale", OP_FLOAT, 1.0);
  opDesc.AddTensorAttr("shift", OP_FLOAT, 0.0);

  compileAndRunop(opDesc, inputBuffers_, outputBuffers_, dest.acl_context,
                  stream_id);

  AclSafeCall(aclDestroyDataBuffer(inputBuffers_[0]));
  AclSafeCall(aclDestroyDataBuffer(outputBuffers_[0]));
}

void max(const aclMat &src, const aclMat &other_src, aclMat &dest,
         int stream_id) {
  bool is_correct;

  is_correct = (src.rows == other_src.rows);
  is_correct &= (src.rows == dest.rows);
  is_correct &= (src.cols == other_src.cols);
  is_correct &= (src.cols == dest.cols);
  is_correct &= (src.type() == other_src.type());
  is_correct &= (src.type() == dest.type());
  CV_Assert(is_correct);

  TwoInAndOneOut(src, other_src, dest, "Maximum", stream_id);
}

void min(const aclMat &src, const aclMat &other_src, aclMat &dest,
         int stream_id) {
  bool is_correct;

  is_correct = (src.rows == other_src.rows);
  is_correct &= (src.rows == dest.rows);
  is_correct &= (src.cols == other_src.cols);
  is_correct &= (src.cols == dest.cols);
  is_correct &= (src.type() == other_src.type());
  is_correct &= (src.type() == dest.type());
  CV_Assert(is_correct);

  TwoInAndOneOut(src, other_src, dest, "Minimum", stream_id);
}

void sqrt(const aclMat &src, aclMat &dest, int stream_id) {
  CV_Assert(src.rows == dest.rows && src.cols == dest.cols &&
            src.type() == dest.type());

  OneInAndOneOut(src, dest, "Sqrt", stream_id);
}
} /* end of namespace acl */
} /* end of namespace cv */