#include "precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::acl;
namespace cv {
namespace acl {
/**
 * @brief: matrix multiplication
 *
 */
void MatMul(const aclMat& src1, const aclMat& src2, aclMat& dest,
            int stream_id) {
  CV_Assert(src1.cols == src2.rows && src1.type() == src2.type());
  vector<aclMat> input_Mat;
  vector<aclMat> output_Mat;
  vector<aclDataBuffer*> inputBuffers_;
  vector<aclDataBuffer*> outputBuffers_;

  input_Mat.emplace_back(src1);
  input_Mat.emplace_back(src2);
  output_Mat.emplace_back(dest);

  inputBuffers_.emplace_back(aclCreateDataBuffer(src1.data, src1.totalSize));
  inputBuffers_.emplace_back(aclCreateDataBuffer(src2.data, src2.totalSize));
  inputBuffers_.emplace_back(aclCreateDataBuffer(nullptr, 0));
  outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));

  OperatorDesc opDesc =
      CreateOpDesc("MatMul", input_Mat, output_Mat, ACL_FORMAT_NHWC, TWO_DIMS);
  opDesc.AddInputTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
  opDesc.AddTensorAttr("transpose_x1", OP_BOOL, false);
  opDesc.AddTensorAttr("transpose_x2", OP_BOOL, false);
  compileAndRunop(opDesc, inputBuffers_, outputBuffers_, dest.acl_context,
                  stream_id);

  for (size_t i = 0; i < inputBuffers_.size(); i++)
    AclSafeCall(aclDestroyDataBuffer(inputBuffers_[i]));
  for (size_t i = 0; i < outputBuffers_.size(); i++)
    AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));
}

/**
 * @brief convolution
 * @param [in] src: characteristic matrix
 * @param [in] kernel: convolution kernel
 * @param [in] dest: destination matrix
 * @param [in] stridesList: strides, The N and C dimensions must be set to 1
 * @param [in] padSList: pads, vector<int64_t>(top, bottom, left, right)
 */
void Convolution(const aclMat& src, const aclMat& kernel, aclMat& dest,
                 const vector<int64_t>& stridesList,
                 const vector<int64_t>& padsList, int stream_id) {
  vector<aclDataBuffer*> inputBuffers_;
  vector<aclDataBuffer*> outputBuffers_;
  vector<int64_t> dilationsList {1, 1, 1, 1};
  string opType = "Conv2D";
  int dest_rows =
      (src.rows + padsList[0] + padsList[1] - (1 * (kernel.rows - 1) + 1)) /
          stridesList[2] +
      1;
  int dest_cols =
      (src.cols + padsList[2] + padsList[3] - (1 * (kernel.cols - 1) + 1)) /
          stridesList[3] +
      1;
  aclMat acl_dest {dest_rows, dest_cols, src.type(), src.acl_context};

  vector<int64_t> shape {1, 1, src.rows, src.cols};
  vector<int64_t> shape1 {1, 1, kernel.rows, kernel.cols};
  vector<int64_t> shape2 {1, 1, acl_dest.rows, acl_dest.cols};

  aclDataType dataType = type_transition(src.depth());
  aclFormat format = ACL_FORMAT_NCHW;
  OperatorDesc opDesc(opType);
  opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
  opDesc.AddInputTensorDesc(dataType, shape1.size(), shape1.data(), format);
  opDesc.AddOutputTensorDesc(dataType, shape2.size(), shape2.data(), format);

  auto opAttr = opDesc.opAttr;
  aclopSetAttrListInt(opAttr, "strides", stridesList.size(),
                      stridesList.data());
  aclopSetAttrListInt(opAttr, "pads", padsList.size(), padsList.data());
  aclopSetAttrListInt(opAttr, "dilations", dilationsList.size(),
                      dilationsList.data());

  inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));
  inputBuffers_.emplace_back(
      aclCreateDataBuffer(kernel.data, kernel.totalSize));
  outputBuffers_.emplace_back(
      aclCreateDataBuffer(acl_dest.data, acl_dest.totalSize));
  compileAndRunop(opDesc, inputBuffers_, outputBuffers_, src.acl_context,
                  stream_id);
  acl_dest.data = aclGetDataBufferAddr(outputBuffers_[0]);
  dest = acl_dest;

  for (size_t i = 0; i < inputBuffers_.size(); i++)
    AclSafeCall(aclDestroyDataBuffer(inputBuffers_[i]));
  for (size_t i = 0; i < outputBuffers_.size(); i++)
    AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));
}
} /* end of namespace acl */
} /* end of namespace cv */