#include "NucSegModel.h"

int saveCount = 0;

NucSegModel::NucSegModel()
{
}

NucSegModel::NucSegModel(modelConfig config, char* buffer, int size) :model(config, buffer, size)
{

}

void NucSegModel::setSrcResolution(float srcResolution)
{
	m_src_resolution = srcResolution;
}

void NucSegModel::setDstResolution(float dstResolution)
{
	m_dst_resolution = dstResolution;
}

void NucSegModel::setBatchsize(int batchsize)
{
	m_batchsize = batchsize;
}

NucSegModel::~NucSegModel()
{

}

//将多个mask拼接到一起，得到一个nucSegResult
void NucSegModel::resultOutput(const vector<cv::Mat> &imgs, const vector<cv::Rect> &rects, nucSegResult &result)
{
	//将imgs整合成一张图像
	if (imgs.size() != rects.size())
	{
		cout << "resultOutput: imgs size shoule equal to rects size\n";
		return;
	}
	int bigWidth = rects[rects.size() - 1].x + rects[rects.size() - 1].width;
	int bigHeight = rects[rects.size() - 1].y + rects[rects.size() - 1].height;
	cv::Mat bigImg(bigHeight, bigWidth, CV_8UC1, cv::Scalar(0));
	for (int i = 0; i < imgs.size(); i++)
	{
		imgs[i].copyTo(bigImg(rects[i]));
	}
	//cv::imwrite("D:\\TEST_OUTPUT\\WSIQualityJudge\\masks\\" + to_string(saveCount) + ".tif", bigImg);
	//++saveCount;
	resultOutput(bigImg, result);
	return;
}
//mat: 8UC1
void NucSegModel::resultOutput(cv::Mat &mat, nucSegResult &result)
{
	int width = mat.cols;
	int height = mat.rows;
	vector<vector<cv::Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(mat, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	double threshold = 100;//面积的阈值
	vector<vector<cv::Point>> finalContours;
	for (int i = 0; i < contours.size(); i++)
	{
		vector<cv::Point> cont;
		cv::convexHull(contours[i], cont);
		double area = cv::contourArea(cont);
		if (area > threshold/* && areaHull / area < 1.1*/)
		{
			result.contours.emplace_back(contours[i]);
			result.area.emplace_back(int(area));
			Rect boundRect = cv::boundingRect(contours[i]);
			result.rects.emplace_back(boundRect);
			finalContours.emplace_back(contours[i]);
		}
	}
	if (finalContours.size() > 0)
	{
		cv::Mat finalMat(width, height, CV_8UC1, Scalar(0));
		cv::fillPoly(finalMat, finalContours, Scalar(255));
		result.mat = finalMat.clone();
	}
	return;
}

//这个函数只处理了batchsize为1的这种情况，现在需要修改，能够应对各种batchsize的tensor
void NucSegModel::resultOutput(const Tensor &tensor, vector<nucSegResult> &results)
{
	if (tensor.dims() != 4)
	{
		cout << "nucSegModel output size dims should be three...\n";
		return ;
	}
	int batchsize = tensor.dim_size(0);
	int height = tensor.dim_size(1);
	int width = tensor.dim_size(2);
	int channel = tensor.dim_size(3);
	vector<cv::Mat> imgs;
	tensor2Mat(tensor, imgs);
	for (int i = 0; i < imgs.size(); i++)
	{
		nucSegResult result;
		resultOutput(imgs[i], result);
		results.emplace_back(result);
	}
	return ;
}

void NucSegModel::tensor2Mat(const Tensor &tensorInput, vector<cv::Mat> &imgs)
{
	int tensorDims = tensorInput.dims();
	if (tensorDims != 4)
	{
		cout << "tensor2Mat: tensor size shoule be 4\n";
		return;
	}
	int batchsize = tensorInput.dim_size(0);
	int height = tensorInput.dim_size(1);
	int width = tensorInput.dim_size(2);
	auto tensorData = tensorInput.tensor<float, 4>();
	float *data = new float[height*width];
	for (int bs = 0; bs < batchsize; bs++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				data[h*width + w] = tensorData(bs, h, w, 0);
			}
		}
		cv::Mat mat = Mat(height, width, CV_32FC1, data).clone();
		cv::threshold(mat, mat, 0.5f, 255, THRESH_TOZERO);
		cv::normalize(mat, mat, 0, 255, NORM_MINMAX, CV_8UC1);
		imgs.emplace_back(mat);
	}
}

void NucSegModel::tensor2Mat(Tensor& tensor, cv::Mat &mat)
{
	int height = tensor.dim_size(1);
	int width = tensor.dim_size(2);
	auto tensordata = tensor.tensor<float, 4>();
	//为了以防万一，还是使用老的方法(遍历)
	tensorflow::DataType datatype = tensor.dtype();
	float *data = new float[height*width];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			data[i*width + j] = tensordata(0, i, j, 0);
		}
	}
	mat = Mat(height, width, CV_32FC1, data).clone();
	cv::threshold(mat, mat, 0.5f, 255, THRESH_TOZERO);
	cv::normalize(mat, mat, 0, 255, NORM_MINMAX, CV_8UC1);

	delete[]data;
}

void NucSegModel::nucSegModelProcess(vector<cv::Mat> &imgs, const vector<cv::Rect> &rects, nucSegResult &result)
{
	vector<tensorflow::Tensor> tensorOutput;
	this->output(imgs, tensorOutput);
	vector<cv::Mat> masks;
	tensor2Mat(tensorOutput[0], masks);
	resultOutput(masks, rects, result);
}

void NucSegModel::nucSegModelProcess(vector<cv::Mat> &imgs, vector<nucSegResult> &results)
{
	vector<tensorflow::Tensor> tensorOutput;
    this->output(imgs, tensorOutput);
    this->resultOutput(tensorOutput[0], results);
}