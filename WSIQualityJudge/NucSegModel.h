#pragma once
#ifndef _NUCSEGMODEL_H_
#define _NUCSEGMODEL_H_
#include "model.h"

//针对256*256的结果
struct nucSegResult
{
	cv::Mat mat;
	vector<vector<cv::Point>> contours;//各个连通域的轮廓
	vector<double> area;//各个连通域的面积
	vector<cv::Rect> rects;//各个连通域的外接矩形
};

//核分割的mpp为0.293f

class NucSegModel : public model
{
private:
	float m_src_resolution;
	float m_dst_resolution;
	int m_batchsize;
public:
	NucSegModel();
	NucSegModel(modelConfig config, char* buffer, int size);
	void setSrcResolution(float);//相当于再次构造了
	void setDstResolution(float);
	void setBatchsize(int);
	~NucSegModel();

	//single batchsize tensor to single mat
	void tensor2Mat(Tensor& tensor, cv::Mat &mat);
	//multiple batchsize tensor to multiple images
	void tensor2Mat(const Tensor &tensorInput, vector<cv::Mat> &imgs);
	//get results from multiple batchsize tensor
	void resultOutput(const Tensor &tensor, vector<nucSegResult> &results);
	//get result from mask(8UC1)
	void resultOutput(cv::Mat &mask, nucSegResult &result);
	//get result from multiple mask(splice them together)
	void resultOutput(const vector<cv::Mat> &masks, const vector<cv::Rect> &rects, nucSegResult &result);
	//get result from multiple imgs(splice their mask together)
	void nucSegModelProcess(vector<cv::Mat> &imgs, const vector<cv::Rect> &rects, nucSegResult &result);
	//get result from multiple imgs
	void nucSegModelProcess(vector<cv::Mat> &imgs, vector<nucSegResult> &results);
};


#endif