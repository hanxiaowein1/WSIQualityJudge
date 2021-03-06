#pragma once
#ifndef _TYPES_H_
#define _TYPES_H_
#include "opencv2/opencv.hpp"
//保存着model1的结果
typedef unsigned long long* handle;
struct model1Result
{
	float score;//model1的分数
	std::vector<cv::Point> points;//定位点
};

//一个512*512块的结果
struct regionResult
{
	cv::Point point;//全局坐标
	model1Result result;//model的结果
	vector<float> score2;//model2的结果
};

struct PointScore
{
	cv::Point point;
	float score;
};

//写到srp文件里的信息
typedef struct {
	int id;
	int x;
	int y;
	int type;
	double score;
}Anno;


#endif