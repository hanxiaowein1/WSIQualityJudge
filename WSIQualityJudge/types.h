#pragma once
#ifndef _TYPES_H_
#define _TYPES_H_
#include "opencv2/opencv.hpp"
//������model1�Ľ��
typedef unsigned long long* handle;
struct model1Result
{
	float score;//model1�ķ���
	std::vector<cv::Point> points;//��λ��
};

//һ��512*512��Ľ��
struct regionResult
{
	cv::Point point;//ȫ������
	model1Result result;//model�Ľ��
	vector<float> score2;//model2�Ľ��
};

struct PointScore
{
	cv::Point point;
	float score;
};

//д��srp�ļ������Ϣ
typedef struct {
	int id;
	int x;
	int y;
	int type;
	double score;
}Anno;


#endif