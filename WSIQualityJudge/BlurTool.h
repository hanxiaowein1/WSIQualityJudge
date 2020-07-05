#pragma once

#include <numeric>
#include <opencv2/opencv.hpp>
#include <algorithm>

static const int KERNEL_NUM_THRE = 3;
static const int FORE_AREA_THRE = 8000;
static const int LAP_FORE_THRE = 100;

class BlurTool
{
	typedef unsigned char uc;
public:
	BlurTool();

	static double laplacian(cv::Mat , bool);

	static double laplacian_v1(cv::Mat, bool, int);

	static double laplacian_v2(cv::Mat&, bool, int);

	~BlurTool();

private:
	static const int erode_radius = 8;
};

