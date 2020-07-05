#include "BlurTool.h"



BlurTool::BlurTool()
{

}


double BlurTool::laplacian_v2(cv::Mat &img, bool multi_channel = false, int thre_col = 20)
{
	//cv::imwrite("a.tif", img);
	std::cout << img << std::endl;
	unsigned char* binBuf = new unsigned char[img.cols*img.rows];
	unsigned char* pStart = (unsigned char*)img.datastart;
	unsigned char* pEnd = (unsigned char*)img.dataend;
	for (unsigned char* start = pStart; start < pEnd; start = start + 3)
	{
		//选择rgb元素中的最大最小值
		unsigned char R = *start;
		unsigned char G = *(start + 1);
		unsigned char B = *(start + 2);
		unsigned char maxValue = R;
		unsigned char minValue = R;
		if (maxValue < G)
			maxValue = G;
		if (maxValue < B)
			maxValue = B;
		if (minValue > G)
			minValue = G;
		if (minValue > B)
			minValue = B;
		if (maxValue - minValue > thre_col)
		{
			binBuf[(start - pStart) / 3] = 255;
		}
		else
		{
			std::cout << (start - pStart) / 3 << std::endl;
			binBuf[(start - pStart) / 3] = 0;
		}
	}
	cv::Mat binImg = cv::Mat(img.rows, img.cols, CV_8UC1, binBuf, cv::Mat::AUTO_STEP).clone();
	delete binBuf;
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(binImg, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	cv::Mat finalMat(binImg.rows, binImg.cols, CV_8UC1, cv::Scalar(0));
	cv::fillPoly(finalMat, contours, cv::Scalar(255));
	binImg = finalMat.clone();
	int erode_radius = 8;
	cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erode_radius, erode_radius));
	cv::erode(binImg, binImg, erode_kernel);
	int nonzeroCount = 0;
	//计算binImg中255的个数
	pStart = (unsigned char*)binImg.datastart;
	pEnd = (unsigned char*)binImg.dataend;
	for (unsigned char* start = pStart; start < pEnd; start++)
	{
		if (*start == 255)
		{
			nonzeroCount++;
		}
	}
	int binCols = binImg.cols;
	int binRows = binImg.rows;
	if (nonzeroCount < binCols*binRows*float(1 / 3))
		return -1;
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	cv::Mat lapMat;
	cv::Laplacian(grayImg, lapMat, CV_64F);
	double *lapStart = (double*)lapMat.datastart;
	double *lapEnd = (double*)lapMat.dataend;
	//计算binImg非零元素位置，在lapMat上的求和
	double lapBinSum = 0;
	for (unsigned char* start = pStart; start < pEnd; start++)
	{
		if (*start == 255)
		{
			int place = start - pStart;
			lapBinSum += *(lapStart + place);
		}
	}
	double mean_value = lapBinSum / nonzeroCount;
	for (double *start = lapStart; start < lapEnd; start++)
	{
		int place = start - lapStart;
		if (*(pStart + place) == 255)
			continue;
		else
			*(start) = *(start)+mean_value;
	}
	double sum = std::accumulate(lapStart, lapEnd, 0.0);
	double mean = sum / (lapEnd - lapStart);
	double accum = 0.0;
	std::for_each(lapStart, lapEnd, [&](const double d) {
		accum += (d - mean)*(d - mean);
	});
	accum = accum / (lapEnd - lapStart);//方差
	return accum;
}

double BlurTool::laplacian_v1(cv::Mat img, bool multi_channel = false, int thre_col = 20)
{
	assert(multi_channel);
	int cols = img.cols, rows = img.rows;
	cv::Mat binImg(rows, cols, CV_8UC1, cv::Scalar(0));
	uc* ps = (uc*)img.datastart;
	uc* pe = (uc*)img.dataend;
	uc* t = (uc*)binImg.datastart;
	for (unsigned char* start = ps; start < pe; start = start + 3)
	{
		//choose max and min
		uc R = *start , G = *(start + 1) , B = *(start + 2);
		uc maxValue = std::max(std::max(R, G), B);
		uc minValue = std::min(std::min(R, G), B);
		if (maxValue - minValue > thre_col)
		{
			*(t + (start - ps) / 3) = 255;
		}
	}

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hier;
	cv::findContours(binImg, contours, hier, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	cv::Mat finalMat(rows, cols, CV_8UC1, cv::Scalar(0));
	cv::fillPoly(finalMat, contours, cv::Scalar(255));
	binImg = finalMat.clone();
	cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erode_radius, erode_radius));
	cv::erode(binImg , binImg , erode_kernel);

	int nonzeroCount = 0;
	ps = (uc*)binImg.datastart;
	pe = (uc*)binImg.dataend;
	for (uc* start = ps; start < pe; start++)
	{
		if (*start == 255)
		{
			++nonzeroCount;
		}
	}
	if (nonzeroCount < cols * rows * 0.33333f/*( 1 / 3 )*/) //unsatisfied
	{
		return -1;
	}

	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	cv::Mat lapMat;
	cv::Laplacian(grayImg, lapMat, CV_64F);
	double *ls = (double*)lapMat.datastart;
	double *le = (double*)lapMat.dataend;
	
	//sum
	double lapBinSum = 0;
	for (uc* start = ps ; start < pe; start++)
	{
		if (*start == 255)
		{
			lapBinSum += *(ls + (start - ps));
		}
	}
	double mean_value = lapBinSum / nonzeroCount;
	double accum = 0.0;
	for (double* s = ls ; s < le ; ++s)
	{
		if(*(ps + (s - ls)) == 255)
		accum += std::pow(*s - mean_value, 2);
	}
	accum = accum / nonzeroCount ;
	return accum;
}

double BlurTool::laplacian(cv::Mat img, bool multi_channel = false)
{
	cv::Mat grayImg;
	if (multi_channel)
	{
		cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	}
	else
	{
		grayImg = img;
	}
	cv::Mat dst;
	cv::Laplacian(img, dst, CV_64F);
	double *pStart = (double*)dst.datastart;
	double *pEnd = (double*)dst.dataend;
	double sum = std::accumulate(pStart, pEnd, 0.0);
	double mean = sum / (pEnd - pStart);
	double accum = 0.0;
	std::for_each(pStart, pEnd, [&](const double d)
	{
		accum += (d - mean)*(d - mean);
	});
	accum = accum / (pEnd - pStart);//方差
	return accum;
}

BlurTool::~BlurTool()
{
}
