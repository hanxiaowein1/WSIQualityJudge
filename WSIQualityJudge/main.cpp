/*
####测量一张片子的质量是否良好。
####测量标准：1）片子细胞核的个数;
			 2）片子模糊程度。
####测量方法：抽样采取评测。
*/

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>
#include <numeric>
#include <windows.h>
#include <cstdlib>
#include <time.h>
#include <map>
#include <memory>
#include "NucSegModel.h"
#include "sdpcRead.h"
#include "commonFunction.h"
#include "modelConf.h"
#include "SlideFactory.h"
#include "BlurTool.h"
using namespace std;

extern handle nucSegHandle;
extern int saveCount;
const int block_size = 1024;
const int fuzzy_radius = 30;
const double redun_ratio = 0.25f;
const int sp = 6;
const int out_sample_nums = 16;
const double pi = 3.1415926f;
const double target_ratio = 0.293f;
const int thre_cell_area = 8000;
const int thre_vol = 1000;
//static const int KERNEL_NUM_THRE = 3;
//static const int FORE_AREA_THRE = 8000;
//static const int LAP_FORE_THRE = 100;
map<string, int> thre_col_dict = { {"3D", 24}, {"our",30 },{"szsq" ,20} ,{"srp", 20} };

#ifdef _SLIDECHECK_EXPORT_
#define SLIDECHECK_API extern "C" __declspec(dllexport)
#else 
#define SLIDECHECK_API extern "C" __declspec(dllimport)
#endif // !TenCExport

SLIDECHECK_API handle initialize_handle(const char* nucSegModelPath);
SLIDECHECK_API bool slideProcess(handle handlePara, const char* slidePath);
SLIDECHECK_API void freeMemory(handle handlePara);

vector<cv::Rect> iniRects(const int &sWidth, const int &sHeight, const int &boundX, const int &boundY);
vector<cv::Rect> iniRects2(const int &rWidth, const int &rHeight);
//double computeLaplacian(cv::Mat &img, nucSegResult &result, int count , int &lap_nums);
//void process(string slidePath);
//void process2(string slidePath, string origin);
std::vector<double> process2(string slidePath, string origin, double c_ratio = 0.235f);
void computeLaplacian(cv::Mat& img, nucSegResult& result, int count, std::vector<double>& laps);
void post_bin_img(cv::Mat &binImg, double ratio = 0.293f);
void threshold_segmentation(cv::Mat &img, cv::Mat &binImg, int level, int thre_col = 24, int thre_vol = 1000);
void remove_small_objects(cv::Mat &binImg, int thre_vol = 15);
vector<int> get_rect(cv::Mat &binImg);
float myLaplacian(cv::Mat &img, cv::Mat &binImg, int thre_col);
double laplacianTest(cv::Mat &img , bool multi_channel = false)
{
	cv::Mat grayImg;
	if (multi_channel)
	{
		cv::cvtColor(img, grayImg, COLOR_BGR2GRAY);
	}
	else
	{
		grayImg = img;
	}
	cv::Mat dst;
	cv::Laplacian(img, dst, CV_64F/*, 1, 1.0, BORDER_DEFAULT*/);
	//cout << dst << endl;
	double *pStart = (double*)dst.datastart;
	double *pEnd = (double*)dst.dataend;
	double sum = std::accumulate(pStart, pEnd, 0.0);
	double mean = sum / (pEnd - pStart);
	double accum = 0.0;
	std::for_each(pStart, pEnd, [&](const double d) {
		accum += (d - mean)*(d - mean);
	});
	accum = accum / (pEnd - pStart);//方差
	return accum;
}

handle initialize_handle(const char* nucSegModelPath)
{
	nucSegModelConf(nucSegModelPath);
	return nucSegHandle;
}

bool slideProcess(handle handlePara, const char* slidePath)
{
	nucSegHandle = handlePara;
	vector<double> list;
	try
	{
		list = process2(string(slidePath), "srp");
		if (list.size() == 0)
			return false;
		if (list[list.size() - 1] == true)
		{
			return  true;
		}
	}
	catch (...)
	{
		cout << "quality srp failed\n";
		return false;
	}
	
}

void freeMemory(handle handlePara)
{
	nucSegHandle = handlePara;
	NucSegModel* NucSegModel_obj = (NucSegModel*)nucSegHandle;
	delete NucSegModel_obj;
}

vector<Rect> getRects(int srcImgWidth, int srcImgHeight, int dstImgWidth, int dstImgHeight, int m, int n)
{
	vector<Rect> myRects;
	//计算每次裁剪的间隔(hDirect,wDirect)
	int wDirect = (srcImgWidth - dstImgWidth) / (m - 1);
	int hDirect = (srcImgHeight - dstImgHeight) / (n - 1);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			int topValue = i * hDirect;
			int leftValue = j * wDirect;
			Rect myRect(leftValue, topValue, dstImgWidth, dstImgHeight);
			myRects.push_back(myRect);
		}
	}
	return myRects;
}

int main()
{
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	string nucSegModelPath = "X:\\HanWei\\pb\\hw_asset\\seg\\segmentation_block380.pb";
	handle myHandle = initialize_handle(nucSegModelPath.c_str());
	string slidePath = "D:\\TEST_DATA\\srp\\complete\\051300060.srp";
	string srpPath = "D:\\TEST_DATA\\srp\\broken\\";
	vector<string> srp;
	getFiles(srpPath, srp, "srp");
	for (int i = 0; i < srp.size(); i++)
	{
		cout << srp[i] << " is processing" << endl;
		bool flag = slideProcess(myHandle, srp[i].c_str());
		if (flag)
		{
			cout << "happy\n";
		}
		else {
			cout << "sad\n";
		}
	}

	system("pause");
	return 0;

}

std::vector<double> process2(string slidePath, string origin, double c_ratio)
{
	vector<double> ret;
	time_t now = time(0);
	cout << "start process2 " << (char*)ctime(&now);
	std::unique_ptr<SlideFactory> sFactory(new SlideFactory());
	std::unique_ptr<SlideRead> sRead = sFactory->createSlideProduct((char*)slidePath.c_str());
	if (!sRead->status())
		return ret;
	int slideWidth = 0;
	int slideHeight = 0;
	sRead->getSlideWidth(slideWidth);
	sRead->getSlideHeight(slideHeight);
	if (slideWidth == 0 || slideHeight == 0)
		return ret;
	//SlideFactory* sFactory = new SlideFactory();
	//SlideRead* sRead = sFactory->createSlideProduct((char*)slidePath.c_str());
	//取第6层级的图像
	int widthL6 = 0;
	int heightL6 = 0;
	sRead->getLevelDimensions(6, widthL6, heightL6);
	double mpp = 0.0f;
	sRead->getSlideMpp(mpp);
	c_ratio = mpp;
	cv::Mat img;
	sRead->getTile(6, 0, 0, widthL6, heightL6, img);
	cv::Mat binImg;
	threshold_segmentation(img, binImg, 0, thre_col_dict[origin], thre_vol);
	now = time(0);
	cout << "end threshold_segmentation " << (char*)ctime(&now);
	vector<int> tdlr = get_rect(binImg);
	for (auto iter = tdlr.begin(); iter != tdlr.end(); iter++)
	{
		*iter = *iter * (pow(2, 6));
	}
	int widthL0 = 0;
	int heightL0 = 0;
	sRead->getLevelDimensions(0, widthL0, heightL0);
	int c_x = widthL0 / 2;
	int c_y = heightL0 / 2;
	int ts = c_x < c_y ? c_x : c_y;
	int radius = block_size * (target_ratio / c_ratio) / 2;
	if (((tdlr[1] - tdlr[0]) > (c_y / 2) && (tdlr[1] - tdlr[0]) <= heightL0) &&
		((tdlr[3] - tdlr[2]) > c_x) && (tdlr[3] - tdlr[2]) <= widthL0)
	{
		tdlr[0] = tdlr[0] + radius;
		tdlr[1] = tdlr[1] - radius;
		tdlr[2] = tdlr[2] + radius;
		tdlr[3] = tdlr[3] - radius;
	}
	else
	{
		tdlr[0] = c_y - ts + radius;
		tdlr[1] = c_y + ts - radius;
		tdlr[2] = c_x - ts + radius;
		tdlr[3] = c_x + ts - radius;
	}
	//获得边框之后，开始获取区域的图像
	int v_dis = tdlr[3] - tdlr[2] + 1;
	int h_dis = tdlr[1] - tdlr[0] + 1;
	int center_x = v_dis / 2 + tdlr[2];
	int center_y = h_dis / 2 + tdlr[0];
	radius = v_dis > h_dis ? h_dis / 2 : v_dis / 2;
	vector<cv::Point> position_list;
	double dis = radius / (double)sp;
	for (int i = 1; i < sp; i++)
	{
		double cur_r = i * dis;
		int sample_nums = out_sample_nums * i * dis / radius;
		double ang_dis = 2 * pi / sample_nums;
		srand((unsigned)time(NULL));
		//double start_angle = double(((rand() % (100 - 0)) + 0) / 100.0f) * 2 * pi;
		double start_angle = 0;
		vector<double> angles;
		for (int j = 0; j < sample_nums; j++)
		{
			double angle_element = j * ang_dis + start_angle;
			angles.emplace_back(angle_element);
		}
		for (int j = 0; j < angles.size(); j++)
		{
			cv::Point point;
			point.x = center_x + cur_r * cos(angles[j]);
			point.y = center_y + cur_r * sin(angles[j]);
			position_list.emplace_back(point);
		}
	}
	position_list.emplace_back(cv::Point(center_x, center_y));

	vector<cv::Mat> blocks;

	int block_size_now = ceil(block_size * (target_ratio / c_ratio));
	for (auto iter = position_list.begin(); iter != position_list.end(); iter++)
	{
		cv::Mat tmpMat;
		sRead->getTile(0, iter->x - block_size_now / 2, iter->y - block_size_now / 2,
			block_size_now, block_size_now, tmpMat);
		cv::resize(tmpMat, tmpMat, cv::Size(block_size, block_size));
		blocks.emplace_back(std::move(tmpMat));
	}
	now = time(0);
	cout << "end read image " << (char*)ctime(&now);
	//遍历blocks，进行预测，得到结果
	vector<int> foreground_areas;
	vector<cv::Rect> rects2 = iniRects2(1024, 1024);
	NucSegModel* NucSegModel_obj = (NucSegModel*)nucSegHandle;
	double kernelNum = 0.0;
	double kernelArea = 0.0;
	std::vector<double> laps;
	int totalImgNum = blocks.size();
	int count = 0;
	for (int i = 0; i < blocks.size(); i++)
	{
		//保存block
		//cv::imwrite("E:\\VS2015_Project\\Test1\\WSI_Helper\\test_output\\imgs\\" + to_string(i) + ".tif", blocks[i]);
		threshold_segmentation(blocks[i], binImg, 0, thre_col_dict[origin], thre_vol);
		//cv::imwrite("E:\\VS2015_Project\\Test1\\WSI_Helper\\test_output\\imgs\\" + to_string(i) + "_b.tif", binImg);
		post_bin_img(binImg);
		//获得非零元素的个数（即面积）
		unsigned char* pStart = (unsigned char*)binImg.datastart;
		unsigned char* pEnd = (unsigned char*)binImg.dataend;
		int area = 0;
		for (unsigned char* start = pStart; start < pEnd; start++)
		{
			if (*start != 0)
				area++;
		}
		foreground_areas.emplace_back(area);
		//将图像送入模型进行运算
		vector<cv::Mat> imgs;
		for (auto iter = rects2.begin(); iter != rects2.end(); iter++)
		{
			imgs.emplace_back(blocks[i](*iter));
		}
		nucSegResult result;
		NucSegModel_obj->nucSegModelProcess(imgs, rects2, result);
		kernelNum += result.area.size();
		for (auto iter = result.area.begin(); iter != result.area.end(); iter++)
		{
			kernelArea += *iter;
		}
		computeLaplacian(blocks[i], result, count, laps);
		count++;
	}
	now = time(0);
	cout << "end process2 " << (char*)ctime(&now);
	std::sort(laps.begin(), laps.end());
	double mean_laps = 0;
	for (auto iter = 0; iter < laps.size() / 2; ++iter)
	{
		//std::cout << laps[iter] << std::endl;
		mean_laps += laps[iter];
	}
	mean_laps = mean_laps / (laps.size() / 2 + 1);

	kernelNum = kernelNum / totalImgNum;
	kernelArea = kernelArea / (totalImgNum * kernelNum);
	double mean_fore_area = std::accumulate(foreground_areas.begin(), foreground_areas.end(), 0.0) / foreground_areas.size();

	cout << slidePath << "\t" << kernelNum << "\t" << mean_laps << "\t" << kernelArea << "\t" << mean_fore_area << endl;

	double lap_fore = mean_laps * std::log10f(mean_fore_area);
	double satisfied = true;
	if (kernelNum <= KERNEL_NUM_THRE || mean_fore_area < FORE_AREA_THRE)
	{
		satisfied = false;
	}
	if (mean_fore_area < 30000 && lap_fore < LAP_FORE_THRE || mean_laps < 10)
	{
		satisfied = false;
	}

	return { kernelNum , mean_laps , kernelArea , mean_fore_area , lap_fore , satisfied };
}

void post_bin_img(cv::Mat &binImg, double ratio)
{
	int erode_radius = 10 * ratio / target_ratio;
	cv::Mat erode_kernel = cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(erode_radius, erode_radius));
	cv::erode(binImg, binImg, erode_kernel);

	int dilate_radius = 3 * ratio / target_ratio;
	cv::Mat dilate_kernel = cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(dilate_radius, dilate_radius));
	cv::dilate(binImg, binImg, dilate_kernel);

	vector<vector<cv::Point>> contours;
	cv::findContours(binImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<cv::Point>> finalContours;
	for (int i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);
		if (area > thre_cell_area)
			finalContours.emplace_back(contours[i]);
	}
	if (finalContours.size() > 0)
	{
		cv::Mat finalMat(binImg.rows, binImg.cols, CV_8UC1, Scalar(0));
		cv::fillPoly(finalMat, finalContours, Scalar(255));
		binImg = finalMat.clone();
	}	
}

void threshold_segmentation(cv::Mat &img, cv::Mat &binImg, int level, int thre_col, int thre_vol)
{
	//对img进行遍历，每三个unsigned char类型，选择其中的最大最小值
	std::unique_ptr<unsigned char[]> pBinBuf(new unsigned char[img.cols * img.rows]);
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
			pBinBuf[(start - pStart) / 3] = 255;
		}
		else
		{
			pBinBuf[(start - pStart) / 3] = 0;
		}
	}
	binImg = cv::Mat(img.rows, img.cols, CV_8UC1, pBinBuf.get(), cv::Mat::AUTO_STEP).clone();
	//对binImg二值图进行操作
	remove_small_objects(binImg, thre_vol / pow(2, level));
}
//获得binImg的上下左右非零元素的边界
vector<int> get_rect(cv::Mat &binImg)
{
	//cv::imwrite("D:\\TEST_OUTPUT\\WSIQualityJudge\\binImg.tif", binImg);
	int top = 20000, down = 0, left = 20000, right = 0;
	//找非零元素
	int width = binImg.cols;
	int height = binImg.rows;
	for (int h = 0; h < height; h++)
	{
		auto linePtr = binImg.ptr(h);
		for (int w = 0; w < width; w++)
		{
			if (*(linePtr+w) != 0)
			{
				left = w < left ? w : left;
				right = w > right ? w : right;
				top = h < top ? h : top;
				down = h > down ? h : down;
			}
		}
	}
	vector<int> tdlr;
	tdlr.emplace_back(top);
	tdlr.emplace_back(down);
	tdlr.emplace_back(left);
	tdlr.emplace_back(right);
	return tdlr;
}

void remove_small_objects(cv::Mat &binImg, int thre_vol)
{
	//去除img中小的区域
	vector<vector<cv::Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	double threshold = thre_vol;//面积的阈值
	vector<vector<cv::Point>> finalContours;
	for (int i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);
		if (area > threshold)
		{
			finalContours.emplace_back(contours[i]);
		}
	}
	if (finalContours.size() > 0)
	{
		cv::Mat finalMat(binImg.rows, binImg.cols, CV_8UC1, Scalar(0));
		cv::fillPoly(finalMat, finalContours, Scalar(255));
		binImg = finalMat.clone();
	}
}

float myLaplacian(cv::Mat &img, cv::Mat &binImg, int thre_col = 20)
{
	std::unique_ptr<unsigned char[]> pBinBuf(new unsigned char[img.cols * img.rows]);
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
			pBinBuf[(start - pStart) / 3] = 255;
		}
		else
		{
			pBinBuf[(start - pStart) / 3] = 0;
		}
	}
	binImg = cv::Mat(img.rows, img.cols, CV_8UC1, pBinBuf.get(), cv::Mat::AUTO_STEP).clone();
	vector<vector<cv::Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	cv::Mat finalMat(binImg.rows, binImg.cols, CV_8UC1, Scalar(0));
	cv::fillPoly(finalMat, contours, Scalar(255));
	binImg = finalMat.clone();
	int erode_radius = 8;
	cv::Mat erode_kernel = cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(erode_radius, erode_radius));
	cv::erode(binImg, binImg, erode_kernel);
	//cv::imwrite("./binImg.tif", binImg);
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
	if (nonzeroCount < binCols*binRows*float(1.0f / 3.0f))
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

void computeLaplacian(cv::Mat& img, nucSegResult& result, int count, std::vector<double>& laps)
{
	double laplace = 0.0f;
	int rows = img.rows;
	int cols = img.cols;
	for (int i = 0; i < result.contours.size(); i++)
	{
		vector<cv::Point> contour = result.contours[i];
		int centerX = 0;
		int centerY = 0;
		for (int j = 0; j < contour.size(); j++)
		{
			centerX = centerX + contour[j].x;
			centerY = centerY + contour[j].y;
		}
		centerX = centerX / contour.size();
		centerY = centerY / contour.size();
		cv::Point center(centerX, centerY);
		//取子图
		int top = 0, bottom = 0, left = 0, right = 0;
		top = center.y - 30;
		bottom = center.y + 30;
		left = center.x - 30;
		right = center.x + 30;
		if (top >= 0 && left >= 0 && bottom <= 1024 && right <= 1024)
		{
			cv::Rect rectMat(left, top, right - left, bottom - top);
			cv::Mat saveColorfulImg = img(rectMat);
			double lap = BlurTool::laplacian_v1(saveColorfulImg.clone(), true, 20);
			if (lap > 0)
			{
				laps.push_back(lap);
			}
		}
	}
}


//一张图像上的坐标
vector<cv::Rect> iniRects2(const int &rWidth, const int &rHeight)
{
	vector<cv::Rect> rects;
	int width = 256;
	int height = 256;
	int redun_size[2] = { 0.75f*width,0.75f*height };
	int ws = (rWidth - width) / redun_size[0] + 1;
	int hs = (rHeight - height) / redun_size[1] + 1;
	for (int i = 0; i < ws; i++) {
		for (int j = 0; j < hs; j++) {
			cv::Rect rect;
			rect.x = i * redun_size[0];
			rect.y = j * redun_size[1];
			rect.width = width;
			rect.height = height;
			rects.emplace_back(rect);
		}
	}
	return rects;
}

/*
****sWidth, sHeight:带有边框的宽高
****boundX, boundY:边框的起始点
*/
vector<cv::Rect> iniRects(const int &sWidth, const int &sHeight, const int &boundX, const int &boundY)
{
	vector<cv::Rect> rects;
	//gxb是根据sWidth和sHeight又从上下左右各去掉10个pixel
	int centerX = sWidth / 2;
	int centerY = sHeight / 2;
	int ts = centerX < centerY ? centerX : centerY;
	int left = centerX - ts + 10;
	int top = centerY - ts + 10;
	int right = centerX + ts - 10;
	int down = centerY + ts - 10;
	//根据sWidth和sHeight来选择边框
	float srcResolution = 0.1803f;
	float dstResolution = 0.293f;
	int width = 1024 * float(dstResolution / srcResolution);
	int height = width;
	int sp = 7;
	int verticalNum = (down - top) / sp;
	int horizontalNum = (right - left) / sp;
	for (int i = 1; i < sp; i++)
	{
		for (int j = 1; j < sp; j++)
		{
			if ((i == 1 && j == sp - 1) || (i == 1 && j == 1) || (j == 1 && i == sp - 1) || (i == sp - 1 && j == sp - 1))
				continue;
			cv::Rect rect;
			rect.y = i * horizontalNum + left - boundX;
			rect.x = j * verticalNum + top - boundY;
			rect.width = width;
			rect.height = height;
			rects.emplace_back(rect);
		}
	}
	return rects;
}