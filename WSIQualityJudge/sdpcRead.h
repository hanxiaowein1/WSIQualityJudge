#pragma once
#ifndef _SDPCREAD_H_
#define _SDPCREAD_H_

#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <mutex>
#include <vector>
#include <windows.h>
#include <thread>
#include "openslide.h"
#include "sdpcsdk.h"
using namespace std;
using namespace cv;

//将rect对象和mat对象绑定在一起
struct rect_mat
{
	cv::Rect rect;
	cv::Mat mat;
};

struct SlideHd {
	string path;//sdpc文件路径
	SdpcHandler h;
	openslide_t* osr;
};
//...好像并没有什么diao用
struct tile_sdpc {
	uint8_t *m_buffer;
	int m_size;     //buffer的大小
	cv::Rect m_rect;//保存坐标
	tile_sdpc(int g_size){
		g_size = m_size;
		m_buffer = new uint8_t[m_size];
	}
	~tile_sdpc()
	{
		delete m_buffer;
	}
};

struct openInfo {
	unsigned int L0_WIDTH;
	unsigned int L0_HEIGHT;//这里我设置为有效区域

	unsigned int L0_WIDTH_WITH_BORDER;//带边框的宽高
	unsigned int L0_HEIGHT_WITH_BORDER;

	unsigned int tileWidth;
	unsigned int tileHeight;

	unsigned int boundX;
	unsigned int boundY;
};

struct publicInfo {
	unsigned int L0_WIDTH;
	unsigned int L0_HEIGHT;

	unsigned int L0_WIDTH_WITH_BORDER;//带边框的宽高
	unsigned int L0_HEIGHT_WITH_BORDER;

	unsigned int tileWidth;
	unsigned int tileHeight;

	unsigned int boundX;
	unsigned int boundY;
};


class sdpcRead
{
public:
	//成员变量
	//queue<cv::Mat> matQueue;//存放图的列表
	queue<rect_mat> matRect_Queue;//绑定mat和rect
	queue<Rect> m_myRects;//裁图sdpc文件中的坐标
	queue<Rect> m_myRects_out;//Mat进入队列的顺序，与Mat保持一致。每一次m_myRects
	std::mutex matMutex;//对于mat队列所需要的锁
	std::mutex rectMutex;//对于rect所需要的锁

	string m_suffix;

	std::condition_variable cv_pop;//pop之后触发条件，然后通知线程把图放到队列

	int m_xOverlap;
	int m_yOverlap;

	SdpcInfo m_sdpcInfo;
	openInfo m_opensInfo;
	publicInfo m_publicInfo;

	vector<SlideHd> m_slideHd;
	
	vector<std::thread> m_threads;
	vector<bool> m_thread_flags;

	string m_sdpcPath;
	string m_opensPath;
	string m_gammaC_flag;
	unsigned int m_coreNum;

	/*
	m_readflag:	保存图像的选项
	必须在gamma变换中，该选项才起作用
	0:表示只保存gamma变换的图像
	1:表示先保存gamma变换前的图像，在保存gamma变换后的图像
	*/
	int m_readflag;

public:
	sdpcRead(string, string);
	~sdpcRead();

	//void iniMyRects();
	void m_GammaCorrection(Mat& src, Mat& dst, float fGamma);
	void m_szsq_normal(Mat& src, Mat& dst);
	bool enterMat(rect_mat rectMat);
	bool enterMat(vector<rect_mat> &rectMat);//针对一次性进入gamma前后的函数
	bool getRect(Rect *rect);
	bool popMat(rect_mat *rectMat);

	void getTile(SlideHd* phd, int y, int x, int tw, int th, cv::Mat & mat);
	void sdpc_EnterThread(int threadCount);
	void createThread();
	bool checkFlag();
	void getOpensInfo(openslide_t *osr);
	void enter_mat_upon_point(vector<Rect> &rects, string g_gammaC_flag = "ON", int g_readflag = 0);
	int get_os_property(openslide_t *slide, string propName);
	void resetPublicInfo(float resolution);
};


#endif // !_SDPCREAD_H_


