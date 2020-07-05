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

//��rect�����mat�������һ��
struct rect_mat
{
	cv::Rect rect;
	cv::Mat mat;
};

struct SlideHd {
	string path;//sdpc�ļ�·��
	SdpcHandler h;
	openslide_t* osr;
};
//...����û��ʲôdiao��
struct tile_sdpc {
	uint8_t *m_buffer;
	int m_size;     //buffer�Ĵ�С
	cv::Rect m_rect;//��������
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
	unsigned int L0_HEIGHT;//����������Ϊ��Ч����

	unsigned int L0_WIDTH_WITH_BORDER;//���߿�Ŀ��
	unsigned int L0_HEIGHT_WITH_BORDER;

	unsigned int tileWidth;
	unsigned int tileHeight;

	unsigned int boundX;
	unsigned int boundY;
};

struct publicInfo {
	unsigned int L0_WIDTH;
	unsigned int L0_HEIGHT;

	unsigned int L0_WIDTH_WITH_BORDER;//���߿�Ŀ��
	unsigned int L0_HEIGHT_WITH_BORDER;

	unsigned int tileWidth;
	unsigned int tileHeight;

	unsigned int boundX;
	unsigned int boundY;
};


class sdpcRead
{
public:
	//��Ա����
	//queue<cv::Mat> matQueue;//���ͼ���б�
	queue<rect_mat> matRect_Queue;//��mat��rect
	queue<Rect> m_myRects;//��ͼsdpc�ļ��е�����
	queue<Rect> m_myRects_out;//Mat������е�˳����Mat����һ�¡�ÿһ��m_myRects
	std::mutex matMutex;//����mat��������Ҫ����
	std::mutex rectMutex;//����rect����Ҫ����

	string m_suffix;

	std::condition_variable cv_pop;//pop֮�󴥷�������Ȼ��֪ͨ�̰߳�ͼ�ŵ�����

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
	m_readflag:	����ͼ���ѡ��
	������gamma�任�У���ѡ���������
	0:��ʾֻ����gamma�任��ͼ��
	1:��ʾ�ȱ���gamma�任ǰ��ͼ���ڱ���gamma�任���ͼ��
	*/
	int m_readflag;

public:
	sdpcRead(string, string);
	~sdpcRead();

	//void iniMyRects();
	void m_GammaCorrection(Mat& src, Mat& dst, float fGamma);
	void m_szsq_normal(Mat& src, Mat& dst);
	bool enterMat(rect_mat rectMat);
	bool enterMat(vector<rect_mat> &rectMat);//���һ���Խ���gammaǰ��ĺ���
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


