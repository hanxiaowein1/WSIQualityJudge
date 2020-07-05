#include "sdpcRead.h"


void clear(std::queue<cv::Rect> &q)
{
	std::queue<cv::Rect> empty;
	std::swap(q, empty);
}

void clear(std::queue<rect_mat> &q)
{
	std::queue<rect_mat> empty;
	std::swap(q, empty);
}

sdpcRead::sdpcRead(string g_path, string g_suffix)
{	
	m_coreNum = 2;//线程改为固定值
	m_suffix = g_suffix;
	//m_gammaC_flag = g_gammaC_flag;
	if (m_suffix == "sdpc")
	{
		m_sdpcPath = g_path;		
		for (unsigned int i = 0; i < m_coreNum; i++)
		{
			SlideHd slidehd;
			slidehd.h = openSdpc((char*)m_sdpcPath.c_str());
			slidehd.path = m_sdpcPath;
			m_slideHd.emplace_back(slidehd);
		}
		getSdpcInfo(m_slideHd[0].h, &m_sdpcInfo);
		m_xOverlap = 620;
		m_yOverlap = 620;
		m_sdpcInfo.tileHeight = 6302;
		m_sdpcInfo.tileWidth = 3958;

		m_publicInfo.tileHeight = m_sdpcInfo.tileHeight;
		m_publicInfo.tileWidth = m_sdpcInfo.tileWidth;
		m_publicInfo.L0_HEIGHT = m_sdpcInfo.height;
		m_publicInfo.L0_WIDTH = m_sdpcInfo.width;
		m_publicInfo.L0_HEIGHT_WITH_BORDER = m_publicInfo.L0_HEIGHT;
		m_publicInfo.L0_WIDTH_WITH_BORDER = m_publicInfo.L0_WIDTH;
		m_publicInfo.boundX = 0;
		m_publicInfo.boundY = 0;
	}
	else
	{
		m_opensPath = g_path;
		for (unsigned int i = 0; i < m_coreNum; i++)
		{
			SlideHd slidehd;
			slidehd.osr = openslide_open(m_opensPath.c_str());
			slidehd.path = m_opensPath;
			m_slideHd.emplace_back(slidehd);
		}
		getOpensInfo(m_slideHd[0].osr);

		m_publicInfo.tileHeight = m_opensInfo.tileHeight;
		m_publicInfo.tileWidth = m_opensInfo.tileWidth;
		m_publicInfo.L0_HEIGHT = m_opensInfo.L0_HEIGHT;
		m_publicInfo.L0_WIDTH = m_opensInfo.L0_WIDTH;
		m_publicInfo.boundX = m_opensInfo.boundX;
		m_publicInfo.boundY = m_opensInfo.boundY;
	}
}

sdpcRead::~sdpcRead()
{
	//先释放线程
	for (int i = 0; i < m_threads.size(); i++){
		m_thread_flags[i] = false;
		m_threads[i].~thread();
	}
	//然后释放文件对象
	if (m_suffix == "sdpc"){
		for (int i = 0; i < m_slideHd.size(); i++){
			closeSdpc(m_slideHd[i].h);
		}
	}
	else{
		for (int i = 0; i < m_slideHd.size(); i++){
			openslide_close(m_slideHd[i].osr);
		}
	}

}

void sdpcRead::m_GammaCorrection(Mat& src, Mat& dst, float fGamma)
{
	unsigned char lut[256];
	for (int i = 0; i < 256; i++) {
		lut[i] = saturate_cast<uchar>(int(pow((float)(i / 255.0), fGamma) * 255.0f));
	}
	dst = src.clone();
	const int channels = dst.channels();
	switch (channels) {
	case 1: {
		MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			*it = lut[(*it)];
		break;
	}
	case 3: {
		for (int i = 0; i < dst.rows; i++) {
			uchar* linePtr = dst.ptr(i);
			for (int j = 0; j < dst.cols; j++) {
				*(linePtr + j * 3) = lut[*(linePtr + j * 3)];
				*(linePtr + j * 3 + 1) = lut[*(linePtr + j * 3 + 1)];
				*(linePtr + j * 3 + 2) = lut[*(linePtr + j * 3 + 2)];
			}
		}
		break;
	}
	}
}

void sdpcRead::m_szsq_normal(Mat& src, Mat& dst)
{
	m_GammaCorrection(src, dst, 0.6);
	/*cvtColor(src, src, COLOR_BGR2HSV);
	src.convertTo(src, CV_64FC3);
	for (int i = 0; i < src.rows; i++) {
		double* linePtr = (double*)src.ptr(i);
		for (int j = 0; j < src.cols; j++) {
			*(linePtr + j * 3) = (*(linePtr + j * 3)*0.973 + 7) > 255 ? 255 : (*(linePtr + j * 3)*0.973 + 7);
			*(linePtr + j * 3 + 1) = (*(linePtr + j * 3 + 1)*0.929 + 18) > 255 ? 255 : (*(linePtr + j * 3 + 1)*0.929 + 18);
			*(linePtr + j * 3 + 2) = (*(linePtr + j * 3 + 2)*0.976 + 6) > 255 ? 255 : (*(linePtr + j * 3 + 2)*0.976 + 6);
			*(linePtr + j * 3) = (unsigned long long)(*(linePtr + j * 3));
			*(linePtr + j * 3 + 1) = (unsigned long long)(*(linePtr + j * 3 + 1));
			*(linePtr + j * 3 + 2) = (unsigned long long)(*(linePtr + j * 3 + 2));
		}
	}
	src.convertTo(src, CV_8UC3);
	cvtColor(src, src, COLOR_HSV2BGR);
	dst = src.clone();*/
}


//void sdpcRead::iniMyRects()
//{
//	std::unique_lock<std::mutex> lock(rectMutex);
//	m_sdpcInfo.L0_nx = (m_sdpcInfo.width - m_sdpcInfo.tileWidth) / (m_sdpcInfo.tileWidth - m_xOverlap) + 1;
//	m_sdpcInfo.L0_ny = (m_sdpcInfo.height - m_sdpcInfo.tileHeight) / (m_sdpcInfo.tileHeight - m_yOverlap) + 1;
//	for (int i = 0; i < m_sdpcInfo.L0_nx; i++)
//	{
//		for (int j = 0; j < m_sdpcInfo.L0_ny; j++)
//		{
//			int x = i*(m_sdpcInfo.tileWidth - m_xOverlap);
//			int y = j*(m_sdpcInfo.tileHeight - m_yOverlap);
//			Rect rect(x, y, m_sdpcInfo.tileWidth, m_sdpcInfo.tileHeight);
//			m_myRects.push(rect);
//		}
//	}
//}

bool sdpcRead::enterMat(vector<rect_mat> &rectMats)
{
	std::unique_lock<std::mutex> myGuard(matMutex);
	if (matRect_Queue.size() >= 200) {
		cv_pop.wait(myGuard, [this] {
			if (matRect_Queue.size() < 200) {
				return true;
			}
			else {
				return false;
			}
		});
		for (int i = 0; i < rectMats.size(); i++)
		{
			matRect_Queue.emplace(rectMats[i]);
		}	
		return true;
	}
	else
	{
		//直接push进去
		for (int i = 0; i < rectMats.size(); i++)
		{
			matRect_Queue.emplace(rectMats[i]);
		}
		return true;
	}
}

//已经得到mat，准备输入队列中
bool sdpcRead::enterMat(rect_mat rectMat)
{
	std::unique_lock<std::mutex> myGuard(matMutex);
	if (matRect_Queue.size()>=200) {
		cv_pop.wait(myGuard, [this] {
			if (matRect_Queue.size() < 200) {
				return true;
			}
			else {
				return false;
			}
		});
		matRect_Queue.emplace(rectMat);
		return true;
	}
	else
	{
		//直接push进去
		matRect_Queue.emplace(rectMat);
		return true;
	}
	//int count = 0;
	//while (true)
	//{	
	//	{
	//		std::unique_lock<std::mutex> myGuard(matMutex);
	//		cout << "the mat size is(...):" << matRect_Queue.size() << endl;
	//		if (matRect_Queue.size() >= 20) {//只有大于100的时候才会进入wait堵塞线程
	//			cout << "size bigger than 20, enter if" << endl;
	//			//这里先释放掉锁，在Sleep
	//			myGuard.unlock();
	//			cout << "sleep" << endl;
	//			Sleep(100);
	//			count++;
	//			if (count > 1000) {
	//				return false;//超过100秒则返回false
	//			}
	//		}
	//		else {
	//			cout << "the mat size is(else)" << matRect_Queue.size() << endl;
	//			break;
	//		}
	//	}
	//}	
	//std::unique_lock<std::mutex> myGuard(matMutex);
	//matRect_Queue.emplace(rectMat);
	//return true;
}


bool sdpcRead::getRect(Rect *rect)
{
	if (m_myRects.size() > 0)
	{//双重锁定
		std::lock_guard<mutex> myGuard(rectMutex);
		if (m_myRects.size() > 0) {
			//存在框框可以使用
			*rect = m_myRects.front();
			m_myRects.pop();
			return true;
		}
		else {
			return false;
		}
	}
	else{
		return false;
	}
}

bool sdpcRead::popMat(rect_mat *rectMat)
{
	/*
	1.首先判断是否有线程在运行
	2.在判断是否mat队列中是否有图
	*/
	if (checkFlag()) {
		std::unique_lock<mutex> myMatGuard(matMutex);
		if (matRect_Queue.size() > 0) {
			*rectMat = matRect_Queue.front();
			matRect_Queue.pop();
			cv_pop.notify_one();//只要pop之后，就开始notify
			return true;
		}
		else {
			//一直等待，直到mat队列中有数据
			//一直等待才会卡住，在这里设置超过100秒不等待
			//发现Bug：如果checkFlag之后线程刚刚结束，那么就会永远的死循环在这里，所以，在while的内部也加上checkFlag检查线程是否继续运行
			int count = 0;
			myMatGuard.unlock();
			while (true) {
				Sleep(10000);
				count++;
				if (count > 10)
					break;
				if (!checkFlag())
				{
					//如果线程已经停止了
					return popMat(rectMat);//再次调用自己即可
				}
				myMatGuard.lock();
				if (matRect_Queue.size() > 0) {
					*rectMat = matRect_Queue.front();
					matRect_Queue.pop();
					//在pop之后唤醒线程进入队列
					cv_pop.notify_one();
					return true;
				}
				myMatGuard.unlock();
			}
			cout << "something wrong in popMat\n";
			return false;
		}
	}
	else {
		std::unique_lock<mutex> myMatGuard(matMutex);
		if (matRect_Queue.size() > 0) {
			*rectMat = matRect_Queue.front();
			matRect_Queue.pop();
			//cv_pop.notify_one();//这个时候thread都已经退出了，notify_one()不起作用，添上可能会出现其他的错误
			return true;
		}
		else {
			cout << "mat queue is empty\n";
			return false;
		}
	}

}

void sdpcRead::getTile(SlideHd* phd, int x, int y, int tw, int th, cv::Mat & mat) {
	if (m_suffix == "sdpc")
	{
		uint8_t* buf = new uint8_t[tw * th * 3];
		::getTile(phd->h, 0, y, x, tw, th, buf);//调用全局的函数
		mat = cv::Mat(th, tw, CV_8UC3, buf, cv::Mat::AUTO_STEP).clone();
		delete buf;
	}
	else
	{
		uint8_t* buf = new uint8_t[tw * th * 4];
		openslide_read_region(phd->osr, (uint32_t*)buf, m_publicInfo.boundX + x, m_publicInfo.boundY + y, 0, tw, th);
		cv::Mat image = cv::Mat(th, tw, CV_8UC4, buf, cv::Mat::AUTO_STEP).clone();
		cvtColor(image, mat, COLOR_RGBA2RGB);//要转换一下通道
		delete buf;
	}
	
}

void sdpcRead::sdpc_EnterThread(int threadCount){
	Rect rect;
	while (getRect(&rect) && m_thread_flags[threadCount]){
		Mat src;
		getTile(&m_slideHd[threadCount], rect.x, rect.y, rect.width, rect.height, src);
		Mat dst;
		if (m_gammaC_flag == "ON") {
			m_szsq_normal(src, dst);
		}
		else {
			dst = src;
		}
		if (m_readflag == 0)
		{
			rect_mat rectMat;
			rectMat.mat = dst;
			rectMat.rect = rect;
			bool flag = enterMat(rectMat);
			if (flag == false) {
				std::thread::id this_id = std::this_thread::get_id();
				cout << "thread " << this_id << " exit with error" << endl;
				return;
			}
		}
		if (m_readflag == 1)
		{
			rect_mat rectMat;
			rectMat.mat = src;
			rectMat.rect = rect;
			rect_mat rectMat2;
			rectMat2.mat = dst;
			rectMat2.rect = rect;
			vector<rect_mat> rectMats;
			rectMats.emplace_back(rectMat);
			rectMats.emplace_back(rectMat2);
			bool flag = enterMat(rectMats);
			if (flag == false) {
				std::thread::id this_id = std::this_thread::get_id();
				cout << "thread " << this_id << " exit with error" << endl;
				return;
			}
		}
	}
	std::thread::id this_id = std::this_thread::get_id();
	cout << "thread " << this_id << " exit" << endl;
	m_thread_flags[threadCount] = false;
}

void sdpcRead::createThread()
{
	for (int i = 0; i < m_coreNum; i++){
		m_thread_flags.emplace_back(true);
	}
	for (int i = 0; i < m_coreNum; i++){
		//创建线程
		m_threads.emplace_back(std::thread(&sdpcRead::sdpc_EnterThread, this, i));
	}
	for (auto& t : m_threads){
		t.detach();
	}	
}

bool sdpcRead::checkFlag(){
	for (int i = 0; i < m_thread_flags.size(); i++){
		if (m_thread_flags[i] == true)
			return true;
	}
	return false;
}

int sdpcRead::get_os_property(openslide_t *slide, string propName)
{
	const char *property = openslide_get_property_value(slide, propName.c_str());
	if (property == NULL) {
		return 0;
	}

	stringstream strValue;
	strValue << property;
	int intValue;
	strValue >> intValue;

	return intValue;
}

void sdpcRead::resetPublicInfo(float resolution)
{
	m_publicInfo.tileHeight = float(0.586f / resolution) * 1936;//都是向下取整，这个没问题
	m_publicInfo.tileWidth = float(0.586f / resolution) * 1216;
}

void sdpcRead::getOpensInfo(openslide_t *osr)
{
	//int64_t w = 0, h = 0;
	//openslide_get_level0_dimensions(osr, &w, &h);
	//咳咳，这里还需要分情况讨论
	if (m_suffix == "mrxs")
	{
		m_opensInfo.L0_WIDTH = get_os_property(osr, "openslide.bounds-width");//去掉边框的宽
		m_opensInfo.L0_HEIGHT = get_os_property(osr, "openslide.bounds-height");//去掉边框的高
		int64_t height = 0;
		int64_t width = 0;
		openslide_get_level0_dimensions(osr, &width, &height);
		m_opensInfo.L0_HEIGHT_WITH_BORDER = height;
		m_opensInfo.L0_WIDTH_WITH_BORDER = width;
		m_opensInfo.boundX = get_os_property(osr, "openslide.bounds-x");//x方向有效区域起始点
		m_opensInfo.boundY = get_os_property(osr, "openslide.bounds-y");//y方向有效区域起始点

	}
	else
	{
		int64_t w = 0, h = 0;
		openslide_get_level0_dimensions(osr, &w, &h);
		m_opensInfo.boundX = 0;
		m_opensInfo.boundY = 0;
		m_opensInfo.L0_HEIGHT = h;
		m_opensInfo.L0_WIDTH = w;
	}
	m_opensInfo.tileHeight = 6302;
	m_opensInfo.tileWidth = 3958;
}

void sdpcRead::enter_mat_upon_point(vector<Rect> &rects, string g_gammaC_flag, int flag)
{
	
	m_gammaC_flag = g_gammaC_flag;
	m_readflag = flag;
	if (m_gammaC_flag == "OFF" && m_readflag == 1)
	{//gamma变换关了之后，m_readflag不可能为1
		cout << "gamma off but readflag is 1, set readflag 0\n";
		m_readflag = 0;
	}

	m_threads.clear();
	m_thread_flags.clear();
	clear(m_myRects);
	clear(matRect_Queue);

	for (int i = 0; i < rects.size(); i++) {
		m_myRects.push(rects[i]);
	}
	//然后在重新创建线程
	for (int i = 0; i < m_coreNum; i++) {
		m_thread_flags.emplace_back(true);
	}
	for (int i = 0; i < m_coreNum; i++) {
		//创建线程
		m_threads.emplace_back(std::thread(&sdpcRead::sdpc_EnterThread, this, i));
	}
	for (auto& t : m_threads) {
		t.detach();
	}
}
