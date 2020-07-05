#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <io.h>
using namespace std;
typedef unsigned long long* handle;

void getFiles(string path, vector<string>& files, string suffix)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
		do {
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib & _A_SUBDIR)) {
				//如果是目录，则什么也不做
				/*if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				getFiles(p.assign(path).append("\\").append(fileinfo.name), files, suffix);*/
			}
			else {
				string tempFilename = string(fileinfo.name);
				//在这里做修改，使得能够适应任何长度的后缀名
				std::size_t pointFound = tempFilename.find_last_of(".");//发现最后一个"."的位置
				std::size_t length = tempFilename.length();
				size_t suffixLen = length - (pointFound + 1);
				string suffixGet = tempFilename.substr(pointFound + 1, suffixLen);
				if (suffixGet == suffix) {
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void testDll()
{
	typedef handle(*function1)(const char*);
	typedef bool(*function2)(handle handlePara, const char* slidePath);
	typedef void(*function3)(handle);
	function1 initialize_handle = nullptr;
	function2 slideProcess = nullptr;
	function3 freeMemory = nullptr;
	HINSTANCE sCheckDll = LoadLibraryA("WSIQualityJudge.dll");
	if (sCheckDll != nullptr)
	{
		initialize_handle = (function1)GetProcAddress(sCheckDll, "initialize_handle");
		slideProcess = (function2)GetProcAddress(sCheckDll, "slideProcess");
		freeMemory = (function3)GetProcAddress(sCheckDll, "freeModelMem");
	}
	string nucSegModelPath = "V:\\HanWei\\pb\\hw_asset\\seg\\segmentation_block380.pb";
	handle myHandle = initialize_handle(nucSegModelPath.c_str());
	string slidePath = "D:\\TEST_DATA\\srp\\complete\\051300060.srp";
	string srpPath = "D:\\TEST_DATA\\srp\\complete\\";
	vector<string> srp;
	getFiles(srpPath, srp, "srp");
	for (int i = 0; i < srp.size(); i++)
	{
		cout << srp[i] << " is processing" << endl;
		bool flag = slideProcess(myHandle, slidePath.c_str());
		if (flag)
		{
			cout << "happy\n";
		}
		else {
			cout << "sad\n";
		}
	}
	freeMemory(myHandle);
}

int main()
{
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	testDll();
	system("pause");
	return 0;
}