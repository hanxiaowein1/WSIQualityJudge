#include "modelConf.h"

handle nucSegHandle;

void nucSegModelConf(string nucSegModelPath)
{
	modelConfig conf;
	conf.height = 256;
	conf.width = 256;
	conf.channel = 3;
	conf.opsInput = "input_1:0";
	conf.opsOutput.emplace_back("conv2d_24/Sigmoid:0");
	std::ifstream file(nucSegModelPath, std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	char *buffer = new char[size];
	file.seekg(0, std::ios::beg);
	if (!file.read(buffer, size)) {
		std::cout << "read file to buffer failed" << endl;
	}
	model *nucSeg = new model(conf, buffer, size);
	nucSegHandle = (handle)nucSeg;
	delete[]buffer;
}

void fuzzyModelConf(string fuzzyModelPath)
{
	//...
}