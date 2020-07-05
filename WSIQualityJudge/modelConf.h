#pragma once

#ifndef _MODELCONF_H_
#define _MODELCONF_H_

#include <fstream>
#include "NucSegModel.h"

struct conf
{
	string nucSegModelPath;
	string fuzzyModelPath;
	string slidePath;
	string savePath;
};

typedef unsigned long long *handle;
void nucSegModelConf(string nucSegModelPath);

#endif