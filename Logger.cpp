/*
 * Logger.cpp
 *
 *  Created on: Sep 28, 2013
 *      Author: bishan
 */

#include "Logger.h"
#include <ctime>

Logger::~Logger() {
	// TODO Auto-generated destructor stub
}

Logger::Logger() {
	std::time_t now = std::time(NULL);
	std::tm * ptm = std::localtime(&now);
	char buffer[32];
	// Format: Mo, 15.06.2009 20:20:00
	std::strftime(buffer, 32, "./log/%d_%m_%Y_%H_%M_%S.txt", ptm);
	outfile.open(buffer, ios::out);
}

Logger::Logger(string filename) {
	outfile.open(filename.c_str(), ios::out);
}


