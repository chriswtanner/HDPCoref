/*
 * Config.h
 *
 *  Created on: Nov 28, 2013
 *      Author: bishan
 */

#ifndef SYSTEMCONFIG_H_
#define SYSTEMCONFIG_H_

#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <assert.h>
#include "./Parsing/Utils.h"

using namespace std;

class Config {
public:
	Config(string filename) {
		ifstream infile(filename.c_str(), ios::in);
		string str;
		while(getline(infile, str)) {
			if (str[0] == '#') continue;
			int index = str.find("=");
			if (index < 0) continue;

			string key = str.substr(0, index);
			string value = str.substr(index+1);
			props[key] = value;
		}
	}

	string GetProperty(string key) {
		if(props.find(key) == props.end()) {
			return "";
		}
		return props[key];
	}

	bool GetBoolProperty(string key) {
		string v = GetProperty(key);
		if (v == "") return false;
		assert(v == "true" || v == "false");
		return (v == "true");
	}

	double GetDoubleProperty(string key) {
		string v = GetProperty(key);
		if (v == "") return 0.0;
		return atof(v.c_str());
	}

	int GetIntProperty(string key) {
		string v = GetProperty(key);
		if (v == "") return 0;
		return atoi(v.c_str());
	}

	void SetProperty(string key, string value) {
		props[key] = value;
	}
private:
	map<string, string> props;
};



#endif /* SYSTEMCONFIG_H_ */
