/*
 * DataStatistics.h
 *
 *  Created on: Sep 17, 2013
 *      Author: bishan
 */

#ifndef DATASTATISTICS_H_
#define DATASTATISTICS_H_

#include "CorefCorpus.h"
#include "Document.h"
#include "./Parsing/Utils.h"
#include <fstream>

using namespace std;

class DataStatistics {
public:
	DataStatistics();
	virtual ~DataStatistics();

	void OutputClusters(string filename);
	void MentionStats(string filename);

	CorefCorpus *corpus;
};

#endif /* DATASTATISTICS_H_ */
