/*
 * CorefEval.h
 *
 *  Created on: Mar 12, 2013
 *      Author: bishan
 */

#ifndef COREFEVAL_H_
#define COREFEVAL_H_

#include "CorefCorpus.h"
#include "Document.h"
#include "Logger.h"

#include "./Scorer/MUCScore.h"
#include "./Scorer/BcubeScore.h"
#include "./Scorer/PairwiseScore.h"
#include "./Scorer/MentionScore.h"

#include "./Parsing/Utils.h"

using namespace std;

namespace InputType{
enum InputType {gold, predict, corefpredict};
}

class CorefEval {
public:
	CorefEval();
	virtual ~CorefEval();
	Scorer* MucS;
	Scorer* BcubeS;
	Scorer* PairS;
	Scorer* mentionS;

	CorefCorpus *corpus;
public:
	void doWDScore();
	void doCDScore();

	void outputTopicScore(string outputfile);

	void printAccumulateScore(Logger &log);
	void printAccumulateScore();

	void ErrorAnalysis(Logger &log);

	void ClusterInfo(map<int, Entity*> &entities);
};

#endif /* COREFEVAL_H_ */
