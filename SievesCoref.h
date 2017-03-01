/*
 * CorefInference.h
 *
 *  Created on: Sep 16, 2013
 *      Author: bishan
 */

#ifndef SIEVESCOREF_H_
#define SIEVESCOREF_H_
#include "Document.h"
#include "CorefCorpus.h"

class SievesCoref {
public:
	SievesCoref();
	virtual ~SievesCoref();

	void ClusterWithinDoc(Document *doc);

	bool HLRule(Document *doc, vector<int> c1, vector<int> c2);
	bool HLorSRLRule(Document *doc, vector<int> c1, vector<int> c2);
	void Sieve_1(Document *doc);
	void Sieve_2(Document *doc);
	void Coreference(vector<Document*> doc);

	Document* MergeDocuments(vector<Document*> docs);

	void UpdateClusters();

public:
	vector<int> cluster_assignments;
	map<int, vector<int> > clusters;
	int max_cid;
};

#endif /* COREFINFERENCE_H_ */
