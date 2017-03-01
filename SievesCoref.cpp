/*
 * CorefInference.cpp
 *
 *  Created on: Sep 16, 2013
 *      Author: bishan
 */

#include "SievesCoref.h"

SievesCoref::SievesCoref() {
	// TODO Auto-generated constructor stub

}

SievesCoref::~SievesCoref() {
	// TODO Auto-generated destructor stub
}

void SievesCoref::ClusterWithinDoc(Document *doc) {

}

Document* SievesCoref::MergeDocuments(vector<Document*> docs) {
	// Concatenate predict mentions.
	Document *doc = new Document();
	for (int i = 0; i < docs.size(); ++i) {
		for (int j = 0; j < docs[i]->SentNum(); ++j) {
			//if (docs[i]->GetSentence(j)->gold_mentions.size() == 0) continue;
			for (int k = 0; k < docs[i]->GetSentence(j)->predict_mentions.size(); ++k) {
				doc->predict_mentions.push_back(docs[i]->GetSentence(j)->predict_mentions[k]);
			}
		}
	}
	return doc;
};

void SievesCoref::Coreference(vector<Document*> docs) {
	Document *doc = MergeDocuments(docs);

	cout<<"mention size "<<doc->predict_mentions.size()<<endl;

	// Initially every mention is one cluster.
	cluster_assignments.clear();
	cluster_assignments.resize(doc->predict_mentions.size());
	for (int i = 0; i < cluster_assignments.size(); ++i) {
		cluster_assignments[i] = i;
	}
	UpdateClusters();

	Sieve_1(doc);
	//Sieve_2(doc);

	/*cout<<clusters.size()<<" "<<max_cid<<endl;
	for (int x = 0; x < clusters.size(); ++x) {
		if (clusters[x].size() == 0) continue;
		cout<<"cluster "<<x<<": ";
		for (int y = 0; y < clusters[x].size(); ++y) {
			cout<<doc->predict_mentions[clusters[x][y]]->ToString()<<", ";
		}
		cout<<endl;
	}*/

	for (int i = 0; i < doc->predict_mentions.size(); ++i) {
		doc->predict_mentions[i]->pred_entity_id = cluster_assignments[i];
	}

	delete doc;
}

bool SievesCoref::HLRule(Document *doc, vector<int> c1, vector<int> c2) {
	for (int i = 0; i < c1.size(); ++i) {
		for (int j = 0; j < c2.size(); ++j) {
			if (doc->predict_mentions[c1[i]]->HeadMatch(doc->predict_mentions[c2[j]])) {
				return true;
			}
		}
	}
	return false;
}

bool SievesCoref::HLorSRLRule(Document *doc, vector<int> c1, vector<int> c2) {
	for (int i = 0; i < c1.size(); ++i) {
		for (int j = 0; j < c2.size(); ++j) {
			if (doc->predict_mentions[c1[i]]->HeadMatch(doc->predict_mentions[c2[j]])) {
				return true;
			}
			if (doc->predict_mentions[c1[i]]->MatchSrlArguments(doc->predict_mentions[c2[j]])) {
				return true;
			}
		}
	}
	return false;
}

void SievesCoref::UpdateClusters() {
	clusters.clear();
	max_cid = 0;
	for (int i = 0; i < cluster_assignments.size(); ++i) {
		int c = cluster_assignments[i];
		if (c > max_cid) {
			max_cid = c;
		}
		if (clusters.find(c) == clusters.end()) {
			vector<int> p;
			clusters[c] = p;
		}
		clusters[c].push_back(i);
	}
	max_cid++;
}

void SievesCoref::Sieve_1(Document *doc) {
	while(true) {
		// Each iteration, pick the two best clusters to merge
		int i = 0;
		int j = 0;
		bool found = false;
		for (i = 0; i < clusters.size(); ++i) {
			for (j = 0; j < i; ++j) {
				if (HLRule(doc, clusters[i], clusters[j])) {
					found = true;
					break;
				}
			}
			if (found) {
				// Merging clusters i and j.
				for (int x = 0; x < clusters[i].size(); ++x) {
					cluster_assignments[clusters[i][x]] = max_cid;
				}
				for (int x = 0; x < clusters[j].size(); ++x) {
					cluster_assignments[clusters[j][x]] = max_cid;
				}
				UpdateClusters();

				// output cluster
				/*cout<<clusters.size()<<" "<<max_cid<<endl;
				for (int x = 0; x < clusters.size(); ++x) {
					cout<<"cluster "<<x<<": ";
					for (int y = 0; y < clusters[x].size(); ++y) {
						cout<<doc->predict_mentions[clusters[x][y]]->ToString()<<", ";
					}
					cout<<endl;
				}*/
				break;
			}
		}
		if (!found) break;
	}
}

void SievesCoref::Sieve_2(Document *doc) {
	while(true) {
		// Each iteration, pick the two best clusters to merge
		int i = 0;
		int j = 0;
		bool found = false;
		for (i = 0; i < clusters.size(); ++i) {
			for (j = 0; j < i; ++j) {
				if (HLorSRLRule(doc, clusters[i], clusters[j])) {
					found = true;
					break;
				}
			}
			if (found) {
				// Merging clusters i and j.
				for (int x = 0; x < clusters[i].size(); ++x) {
					cluster_assignments[clusters[i][x]] = max_cid;
				}
				for (int x = 0; x < clusters[j].size(); ++x) {
					cluster_assignments[clusters[j][x]] = max_cid;
				}
				UpdateClusters();
				break;
			}
		}
		if (!found) break;
	}
}
