/*
 * DataStatistics.cpp
 *
 *  Created on: Sep 17, 2013
 *      Author: bishan
 */

#include "DataStatistics.h"
#include <assert.h>

DataStatistics::DataStatistics() {
	// TODO Auto-generated constructor stub

}

DataStatistics::~DataStatistics() {
	// TODO Auto-generated destructor stub
}

void DataStatistics::OutputClusters(string filename) {
/*	ofstream outfile(filename.c_str(), ios::out);
	// Only output non-singleton clusters.
	for (map<int, Entity *>::iterator iter = corpus->gold_entities.begin(); iter != corpus->gold_entities.end(); ++iter) {
		Entity *en = iter->second;
		if (en->Size() <= 1) continue;
		outfile<<"Entity "<<iter->first<<" : "<<en->ToString()<<endl;
		outfile<<"Cluster "<<iter->first<<" : "<<endl;
		for (int j = 0; j < en->Size(); ++j) {
			outfile<<"Sentence "<<" : "<<
					corpus->documents[en->GetCorefMention(j)->DocID()]->GetSentence(en->GetCorefMention(j)->SentenceID())->toString()<<endl;
		}
		outfile<<endl;
	}

	outfile.close();
	*/
}

void DataStatistics::MentionStats(string filename) {
/*	int total_mention = 0;

	// Parse tag stats.
	map<string, int> parse_counts;

	for (map<int, Entity *>::iterator iter = corpus->gold_entities.begin(); iter != corpus->gold_entities.end(); ++iter) {
		Entity *en = iter->second;
		// For each mention in the entity, find information in the original document.
		for (int j = 0; j < en->Size(); ++j) {
			Mention *m = en->GetCorefMention(j);
			Sentence *sent = corpus->documents[m->DocID()]->GetSentence(m->SentenceID());
			string parse_tag = sent->GetParseTag(m);
			parse_counts[parse_tag] += 1;
		}
		total_mention += en->Size();
	}

	vector<pair<string, int> > parse_stats;
	for (map<string, int>::iterator iter = parse_counts.begin(); iter != parse_counts.end(); ++iter) {
		parse_stats.push_back(make_pair(iter->first, iter->second));
	}
	sort(parse_stats.begin(), parse_stats.end(), Utils::decrease_second<string, int>);

	// Sentence limit stats.
	map<int, int> sent_dist_counts;
	for (map<string, Document *>::iterator iter = corpus->documents.begin(); iter != corpus->documents.end(); ++iter) {
		Document *doc = iter->second;
		for (map<int, Entity *>::iterator it = doc->gold_entities.begin(); it != doc->gold_entities.end(); ++it) {
			Entity *en = it->second;
			for (int j = 1; j < en->Size(); ++j) {
				int dist = en->GetCorefMention(j)->SentenceID() -
						en->GetCorefMention(j-1)->SentenceID();
				assert(dist >= 0);
				if (dist >= 5) {
					cout<<doc->DocID()<<" "<<dist<<" mentions "<<en->GetCorefMention(j)->mention_str
							<<", "<<en->GetCorefMention(j-1)->mention_str<<endl;
				}
				sent_dist_counts[dist] += 1;
			}
		}
	}
	vector<pair<int, int> > dist_counts;
	for (map<int, int>::iterator iter = sent_dist_counts.begin(); iter != sent_dist_counts.end(); ++iter) {
		dist_counts.push_back(make_pair(iter->first, iter->second));
	}
	sort(dist_counts.begin(), dist_counts.end(), Utils::decrease_second<int, int>);

	ofstream outfile(filename.c_str(), ios::out);
	outfile<<"Total documents: "<<corpus->documents.size()<<endl;
	//outfile<<"Total entities: "<<corpus->gold_entities.size()<<endl;
	outfile<<"Total mention: "<<total_mention<<endl;
	outfile<<"Grammar stats: "<<endl;
	for (int i = 0; i < parse_stats.size(); ++i) {
		outfile<<parse_stats[i].first<<" "<<parse_stats[i].second<<endl;
	}
	outfile<<"Sentence dist: "<<endl;
	for (int i = 0; i < dist_counts.size(); ++i) {
		outfile<<dist_counts[i].first<<" "<<dist_counts[i].second<<endl;
	}
	outfile.close();
	*/
}
