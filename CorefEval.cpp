/*
 * CorefEval.cpp
 *
 *  Created on: Mar 12, 2013
 *      Author: bishan
 */

#include "CorefEval.h"

CorefEval::CorefEval() {
	// TODO Auto-generated constructor stub
	MucS = new MUCScore();
	BcubeS = new BcubeScore(Bconll);
	PairS = new PairwiseScore();
	mentionS = new MentionScore();
}

CorefEval::~CorefEval() {
	// TODO Auto-generated destructor stub
}

void CorefEval::doCDScore()
{
	MucS->clear();
	BcubeS->clear();
	PairS->clear();
	mentionS->clear();
	cout << "cross-doc" << endl;
	for (map<string, Document*>::iterator it = corpus->topic_document.begin();
			it != corpus->topic_document.end(); ++it) {
		Document *doc = it->second;

		vector<string> goldMentions;
		//cout << it->first << " has " << doc->gold_entities.size() << " golds and " << doc->predict_entities.size() << " predictions" << endl;
		for (map<int, Entity *>::iterator iter = doc->gold_entities.begin(); iter != doc->gold_entities.end(); ++iter) {
			int coref_id = iter->first;
			Entity *en = iter->second;
			//cout << "entity: " << en << endl;
			for (int i = 0; i < en->Size(); ++i) {
				Mention *m = en->GetCorefMention(i);
				//cout << m->DocID() << "," << m->SentenceID() << "," << m->StartOffset() << "," << m->EndOffset() << "," << m->GetHeadLemma() << ";";
				string key = std::to_string(m->SentenceID()) + "," + std::to_string(m->StartOffset()) + "," + std::to_string(m->EndOffset());
				goldMentions.push_back(key);
			}
			//cout << endl;
		}

		vector<string> predMentions;
		for (map<int, Entity *>::iterator iter = doc->predict_entities.begin(); iter != doc->predict_entities.end(); ++iter) {
			int coref_id = iter->first;
			Entity *en = iter->second;
			//cout << "entity: " << en << endl;
			for (int i = 0; i < en->Size(); ++i) {
				Mention *m = en->GetCorefMention(i);

				string key = std::to_string(m->SentenceID()) + "," + std::to_string(m->StartOffset()) + "," + std::to_string(m->EndOffset());
				predMentions.push_back(key);

				cout << m->DocID() << "," << m->SentenceID() << "," << m->StartOffset() << "," << m->EndOffset() << "," << m->GetHeadLemma() << ";";

				/*
				if (std::find(goldMentions.begin(), goldMentions.end(), key) == goldMentions.end()) {
					cout << "WE DIDNT FIND: " << key << endl;
				}
				*/
				// cout << "\t" << m->GetHeadLemma() << endl;
				// cout << "\t" << m->MentionID() << " (" << m->StartOffset() << " to " << m->EndOffset() << "): " << m->ToString() << endl;
			}
			cout << endl;
		}
		//cout << "size of golds: " << goldMentions.size() << "; size of preds: " << predMentions.size() << endl;
		MucS->calculateScore(doc->gold_entities, doc->predict_entities);
		BcubeS->calculateScore(doc->gold_entities, doc->predict_entities);
		PairS->calculateScore(doc->gold_entities, doc->predict_entities);
		mentionS->calculateScore(doc->gold_entities, doc->predict_entities);
	}
}

void CorefEval::outputTopicScore(string outputfile)
{
	ofstream outfile(outputfile.c_str(), ios::app);

	for (map<string, vector<Document*> >::iterator it = corpus->topic_to_documents.begin();
			it != corpus->topic_to_documents.end(); ++it) {
		// within document
		MucS->clear();
		BcubeS->clear();
		PairS->clear();
		mentionS->clear();

		for (int i = 0; i < it->second.size(); ++i) {
			Document *doc = it->second[i];
			MucS->calculateScore(doc->gold_entities, doc->predict_entities);
			BcubeS->calculateScore(doc->gold_entities, doc->predict_entities);
			PairS->calculateScore(doc->gold_entities, doc->predict_entities);
			mentionS->calculateScore(doc->gold_entities, doc->predict_entities);
		}

		outfile<<"=== Topic "<<it->first<<endl;
		outfile<<"=== WD score"<<endl;
		outfile<<"MUC: "<<MucS->printF1()<<endl;
		outfile<<"Bcube: "<<BcubeS->printF1()<<endl;
		outfile<<"PairWise: "<<PairS->printF1()<<endl;
		outfile<<"Mention: "<<mentionS->printF1()<<endl;
		outfile<<endl;

		MucS->clear();
		BcubeS->clear();
		PairS->clear();
		mentionS->clear();
		Document *doc = corpus->topic_document[it->first];
		MucS->calculateScore(doc->gold_entities, doc->predict_entities);
		BcubeS->calculateScore(doc->gold_entities, doc->predict_entities);
		PairS->calculateScore(doc->gold_entities, doc->predict_entities);
		mentionS->calculateScore(doc->gold_entities, doc->predict_entities);

		outfile<<"=== CD score"<<endl;
		outfile<<"MUC: "<<MucS->printF1()<<endl;
		outfile<<"Bcube: "<<BcubeS->printF1()<<endl;
		outfile<<"PairWise: "<<PairS->printF1()<<endl;
		outfile<<"Mention: "<<mentionS->printF1()<<endl;
		outfile<<endl;

		outfile<<endl;
	}
	outfile.close();
}

void CorefEval::doWDScore()
{
	MucS->clear();
	BcubeS->clear();
	PairS->clear();
	mentionS->clear();
	// cout << "within doc: " << endl;
	int goldCount = 0;
	int predCount = 0;
	for (map<string, Document*>::iterator it = corpus->documents.begin(); it != corpus->documents.end(); ++it) {
		Document *doc = it->second;
		//cout << it->first << " has " << doc->gold_entities.size() << " golds and " << doc->predict_entities.size() << " predictions" << endl;
		//cout << doc->DocID() << endl;
		vector<string> goldMentions;
		// goldMentions.push_back("test");
		// goldMentions.push_back("test2");
		//cout << it->first << " has " << doc->gold_entities.size() << " golds and " << doc->predict_entities.size() << " predictions" << endl;
		// cout << "GOLDS:" << endl;
		for (map<int, Entity *>::iterator iter = doc->gold_entities.begin(); iter != doc->gold_entities.end(); ++iter) {
			int coref_id = iter->first;
			Entity *en = iter->second;
			// cout << "entity: " << en << endl;
			for (int i = 0; i < en->Size(); ++i) {
				Mention *m = en->GetCorefMention(i);
				string key = std::to_string(m->SentenceID()) + "," + std::to_string(m->StartOffset()) + "," + std::to_string(m->EndOffset());
				goldMentions.push_back(key);
				// cout << m->DocID() << "," << m->SentenceID() << "," << m->StartOffset() << "," << m->EndOffset() << endl;
				// cout << "\t" << m->GetHeadLemma() << endl;
				// cout << "\t" << m->DocID() << ": " << m->MentionID() << " (" << m->StartOffset() << " to " << m->EndOffset() << "): " << m->ToString() << endl;
			}
			// cout << endl;
		}
		// cout << "# of gold mentions: " << goldMentions.size() << endl;

		// cout << "PREDICTIONS:" << endl;
		vector<string> predMentions;
		for (map<int, Entity *>::iterator iter = doc->predict_entities.begin(); iter != doc->predict_entities.end(); ++iter) {
			int coref_id = iter->first;
			Entity *en = iter->second;
			// cout << "entity: " << en << endl;
			for (int i = 0; i < en->Size(); ++i) {
				Mention *m = en->GetCorefMention(i);
				// cout << "\t" << m->GetHeadLemma() << endl;
				string key = std::to_string(m->SentenceID()) + "," + std::to_string(m->StartOffset()) + "," + std::to_string(m->EndOffset());
				//cout << m->DocID() << "," << m->SentenceID() << "," << m->StartOffset() << "," << m->EndOffset() << "," << m->GetHeadLemma() << ";";
				predMentions.push_back(key);
				/*
				if (std::find(goldMentions.begin(), goldMentions.end(), key) == goldMentions.end()) {
					cout << "WE DIDNT FIND: " << key << endl;
				}
				*/
				// cout << "\t" << m->DocID() << ": " << m->MentionID() << " (" << m->StartOffset() << " to " << m->EndOffset() << "): " << m->ToString() << endl;
			}
			//cout << endl;
		}
		goldCount = goldCount + goldMentions.size();
		predCount = predCount + predMentions.size();



		//cout << "size of golds: " << goldMentions.size() << "; size of preds: " << predMentions.size() << endl;
		MucS->calculateScore(doc->gold_entities, doc->predict_entities);
		BcubeS->calculateScore(doc->gold_entities, doc->predict_entities);
		PairS->calculateScore(doc->gold_entities, doc->predict_entities);
		mentionS->calculateScore(doc->gold_entities, doc->predict_entities);
	}
	cout << "# golds: " << goldCount << endl;
	cout << "# preds: " << predCount << endl;
}

void CorefEval::ClusterInfo(map<int, Entity*> &entities) {
	int n_singleton = 0;
	for (int i = 0; i < entities.size(); ++i) {
		if (entities[i]->Size() <= 1) {
			n_singleton++;
		}
	}
	cout<<"Singleton : "<<n_singleton<<" ("<<entities.size()<<")"<<endl;
}

void CorefEval::ErrorAnalysis(Logger &log){
	std::stringstream ss;
	ss<<"============= Error analysis =============="<<endl;
	for (map<string, Document*>::iterator it = corpus->topic_document.begin();
			it != corpus->topic_document.end(); ++it) {
		ss<<"Topic "<<it->first<<endl;

		Document *doc = it->second;
		ss<<"Number of clusters: gold "<<doc->gold_entities.size()
				<<" predict "<<doc->predict_entities.size()<<endl;

		map<int, int> gold_id_to_cluster;
		for (map<int, Entity*>::iterator enit = doc->gold_entities.begin();
				enit != doc->gold_entities.end(); ++enit) {
			for (int i = 0; i < enit->second->Size(); ++i) {
				gold_id_to_cluster[enit->second->GetCorefMention(i)->MentionID()] =
					enit->first;
			}
		}

		for (map<int, Entity*>::iterator enit = doc->predict_entities.begin();
				enit != doc->predict_entities.end(); ++enit) {
			for (int i = 0; i < enit->second->Size(); ++i) {
				for (int j = 0; j < i; ++j) {
					Mention *mi = enit->second->GetCorefMention(i);
					Mention *mj = enit->second->GetCorefMention(j);
					if (gold_id_to_cluster.find(mi->MentionID()) != gold_id_to_cluster.end()
							&& gold_id_to_cluster.find(mj->MentionID()) != gold_id_to_cluster.end()
							&& gold_id_to_cluster[mi->MentionID()] != gold_id_to_cluster[mj->MentionID()]) {
						ss<<"Predict entity size("<<enit->second->Size()<<") : "<<enit->second->ToString()<<endl;
						ss<<"Gold entity 1 size("
								<<doc->gold_entities[gold_id_to_cluster[mi->MentionID()]]->Size()<<") : "
								<<doc->gold_entities[gold_id_to_cluster[mi->MentionID()]]->ToString()
								<<endl;
						ss<<"Gold entity 2 size("
								<<doc->gold_entities[gold_id_to_cluster[mj->MentionID()]]->Size()<<") : "
								<<doc->gold_entities[gold_id_to_cluster[mj->MentionID()]]->ToString()
								<<endl;
						ss<<endl;
					}
				}
			}
		}

/*	    map<int, int> predict_id_to_cluster;
		for (map<int, Entity*>::iterator enit = doc->predict_entities.begin();
				enit != doc->predict_entities.end(); ++enit) {
			for (int i = 0; i < enit->second->Size(); ++i) {
				predict_id_to_cluster[enit->second->GetCorefMention(i)->MentionID()] =
					enit->first;
			}
		}
        for (map<int, Entity*>::iterator enit = doc->gold_entities.begin();
				enit != doc->gold_entities.end(); ++enit) {
			for (int i = 0; i < enit->second->Size(); ++i) {
				for (int j = 0; j < i; ++j) {
					int mi = enit->second->GetCorefMention(i)->MentionID();
					int mj = enit->second->GetCorefMention(j)->MentionID();
					if (predict_id_to_cluster[mi] != predict_id_to_cluster[mj]) {
						ss<<"Gold entity size("<<enit->second->Size()<<") : "<<enit->second->ToString()<<endl;
						ss<<"Predict entity 1 size("
								<<doc->predict_entities[predict_id_to_cluster[mi]]->Size()<<") : "
								<<doc->predict_entities[predict_id_to_cluster[mi]]->ToString()<<endl;
						ss<<"Predict entity 2 size("
								<<doc->predict_entities[predict_id_to_cluster[mj]]->Size()<<") : "
								<<doc->predict_entities[predict_id_to_cluster[mj]]->ToString()<<endl;
						ss<<endl;
					}
				}
			}
		}
		*/
		ss<<endl;
	}
	log.Write(ss.str());
}

void CorefEval::printAccumulateScore()
{
	cout<<"MUC: "<<MucS->printF1()<<endl;
	cout<<"Bcube: "<<BcubeS->printF1()<<endl;
	cout<<"PairWise: "<<PairS->printF1()<<endl;
	cout<<"Mention: "<<mentionS->printF1()<<endl;
}

void CorefEval::printAccumulateScore(Logger &log)
{
	cout<<"MUC: "<<MucS->printF1()<<endl;
	cout<<"Bcube: "<<BcubeS->printF1()<<endl;
	cout<<"PairWise: "<<PairS->printF1()<<endl;
	cout<<"Mention: "<<mentionS->printF1()<<endl;

	log.Write("MUC: "+MucS->printF1()+"\n");
	log.Write("Bcube: "+BcubeS->printF1()+"\n");
	log.Write("PairWise: "+PairS->printF1()+"\n");
	log.Write("Mention: "+mentionS->printF1()+"\n");
}


