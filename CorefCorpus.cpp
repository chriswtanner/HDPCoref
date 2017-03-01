/*
 * Corpus.cpp
 *
 *  Created on: Sep 18, 2013
 *      Author: bishan
 */

#include "CorefCorpus.h"
#include "./Clustering/PairwiseClustering.h"
#include <list>
#include <sstream>
#include <assert.h>
#include <algorithm>
#include "PairwiseModel.h"

#include "DDCRF/ddcrp_utils.h"

CorefCorpus::~CorefCorpus() {
	// TODO Auto-generated destructor stub
}

void CorefCorpus::ReadDocumentAnnotation(string filename, EntityType type) {
	ifstream infile(filename.c_str(), ios::in);

	while (true) {
		string line;
		while (getline(infile, line) && line.find("#begin document") == string::npos);
		if (line == "") break;

		int start_index = line.find('(');
		int end_index = line.find(')');
		string doc_id = line.substr(start_index+1, end_index-start_index-1);

		if (documents.find(doc_id) == documents.end()) continue;

		Document *doc = documents[doc_id];

		// Read sentences.
		int sent_id = 0;
		vector<string> coref_tags;
		while (getline(infile, line) && line.find("#end document") == string::npos) {
			if (line == "") {
				//doc->ReadAnnotation(coref_tags, sent_id);
				doc->BuildCorefClusters(coref_tags, doc->sentences[sent_id], type);
				sent_id ++;
				coref_tags.clear();
			} else {
				vector<string> fields;
				Utils::Split(line, '\t', fields);
				coref_tags.push_back(fields[fields.size()-1]);
			}
		}
	}

	infile.close();
}

void CorefCorpus::LoadEntityClusters(vector<string> filenames, EntityType type) {
	for (int i = 0; i < filenames.size(); ++i) {
		ReadDocumentAnnotation(filenames[i], type);
	}
}

void CorefCorpus::LoadCoNLLData(vector<string> filenames) {
  // Each file presents a topic.
  for (int i = 0; i < filenames.size(); ++i) {
    
    // Reading training documents.
    CoNLLDocumentReader read(filenames[i]);
    
    cout<<"read doc "<<filenames[i]<<endl;
    Document *doc = NULL;
    while ( (doc = read.ReadDocument()) != NULL) {
      //if (train_test_split && train_topics.find(doc->TopicID()) == train_topics.end())
      //	continue;
      
      //if (doc->gold_entities.size() <= 0) {
      //cout<<"document "<<doc->DocID()<<" doesn't have coreference annotation!"<<endl;
      //continue;
      //}
      documents[doc->DocID()] = doc;
      
      // build topic_to_documents and topic_document
      string topic = doc->TopicID();
      if (topic_to_documents.find(topic) == topic_to_documents.end()) {
	vector<Document *> docs;
	topic_to_documents[topic] = docs;
	
	Document *topic_doc = new Document();
	topic_doc->doc_id = topic;
	topic_document[topic] = topic_doc;
      }
      topic_to_documents[topic].push_back(doc);
      for (int j = 0; j < doc->SentNum(); ++j) {
	Sentence *sent = doc->GetSentence(j);
	topic_document[topic]->AddSentence(sent);
      }
    }
  }
}

void CorefCorpus::LoadEvalData(string filename) {
	// Reading training documents.
	CoNLLDocumentReader read(filename);

	Document *doc = NULL;
	while ( (doc = read.ReadEvalDocument()) != NULL) {
		if (doc->gold_entities.size() <= 0) {
			cout<<"document "<<doc->DocID()<<" doesn't have coreference annotation!"<<endl;
			//continue;
		}
		documents[doc->DocID()] = doc;

		// build topic_to_documents and topic_document
		string topic = doc->TopicID();
		if (topic_to_documents.find(topic) == topic_to_documents.end()) {
			vector<Document *> docs;
			topic_to_documents[topic] = docs;

			Document *topic_doc = new Document();
			topic_doc->doc_id = topic;
			topic_document[topic] = topic_doc;
		}
		topic_to_documents[topic].push_back(doc);
		for (int j = 0; j < doc->SentNum(); ++j) {
			Sentence *sent = doc->GetSentence(j);
			topic_document[topic]->AddSentence(sent);
		}
	}
}

void CorefCorpus::OutputSentences(string outfilename) {
	ofstream outfile(outfilename.c_str(), ios::out);

	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		for (int i = 0; i < it->second->SentNum(); ++i) {
			outfile<<"("<<it->first<<"); part 0\t"<<i<<"\t"<<it->second->GetSentence(i)->toString()<<endl;
		}
	}

	outfile.close();
}

void CorefCorpus::OutputCoNLLData(string output_path, bool gold) {
	int k;
	for (map<string, Document*>::iterator it = topic_document.begin();
			it != topic_document.end(); ++it) {
		string topic = it->first;
		string filename = output_path + "/" + topic + "._auto_conll";
		ofstream outfile (filename.c_str(), ios::out);
		outfile<<"#begin document ("<<topic<<"); part 0"<<endl;
		Document *doc = it->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			if (doc->GetSentence(i)->gold_mentions.size() == 0) continue;

			string sent_str = doc->GetSentence(i)->conll_str;
			vector<string> lines;
			Utils::Split(sent_str, '\n', lines);
			vector<string> coref_tags;
			if (gold) {
				doc->GetSentence(i)->GetCoNLLTags(coref_tags);
			} else {
				doc->GetSentence(i)->GetCoNLLPredictMentions(coref_tags);
			}

			for (int j = 0; j < lines.size(); ++j) {
				vector<string> fields;
				Utils::Split(lines[j], '\t', fields);
				outfile<<topic<<"\t";
				for (k = 1; k < fields.size()-1; ++k) {
					outfile<<fields[k]<<"\t";
				}
				outfile<<coref_tags[j]<<endl;
			}
			outfile<<endl;
		}
		outfile<<"#end document"<<endl;
		outfile.close();
	}
}

void CorefCorpus::LoadPhrases(string indexfilename, string phrasefilename) {
	ifstream indexfile(indexfilename.c_str(), ios::in);
	ifstream phrasefile(phrasefilename.c_str(), ios::in);
	vector<string> indices;
	string str;
	while (getline(indexfile, str)) {
		indices.push_back(str);
	}
	indexfile.close();

	int id = 0;
	while (getline(phrasefile, str)) {
		string index = indices[id++];
		int i = index.find('\t');
		string docstr = index.substr(0,i);
		int sentid = atoi(index.substr(i+1).c_str());
		i = docstr.find(')');
		docstr = docstr.substr(1, i-1);

		if (documents.find(docstr) == documents.end())
			continue;

		Document *doc = documents[docstr];
		Sentence *sent = doc->GetSentence(sentid);

		vector<string> fields;
		Utils::Split(str, ' ', fields);
		i = 0;
		for (int j = 0; j < fields.size(); ++j) {
			string word = fields[j];
			if (i == sent->TokenSize()) {
				cout<<doc->doc_id<<" "<<sent->toString()<<endl;
			}
			if (word == sent->GetSpanLowerCase(i, i)) {
				i++;
			} else {
				int start = i++;
				vector<string> splits;
				Utils::Split(fields[j], '_', splits);
				while (i < sent->TokenSize()) {
					string str = sent->GetSpanLowerCase(start, i);
					replace(str.begin(), str.end(), ' ', '_');
					// connected by '_'
					if (str == word) {
						// find a span!!!
						Span s(start, i, str);
						sent->phrases.push_back(s);
						i++;
						break;
					}
					i++;
				}
			}
		}
	}
	phrasefile.close();
}

void CorefCorpus::LoadParsingInfo(string sentidfilename, string depparsename) {
	ifstream sentidfile(sentidfilename.c_str(), ios::in);
	ifstream parsefile(depparsename.c_str(), ios::in);
	vector<string> dependencies;
	vector<string> parsetrees;
	string str;
	while (getline(parsefile, str)) {
		// read parse tree first
		string tree_str = "";
		while (str != "") {
			Utils::Trim(str);
			tree_str += str;
			getline(parsefile, str);
		}
		parsetrees.push_back(tree_str);

		string depstr = "";
		getline(parsefile, str);
		while (str != "") {
			depstr += str + "\n";
			getline(parsefile, str);
		}
		dependencies.push_back(depstr);
	}
	parsefile.close();

	// Sanity check.
	int sent_num = 0;
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		sent_num += it->second->SentNum();
	}
	cout<<"Load "<<sent_num<<" "<<dependencies.size()<<" "<<parsetrees.size()<<" number of sents/parses/deps"<<endl;

	int c = 0;
	while (getline(sentidfile, str)) {
		vector<string> fields;
		Utils::Split(str, ' ', fields);
		string doc_id = fields[2];
		int sent_id = atoi(fields[1].c_str());
		string dep_str = dependencies[c];
		string tree_str = parsetrees[c];
		if (documents.find(doc_id) == documents.end()) continue;

		Document *doc = documents[doc_id];
		doc->GetSentence(sent_id)->buildDepGraph(dep_str);
		doc->GetSentence(sent_id)->buildPennTree(tree_str);

		c++;
	}
	sentidfile.close();
}

void CorefCorpus::OutputEntities(string filename) {
	ofstream outfile(filename.c_str(), ios::out);

	set<string> locations;
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
//		for (int i = 0; i < doc->gold_time_mentions.size(); ++i) {
//			if (!Dictionary::containTemporal(doc->gold_time_mentions[i]->mention_str)) {
//				outfile<<doc->gold_time_mentions[i]->mention_str<<endl;
//			}
//		}
		for (int i = 0; i < doc->gold_loc_mentions.size(); ++i) {
			vector<string> words;
			Utils::Split(doc->gold_loc_mentions[i]->mention_str, ' ', words);
			if (words.size() == 0) {
				continue;
			}

//			if (words[0] == "in" || words[0] == "at" || words[0] == "on") {
//				locations.insert(Utils::toLower(words[1]));
//			} else {
//				locations.insert(Utils::toLower(words[0]));
//			}
			int head = -1;
			if (!Dictionary::containLocation(doc->gold_loc_mentions[i]->mention_str, head)) {
				outfile<<doc->gold_loc_mentions[i]->mention_str<<endl;
			}
		}
	}

//	ofstream dictfile("locations", ios::out);
//	for (set<string>::iterator it = locations.begin(); it != locations.end(); ++it) {
//		dictfile<<*it<<endl;
//	}
//	dictfile.close();

	outfile.close();
}

void CorefCorpus::RelaxedMatchMentions() {
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);
			// Map the index of a headword to to a list of mentions.
			map<int, list<Mention*> > head_to_mentions;
			// Map a span to a mention.
			map<pair<int, int>, Mention*> span_to_mention;

			for (int j = 0; j < sent->gold_mentions.size(); ++j) {
				Mention *mention = sent->gold_mentions[j];
				span_to_mention[make_pair(mention->StartOffset(), mention->EndOffset())] = mention;
				int head_index = mention->head_index;
				if (head_to_mentions.find(head_index) == head_to_mentions.end()) {
					list<Mention*> mentions;
					head_to_mentions[head_index] = mentions;
				}
				head_to_mentions[head_index].push_back(mention);
			}

			vector<Mention *> remains;
			// Exact match.
			for (int j = 0; j < sent->predict_mentions.size(); ++j) {
				Mention *m = sent->predict_mentions[j];
				if (span_to_mention.find(make_pair(m->StartOffset(), m->EndOffset())) !=
						span_to_mention.end() &&
						span_to_mention[make_pair(m->StartOffset(), m->EndOffset())]->Equal(m)) {
					Mention *gold_m = span_to_mention[make_pair(m->StartOffset(), m->EndOffset())];
					gold_m->twinless = false;
					m->twinless = false;
					m->SetMentionID(gold_m->mention_id);
					m->gold_entity_id = gold_m->gold_entity_id;

					// Find the mention in the head_to_mention list and delete it.
					for (list<Mention*>::iterator it = head_to_mentions[m->head_index].begin();
							it != head_to_mentions[m->head_index].end(); ++it) {
						Mention *gm = *it;
						if (gm->Equal(m)) {
							head_to_mentions[m->head_index].erase(it);
							break;
						}
					}
				}
				else remains.push_back(m);
			}
			// Relaxed match.
			for(int j = 0; j < remains.size(); ++j){
				Mention *m = remains[j];
				if (head_to_mentions.find(m->head_index) != head_to_mentions.end()) {
					list<Mention*> mentions = head_to_mentions[m->head_index];
					if (mentions.size() > 0) {
						Mention *gm = mentions.front();
						gm->twinless = false;
						m->twinless = false;
						m->mention_id = gm->mention_id;
						m->gold_entity_id = gm->gold_entity_id;
						head_to_mentions[m->head_index].pop_front();
					}
				}
			}
			// Boundary match (for cases where the head is wrong).
			for(int j = 0; j < remains.size(); ++j){
				if (!remains[j]->twinless) continue;
				Mention *pm = remains[j];
				for (int j = 0; j < sent->gold_mentions.size(); ++j) {
					Mention *gm = sent->gold_mentions[j];
					if (!gm->twinless) continue;
					if (gm->Contain(pm) || pm->Contain(gm)) {
						gm->twinless = false;
						pm->twinless = false;
						pm->mention_id = gm->mention_id;
						pm->gold_entity_id = gm->gold_entity_id;
						break;
					}
				}
			}
		}
	}
}

void CorefCorpus::StrictMatchMentions(){
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);
			map<pair<int, int>, Mention*> span_to_mention;
			for (int j = 0; j < sent->gold_mentions.size(); ++j) {
				Mention *mention = sent->gold_mentions[j];
				span_to_mention[make_pair(mention->StartOffset(), mention->EndOffset())] = mention;
			}

			for (int j = 0; j < sent->predict_mentions.size(); ++j) {
				Mention *m = sent->predict_mentions[j];
				if (span_to_mention.find(make_pair(m->StartOffset(), m->EndOffset())) !=
						span_to_mention.end()) {
					Mention *gold_m = span_to_mention[make_pair(m->StartOffset(), m->EndOffset())];
					if (gold_m->Equal(m)) {
						gold_m->twinless = false;
						m->twinless = false;
						// only change mid!!!
						m->SetMentionID(gold_m->MentionID());
						m->gold_entity_id = gold_m->gold_entity_id;
					}
				}
			}
		}
	}
}

void CorefCorpus::MatchMentions(bool exact) {
	if (exact) StrictMatchMentions();
	else RelaxedMatchMentions();
}

void CorefCorpus::EvaluateMentionExtraction() {
	Logger logger;
	int correct_gn = 0, correct_pn = 0;
	int all_gn = 0, all_pn = 0;
	std::stringstream ss;
	ss<<"Missing gold mentions: \n";
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->gold_mentions.size(); ++i) {
			Mention *m = doc->gold_mentions[i];

			if(!m->twinless) {
				correct_gn++;
			}
			else {
				Sentence *sent = doc->GetSentence(m->SentenceID());
				ss<<m->doc_id<<"\t"<<m->sent_id<<"\t"<<m->mention_str<<"\t"<<sent->GetMentionContext(m)<<"\t";
				for (int j = 0; j < sent->predict_mentions.size(); ++j) {
					ss<<sent->predict_mentions[j]->mention_str<<",";
				}
				ss<<"\t";
				for (int j = 0; j < sent->gold_mentions.size(); ++j) {
					ss<<sent->gold_mentions[j]->mention_str<<",";
				}
				ss<<"\n";
			}

			all_gn ++;
		}

		for (int i = 0; i < doc->predict_mentions.size(); ++i) {
			if (!doc->predict_mentions[i]->twinless) correct_pn++;
			all_pn ++;
		}
	}

	cout<<"Mention Extraction Precision : " << (double) correct_pn/all_pn << "(" << correct_pn << " out of "<<all_pn<<")"
			<<" Recall : "<< (double)correct_gn/all_gn << "(" << correct_gn <<" out of "<<all_gn<<")"<<endl;

	ss << "Mention Extraction Precision : " << (double) correct_pn/all_pn << "(" << correct_pn << " out of "<<all_pn<<")"
			<<" Recall : "<< (double)correct_gn/all_gn << "(" << correct_gn <<" out of "<<all_gn<<")"<<endl;

	logger.Write(ss.str());
}

void CorefCorpus::RelaxedMatchParticipantMentions() {
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);
			// Map the index of a headword to to a list of mentions.
			map<int, list<Mention*> > head_to_mentions;
			// Map a span to a mention.
			map<pair<int, int>, Mention*> span_to_mention;

			for (int j = 0; j < sent->gold_participant_mentions.size(); ++j) {
				Mention *mention = sent->gold_participant_mentions[j];
				span_to_mention[make_pair(mention->StartOffset(), mention->EndOffset())] = mention;
				int head_index = mention->head_index;
				if (head_to_mentions.find(head_index) == head_to_mentions.end()) {
					list<Mention*> mentions;
					head_to_mentions[head_index] = mentions;
				}
				head_to_mentions[head_index].push_back(mention);
			}

			vector<Mention *> remains;
			// Exact match.
			for (int j = 0; j < sent->predict_participant_mentions.size(); ++j) {
				Mention *m = sent->predict_participant_mentions[j];
				if (span_to_mention.find(make_pair(m->StartOffset(), m->EndOffset())) !=
						span_to_mention.end() &&
						span_to_mention[make_pair(m->StartOffset(), m->EndOffset())]->Equal(m)) {
					Mention *gold_m = span_to_mention[make_pair(m->StartOffset(), m->EndOffset())];
					gold_m->twinless = false;
					m->twinless = false;
					m->SetMentionID(gold_m->mention_id);
					m->gold_entity_id = gold_m->gold_entity_id;

					// Find the mention in the head_to_mention list and delete it.
					for (list<Mention*>::iterator it = head_to_mentions[m->head_index].begin();
							it != head_to_mentions[m->head_index].end(); ++it) {
						Mention *gm = *it;
						if (gm->Equal(m)) {
							head_to_mentions[m->head_index].erase(it);
							break;
						}
					}
				}
				else remains.push_back(m);
			}
			// Relaxed match.
			for(int j = 0; j < remains.size(); ++j){
				Mention *m = remains[j];
				if (head_to_mentions.find(m->head_index) != head_to_mentions.end()) {
					list<Mention*> mentions = head_to_mentions[m->head_index];
					if (mentions.size() > 0) {
						Mention *gm = mentions.front();
						gm->twinless = false;
						m->twinless = false;
						m->mention_id = gm->mention_id;
						m->gold_entity_id = gm->gold_entity_id;
						head_to_mentions[m->head_index].pop_front();
					}
				}
			}
			// Boundary match (for cases where the head is wrong).
			for(int j = 0; j < remains.size(); ++j){
				if (!remains[j]->twinless) continue;
				Mention *pm = remains[j];
				for (int j = 0; j < sent->gold_participant_mentions.size(); ++j) {
					Mention *gm = sent->gold_participant_mentions[j];
					if (!gm->twinless) continue;
					if (gm->Contain(pm) || pm->Contain(gm)) {
						gm->twinless = false;
						pm->twinless = false;
						pm->mention_id = gm->mention_id;
						pm->gold_entity_id = gm->gold_entity_id;
						break;
					}
				}
			}
		}
	}
}

void CorefCorpus::StrictMatchParticipantMentions(){
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);
			map<pair<int, int>, Mention*> span_to_mention;
			for (int j = 0; j < sent->gold_participant_mentions.size(); ++j) {
				Mention *mention = sent->gold_participant_mentions[j];
				span_to_mention[make_pair(mention->StartOffset(), mention->EndOffset())] = mention;
			}

			for (int j = 0; j < sent->predict_participant_mentions.size(); ++j) {
				Mention *m = sent->predict_participant_mentions[j];
				if (span_to_mention.find(make_pair(m->StartOffset(), m->EndOffset())) !=
						span_to_mention.end()) {
					Mention *gold_m = span_to_mention[make_pair(m->StartOffset(), m->EndOffset())];
					if (gold_m->Equal(m)) {
						gold_m->twinless = false;
						m->twinless = false;
						// only change mid!!!
						m->SetMentionID(gold_m->MentionID());
						m->gold_entity_id = gold_m->gold_entity_id;
					}
				}
			}
		}
	}
}

void CorefCorpus::MatchParticipantMentions(bool exact) {
	if (exact) StrictMatchParticipantMentions();
	else RelaxedMatchParticipantMentions();
}

void CorefCorpus::EvaluateParticipantMentionExtraction() {
	Logger logger;
	int correct_gn = 0, correct_pn = 0;
	int all_gn = 0, all_pn = 0;
	std::stringstream ss;
	ss<<"Missing gold mentions: \n";
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->gold_participant_mentions.size(); ++i) {
			Mention *m = doc->gold_participant_mentions[i];

			if(!m->twinless) {
				correct_gn++;
			}
			else {
				Sentence *sent = doc->GetSentence(m->SentenceID());
				ss<<m->doc_id<<"\t"<<m->sent_id<<"\t"<<m->mention_str<<"\t"<<sent->GetMentionContext(m)<<"\t";
				for (int j = 0; j < sent->predict_participant_mentions.size(); ++j) {
					ss<<sent->predict_participant_mentions[j]->mention_str<<",";
				}
				ss<<"\t";
				for (int j = 0; j < sent->gold_participant_mentions.size(); ++j) {
					ss<<sent->gold_participant_mentions[j]->mention_str<<",";
				}
				ss<<"\n";
			}

			all_gn ++;
		}

		for (int i = 0; i < doc->predict_participant_mentions.size(); ++i) {
			if (!doc->predict_participant_mentions[i]->twinless) correct_pn++;
			all_pn ++;
		}
	}

	cout<<"Mention Extraction Precision : " << (double) correct_pn/all_pn << "(" << correct_pn << " out of "<<all_pn<<")"
			<<" Recall : "<< (double)correct_gn/all_gn << "(" << correct_gn <<" out of "<<all_gn<<")"<<endl;

	ss << "Mention Extraction Precision : " << (double) correct_pn/all_pn << "(" << correct_pn << " out of "<<all_pn<<")"
			<<" Recall : "<< (double)correct_gn/all_gn << "(" << correct_gn <<" out of "<<all_gn<<")"<<endl;

	logger.Write(ss.str());
}

void CorefCorpus::RelaxedMatchTimeMentions() {
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);
			// Map the index of a headword to to a list of mentions.
			map<int, list<Mention*> > head_to_mentions;
			// Map a span to a mention.
			map<pair<int, int>, Mention*> span_to_mention;

			for (int j = 0; j < sent->gold_time_mentions.size(); ++j) {
				Mention *mention = sent->gold_time_mentions[j];
				span_to_mention[make_pair(mention->StartOffset(), mention->EndOffset())] = mention;
				int head_index = mention->head_index;
				if (head_to_mentions.find(head_index) == head_to_mentions.end()) {
					list<Mention*> mentions;
					head_to_mentions[head_index] = mentions;
				}
				head_to_mentions[head_index].push_back(mention);
			}

			vector<Mention *> remains;
			// Exact match.
			for (int j = 0; j < sent->predict_time_mentions.size(); ++j) {
				Mention *m = sent->predict_time_mentions[j];
				if (span_to_mention.find(make_pair(m->StartOffset(), m->EndOffset())) !=
						span_to_mention.end() &&
						span_to_mention[make_pair(m->StartOffset(), m->EndOffset())]->Equal(m)) {
					Mention *gold_m = span_to_mention[make_pair(m->StartOffset(), m->EndOffset())];
					gold_m->twinless = false;
					m->twinless = false;
					m->SetMentionID(gold_m->mention_id);
					m->gold_entity_id = gold_m->gold_entity_id;

					// Find the mention in the head_to_mention list and delete it.
					for (list<Mention*>::iterator it = head_to_mentions[m->head_index].begin();
							it != head_to_mentions[m->head_index].end(); ++it) {
						Mention *gm = *it;
						if (gm->Equal(m)) {
							head_to_mentions[m->head_index].erase(it);
							break;
						}
					}
				}
				else remains.push_back(m);
			}
			// Relaxed match.
			for(int j = 0; j < remains.size(); ++j){
				Mention *m = remains[j];
				if (head_to_mentions.find(m->head_index) != head_to_mentions.end()) {
					list<Mention*> mentions = head_to_mentions[m->head_index];
					if (mentions.size() > 0) {
						Mention *gm = mentions.front();
						gm->twinless = false;
						m->twinless = false;
						m->mention_id = gm->mention_id;
						m->gold_entity_id = gm->gold_entity_id;
						head_to_mentions[m->head_index].pop_front();
					}
				}
			}
			// Boundary match (for cases where the head is wrong).
			for(int j = 0; j < remains.size(); ++j){
				if (!remains[j]->twinless) continue;
				Mention *pm = remains[j];
				for (int j = 0; j < sent->gold_time_mentions.size(); ++j) {
					Mention *gm = sent->gold_time_mentions[j];
					if (!gm->twinless) continue;
					if (gm->Contain(pm) || pm->Contain(gm)) {
						gm->twinless = false;
						pm->twinless = false;
						pm->mention_id = gm->mention_id;
						pm->gold_entity_id = gm->gold_entity_id;
						break;
					}
				}
			}
		}
	}
}

void CorefCorpus::StrictMatchTimeMentions(){
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);
			map<pair<int, int>, Mention*> span_to_mention;
			for (int j = 0; j < sent->gold_time_mentions.size(); ++j) {
				Mention *mention = sent->gold_time_mentions[j];
				span_to_mention[make_pair(mention->StartOffset(), mention->EndOffset())] = mention;
			}

			for (int j = 0; j < sent->predict_time_mentions.size(); ++j) {
				Mention *m = sent->predict_time_mentions[j];
				if (span_to_mention.find(make_pair(m->StartOffset(), m->EndOffset())) !=
						span_to_mention.end()) {
					Mention *gold_m = span_to_mention[make_pair(m->StartOffset(), m->EndOffset())];
					if (gold_m->Equal(m)) {
						gold_m->twinless = false;
						m->twinless = false;
						// only change mid!!!
						m->SetMentionID(gold_m->MentionID());
						m->gold_entity_id = gold_m->gold_entity_id;
					}
				}
			}
		}
	}
}

void CorefCorpus::MatchTimeMentions(bool exact) {
	if (exact) StrictMatchTimeMentions();
	else RelaxedMatchTimeMentions();
}

void CorefCorpus::EvaluateTimeMentionExtraction() {
	Logger logger;
	int correct_gn = 0, correct_pn = 0;
	int all_gn = 0, all_pn = 0;
	std::stringstream ss;
	ss<<"Missing gold mentions: \n";
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->gold_time_mentions.size(); ++i) {
			Mention *m = doc->gold_time_mentions[i];

			if(!m->twinless) {
				correct_gn++;
			}
			else {
				Sentence *sent = doc->GetSentence(m->SentenceID());
				ss<<m->doc_id<<"\t"<<m->sent_id<<"\t"<<m->mention_str<<"\t"<<sent->GetMentionContext(m)<<"\t";
				for (int j = 0; j < sent->predict_time_mentions.size(); ++j) {
					ss<<sent->predict_time_mentions[j]->mention_str<<",";
				}
				ss<<"\t";
				for (int j = 0; j < sent->gold_time_mentions.size(); ++j) {
					ss<<sent->gold_time_mentions[j]->mention_str<<",";
				}
				ss<<"\n";
			}

			all_gn ++;
		}

		for (int i = 0; i < doc->predict_time_mentions.size(); ++i) {
			if (!doc->predict_time_mentions[i]->twinless) correct_pn++;
			all_pn ++;
		}
	}

	cout<<"Mention Extraction Precision : " << (double) correct_pn/all_pn << "(" << correct_pn << " out of "<<all_pn<<")"
			<<" Recall : "<< (double)correct_gn/all_gn << "(" << correct_gn <<" out of "<<all_gn<<")"<<endl;

	ss << "Mention Extraction Precision : " << (double) correct_pn/all_pn << "(" << correct_pn << " out of "<<all_pn<<")"
			<<" Recall : "<< (double)correct_gn/all_gn << "(" << correct_gn <<" out of "<<all_gn<<")"<<endl;

	logger.Write(ss.str());
}

void CorefCorpus::RelaxedMatchLocMentions() {
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);
			// Map the index of a headword to to a list of mentions.
			map<int, list<Mention*> > head_to_mentions;
			// Map a span to a mention.
			map<pair<int, int>, Mention*> span_to_mention;

			for (int j = 0; j < sent->gold_loc_mentions.size(); ++j) {
				Mention *mention = sent->gold_loc_mentions[j];
				span_to_mention[make_pair(mention->StartOffset(), mention->EndOffset())] = mention;
				int head_index = mention->head_index;
				if (head_to_mentions.find(head_index) == head_to_mentions.end()) {
					list<Mention*> mentions;
					head_to_mentions[head_index] = mentions;
				}
				head_to_mentions[head_index].push_back(mention);
			}

			vector<Mention *> remains;
			// Exact match.
			for (int j = 0; j < sent->predict_loc_mentions.size(); ++j) {
				Mention *m = sent->predict_loc_mentions[j];
				if (span_to_mention.find(make_pair(m->StartOffset(), m->EndOffset())) !=
						span_to_mention.end() &&
						span_to_mention[make_pair(m->StartOffset(), m->EndOffset())]->Equal(m)) {
					Mention *gold_m = span_to_mention[make_pair(m->StartOffset(), m->EndOffset())];
					gold_m->twinless = false;
					m->twinless = false;
					m->SetMentionID(gold_m->mention_id);
					m->gold_entity_id = gold_m->gold_entity_id;

					// Find the mention in the head_to_mention list and delete it.
					for (list<Mention*>::iterator it = head_to_mentions[m->head_index].begin();
							it != head_to_mentions[m->head_index].end(); ++it) {
						Mention *gm = *it;
						if (gm->Equal(m)) {
							head_to_mentions[m->head_index].erase(it);
							break;
						}
					}
				}
				else remains.push_back(m);
			}
			// Relaxed match.
			for(int j = 0; j < remains.size(); ++j){
				Mention *m = remains[j];
				if (head_to_mentions.find(m->head_index) != head_to_mentions.end()) {
					list<Mention*> mentions = head_to_mentions[m->head_index];
					if (mentions.size() > 0) {
						Mention *gm = mentions.front();
						gm->twinless = false;
						m->twinless = false;
						m->mention_id = gm->mention_id;
						m->gold_entity_id = gm->gold_entity_id;
						head_to_mentions[m->head_index].pop_front();
					}
				}
			}
			// Boundary match (for cases where the head is wrong).
			for(int j = 0; j < remains.size(); ++j){
				if (!remains[j]->twinless) continue;
				Mention *pm = remains[j];
				for (int j = 0; j < sent->gold_loc_mentions.size(); ++j) {
					Mention *gm = sent->gold_loc_mentions[j];
					if (!gm->twinless) continue;
					if (gm->Contain(pm) || pm->Contain(gm)) {
						gm->twinless = false;
						pm->twinless = false;
						pm->mention_id = gm->mention_id;
						pm->gold_entity_id = gm->gold_entity_id;
						break;
					}
				}
			}
		}
	}
}

void CorefCorpus::StrictMatchLocMentions(){
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);
			map<pair<int, int>, Mention*> span_to_mention;
			for (int j = 0; j < sent->gold_loc_mentions.size(); ++j) {
				Mention *mention = sent->gold_loc_mentions[j];
				span_to_mention[make_pair(mention->StartOffset(), mention->EndOffset())] = mention;
			}

			for (int j = 0; j < sent->predict_loc_mentions.size(); ++j) {
				Mention *m = sent->predict_loc_mentions[j];
				if (span_to_mention.find(make_pair(m->StartOffset(), m->EndOffset())) !=
						span_to_mention.end()) {
					Mention *gold_m = span_to_mention[make_pair(m->StartOffset(), m->EndOffset())];
					if (gold_m->Equal(m)) {
						gold_m->twinless = false;
						m->twinless = false;
						// only change mid!!!
						m->SetMentionID(gold_m->MentionID());
						m->gold_entity_id = gold_m->gold_entity_id;
					}
				}
			}
		}
	}
}

void CorefCorpus::MatchLocMentions(bool exact) {
	if (exact) StrictMatchLocMentions();
	else RelaxedMatchLocMentions();
}

void CorefCorpus::EvaluateLocMentionExtraction() {
	Logger logger;
	int correct_gn = 0, correct_pn = 0;
	int all_gn = 0, all_pn = 0;
	std::stringstream ss;
	ss<<"Missing gold mentions: \n";
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->gold_loc_mentions.size(); ++i) {
			Mention *m = doc->gold_loc_mentions[i];

			if(!m->twinless) {
				correct_gn++;
			}
			else {
				Sentence *sent = doc->GetSentence(m->SentenceID());
				ss<<m->doc_id<<"\t"<<m->sent_id<<"\t"<<m->mention_str<<"\t"<<sent->GetMentionContext(m)<<"\t";
				for (int j = 0; j < sent->predict_loc_mentions.size(); ++j) {
					ss<<sent->predict_loc_mentions[j]->mention_str<<",";
				}
				ss<<"\t";
				for (int j = 0; j < sent->gold_loc_mentions.size(); ++j) {
					ss<<sent->gold_loc_mentions[j]->mention_str<<",";
				}
				ss<<"\n";
			}

			all_gn ++;
		}

		for (int i = 0; i < doc->predict_loc_mentions.size(); ++i) {
			if (!doc->predict_loc_mentions[i]->twinless) correct_pn++;
			all_pn ++;
		}
	}

	cout<<"Mention Extraction Precision : " << (double) correct_pn/all_pn << "(" << correct_pn << " out of "<<all_pn<<")"
			<<" Recall : "<< (double)correct_gn/all_gn << "(" << correct_gn <<" out of "<<all_gn<<")"<<endl;

	ss << "Mention Extraction Precision : " << (double) correct_pn/all_pn << "(" << correct_pn << " out of "<<all_pn<<")"
			<<" Recall : "<< (double)correct_gn/all_gn << "(" << correct_gn <<" out of "<<all_gn<<")"<<endl;

	logger.Write(ss.str());
}

void CorefCorpus::LoadMentionFeatures(string filename, bool gold) {
	vector<string> filenames;
	Utils::Split(filename, ',', filenames);
	for (int f = 0; f < filenames.size(); ++f) {
		ifstream infile(filenames[f].c_str(), ios::in);
		string line;
		while(getline(infile, line)) {
			if (line.find("#begin document") != string::npos) {
				int start_index = line.find('(');
				int end_index = line.find(')');
				string doc_title = line.substr(start_index+1, end_index-start_index-1);

				if (documents.find(doc_title) == documents.end()) continue;

				Document *doc = documents[doc_title];
				while (getline(infile, line) && line.find("#end document") == string::npos) {
					vector<string> fields;
					Utils::Split(line, '\t', fields);
					if (fields.size() < 4) continue;

					int sent_num = atoi(fields[0].c_str());
					int start_offset = atoi(fields[1].c_str());
					int end_offset = atoi(fields[2].c_str()) - 1;

					Mention *m = NULL;
					if (gold) {
						m = doc->GetSentence(sent_num)->FetchGoldMention(start_offset, end_offset);
					} else {
						m = doc->GetSentence(sent_num)->FetchPredictMention(start_offset, end_offset);
					}
					if (m != NULL) {
						doc->GetSentence(sent_num)->ParseMentionFeatures(fields, m);
					}
				}
			}
		}
		infile.close();
	}
}

void CorefCorpus::LoadPredictMentions(string filename) {
	
	cout << "load predict mentions file: " << filename << endl;
	ifstream infile(filename.c_str(), ios::in);
	string line;
	int lineno = 0;
	while(getline(infile, line)) {
		vector<string> fields;
		Utils::Split(line, '\t', fields);
		//if (fields.size() < 6) continue;

		string type = fields[4];
		if (type == "HUM" || type == "NON_HUM") type = "PARTICIPANT";

		string docid = fields[0];
		int sent_num = atoi(fields[1].c_str());
		int start_offset = atoi(fields[2].c_str());
		int end_offset = atoi(fields[3].c_str()) - 1;

		if (documents.find(docid) == documents.end()) continue;

		Mention *pm = documents[docid]->GetSentence(sent_num)->FetchPredictMention(start_offset, end_offset);
		if (pm == NULL || pm->anno_type != type) {
			Mention* m = new Mention();
			m->SetDocID(docid);
			m->SetSentenceID(sent_num);
			m->SetStartOffset(start_offset);
			m->SetEndOffset(end_offset);
			m->head_index = start_offset;
			m->head_lemma = documents[docid]->GetSentence(sent_num)->tokens[m->head_index]->lemma;
			m->mention_str = documents[docid]->GetSentence(sent_num)->GetMentionStr(m);
			m->anno_prob = atof(fields[fields.size()-1].c_str());
			m->anno_type = type;

			if (type == "EVENT") {
				documents[docid]->GetSentence(sent_num)->AddEventMention(m);
			} else if (type == "PARTICIPANT") {
				documents[docid]->GetSentence(sent_num)->AddParticipantMention(m);
			} else if (type == "TIME") {
				documents[docid]->GetSentence(sent_num)->AddTimeMention(m);
			} else if (type == "LOC") {
				documents[docid]->GetSentence(sent_num)->AddLocMention(m);
			}

		} else {
			//if (type == "EVENT") lineno++;
		}

	}
	infile.close();
	//cout<<lineno<<endl;
}

void CorefCorpus::SetGoldMentionInfo(string filename) {
	vector<string> filenames;
	Utils::Split(filename, ',', filenames);
	for (int i = 0; i < filenames.size(); ++i) {
		ifstream infile(filenames[i].c_str(), ios::in);

		string line;
		while(getline(infile, line)) {
			if (line.find("#begin document") != string::npos) {
				// For each document.
				//int index = line.find_last_of(' ');
				//string doc_title = line.substr(index+1);
				int start_index = line.find('(');
				int end_index = line.find(')');
				string doc_title = line.substr(start_index+1, end_index-start_index-1);

				if (documents.find(doc_title) == documents.end()) continue;

				Document *doc = documents[doc_title];
				while (getline(infile, line) && line.find("#end document") == string::npos) {
					vector<string> fields;
					Utils::Split(line, '\t', fields);
					if (fields.size() < 4) continue;
					//int sent_num = atoi(fields[0].c_str());
					//int start_offset = atoi(fields[1].c_str());
					//int end_offset = atoi(fields[2].c_str());
					int sent_num = atoi(fields[0].c_str());
					int start_offset = atoi(fields[1].c_str());
					int end_offset = atoi(fields[2].c_str()) - 1;

					Mention *m = doc->GetSentence(sent_num)->FetchGoldMention(start_offset, end_offset);
					if (m != NULL) {
						// Has already processed!!!??
						if (m->MentionID() < 0) {
							doc->GetSentence(sent_num)->ParseMention(fields, m);
						} else {
							cout<<"Repeated gold mentions!!!"<<endl;
						}
					} else {
						cout<<"couldn't find the mention!!!"<<endl;
					}
				}
			}
		}
		infile.close();
	}
}

void CorefCorpus::OutputDDCRPDistance(string path) {
	for (map<string, vector<Document*> >::iterator iter = topic_to_documents.begin();
		iter != topic_to_documents.end(); ++iter) {
		string filename = path + "/" + iter->first + ".dist";
		ofstream outfile(filename.c_str(), ios::out);
		vector<Document*> documents = iter->second;
		for (int d1 = 0; d1 < documents.size(); ++d1) {
			for (int j1 = 0; j1 < documents[d1]->predict_mentions.size(); ++j1) {
				Mention *m1 = documents[d1]->predict_mentions[j1];
				for (int d2 = 0; d2 < d1; ++d2) {
					if (d1 == d2) continue;
					for (int j2 = 0; j2 < documents[d2]->predict_mentions.size(); ++j2) {
						Mention *m2 = documents[d2]->predict_mentions[j2];
						double dist = Constraints::PairwiseDistance(m1, m2);
						if (dist != 0.5) {
							outfile << m1->DocID() << "," << j1 << " "
									<< m2->DocID() << "," << j2 << " " << dist << endl;
						}
					}
				}
			}
		}
		outfile.close();
	}
}

void CorefCorpus::OutputDDCRPCorpus(string path) {
	for (map<string, vector<Document*> >::iterator it = topic_to_documents.begin();
			it != topic_to_documents.end(); ++it) {
		string filename = path + "/" + it->first + ".data";
		ofstream outfile(filename.c_str(), ios::out);
		int nw = 0, nd = 0, total_words = 0;
		map<string, int> wordmap;
		for (int i = 0; i < it->second.size(); ++i) {
			Document *doc = it->second[i];
			int length = doc->predict_mentions.size();
			string doc_id = doc->doc_id;
			outfile<<"#begin document ("<<doc_id<<")"<<endl;
			for (int n = 0; n < length; n++)
			{
				Mention *m = doc->predict_mentions[n];
				//if (m->IsPronoun()) continue;

				// Lemma???
				string headword = m->head_lemma;
				if (wordmap.find(headword) == wordmap.end()) {
					wordmap[headword] = wordmap.size();
				}
				int word_id = wordmap[headword];

				if (word_id >= nw)
				{
					nw = word_id + 1;
				}
				outfile<<word_id<<":"<<headword<<" ";
			}
			outfile<<endl;

			// Output distance info.
			for (int n = 0; n < length; n++) {
				Mention *m = doc->predict_mentions[n];
				for (int j = 0; j < n; ++j) {
					Mention *ant = doc->predict_mentions[j];
					double dist = Constraints::PairwiseDistance(m, ant);
					outfile<<j<<":"<<dist<<" ";
				}
				outfile<<endl;
			}
			outfile<<"#end document"<<endl;

			total_words += length;
			nd++;
		}
		outfile.close();
	}
}

void CorefCorpus::OutputInitialClusters(string path) {
	for (map<string, vector<Document*> >::iterator iter = topic_to_documents.begin();
			iter != topic_to_documents.end(); ++iter) {
		string filename = path + "/" + iter->first + ".clusters";
		ofstream outfile(filename.c_str(), ios::out);
		for (int i = 0; i < iter->second.size(); ++i) {
			Document *doc = iter->second[i];
			int size = doc->predict_mentions.size();
			string doc_id = doc->doc_id;
			outfile<<doc_id<<"\t";
			for (int n = 0; n < size; n++) {
				Mention *m = doc->predict_mentions[n];
				outfile<<n<<":"<<m->pred_entity_id<<" ";
			}
			outfile<<endl;
		}
		outfile.close();
	}
}

void CorefCorpus::AddAllGoldMentions() {
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int s = 0; s < doc->SentNum(); ++s) {
			for (int m = 0; m < doc->GetSentence(s)->gold_mentions.size(); ++m) {
				Mention *men = doc->GetSentence(s)->gold_mentions[m]->Copy();
				doc->GetSentence(s)->AddEventMention(men);
			}

			for (int m = 0; m < doc->GetSentence(s)->gold_participant_mentions.size(); ++m) {
				Mention *men = doc->GetSentence(s)->gold_participant_mentions[m]->Copy();
				doc->GetSentence(s)->AddParticipantMention(men);
			}

			for (int m = 0; m < doc->GetSentence(s)->gold_time_mentions.size(); ++m) {
				Mention *men = doc->GetSentence(s)->gold_time_mentions[m]->Copy();
				doc->GetSentence(s)->AddTimeMention(men);
			}

			for (int m = 0; m < doc->GetSentence(s)->gold_loc_mentions.size(); ++m) {
				Mention *men = doc->GetSentence(s)->gold_loc_mentions[m]->Copy();
				doc->GetSentence(s)->AddLocMention(men);
			}
		}
	}
}

void CorefCorpus::OutputAntecedent(string filename) {
	ofstream logfile(filename.c_str(), ios::out);

	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		Document *doc = it->second;

		for (int i = 0; i < doc->predict_participant_mentions.size(); ++i) {
			Mention *m = doc->predict_participant_mentions[i];
			if (m->gold_entity_id == -1) continue;

			Entity *en = doc->gold_participant_entities[m->gold_entity_id];
			if (m->mention_type == PRONOMINAL) {
				Mention *ant_m = NULL;
				for (int j = 0; j < en->coref_mentions.size(); ++j) {
					if (en->coref_mentions[j]->mention_type != PRONOMINAL) {
						ant_m = en->coref_mentions[j];
						break;
					}
				}
				if (ant_m == NULL) continue;
				logfile<<doc->doc_id<<" "<<m->sent_id<<" "<<doc->GetSentence(m->sent_id)->toString()<<endl;
				logfile<<doc->doc_id<<" "<<ant_m->sent_id<<" "<<doc->GetSentence(ant_m->sent_id)->toString()<<endl;
				logfile<<m->mention_str<<"\t"<<ant_m->mention_str<<endl;

				logfile<<endl;
			}
		}
	}
	logfile.close();
}

void CorefCorpus::LoadAntecedentInfo(string filename) {
	ifstream infile(filename.c_str(), ios::in);

	string line;
	while(getline(infile, line)) {
		vector<string> fields;
		Utils::Split(line, '\t', fields);

		if (fields.size() < 9) continue;

		string docstr = fields[0];
		int start_index = docstr.find('(');
		int end_index = docstr.find(')');
		string doc_title = docstr.substr(start_index+1, end_index-start_index-1);

		if (documents.find(doc_title) == documents.end()) continue;

		Document *doc = documents[doc_title];

		int sent_num = atoi(fields[1].c_str());
		int start_offset = atoi(fields[2].c_str());
		int end_offset = atoi(fields[3].c_str()) - 1;

		int ant_sent_id = atoi(fields[5].c_str());
		int ant_start_offset = atoi(fields[6].c_str());
		int ant_end_offset = atoi(fields[7].c_str())-1;

		Mention *m = doc->GetSentence(sent_num)->FetchPredictMention(start_offset, end_offset);
		Mention *ant_m = doc->GetSentence(ant_sent_id)->FetchPredictMention(ant_start_offset, ant_end_offset);
		if (m != NULL && ant_m != NULL) {
			// if m is pronoun, update its head lemma
			if (m->mention_type == PRONOMINAL) {
				m->ant_mention_id = ant_m->mention_id;
				m->antecedent = ant_m;
				//m = ant_m;
				//m->Copy(ant_m);
//				m->head_lemma = ant_m->head_lemma;
//				m->head_pos = ant_m->head_pos;
//				m->head_index = ant_m->head_index;
//				m->head_word = ant_m->head_word;
//				m->head_span = ant_m->head_span;
//				m->head_pos = ant_m->head_pos;
//				m->head_lemma = ant_m->head_lemma;
//
//				m->CopyFeatures(ant_m);
			}
		} else {
			//cout<<"skip mention "<<doc->doc_id<<" "<<sent_num<<" "<<start_offset<<" "<<end_offset<<endl;
		}
	}
	infile.close();
}

double VecNorm(int *fvec) {
	double score = 0.0;
	int i = 0;
	while (fvec[i] != -1) {
		score += fvec[i] * fvec[i];
		i++;
	}
	score = sqrt(score);
	return score;
}

double VecNorm(ITEM *fvec) {
	double score = 0.0;
	int i = 0;
	while (fvec[i].wid != -1) {
		score += fvec[i].weight * fvec[i].weight;
		i++;
	}
	score = sqrt(score);
	return score;
}

ITEM* WordsToITEMVec(map<int, int> &word_map, double &vec_norm) {
	ITEM* vec = new ITEM[word_map.size() + 1];
	vector<pair<int, int> > sort_map = Utils::sortMap(word_map, true);
	int fnum = 0;
	for (fnum = 0; fnum < sort_map.size(); ++fnum) {
		vec[fnum].wid = sort_map[fnum].first;
		vec[fnum].weight = sort_map[fnum].second;
	}
	vec[fnum].wid = -1;
	vec[fnum].weight = 0;
	vec_norm = VecNorm(vec);

	return vec;
}

void CorefCorpus::BuildEventMentionFeatures() {
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		Document *local_doc = it->second;
		for (int i = 0; i < local_doc->SentNum(); ++i) {
			for (int j = 0; j < local_doc->GetSentence(i)->predict_mentions.size(); ++j) {
				Sentence *s = local_doc->GetSentence(i);
				Mention *m = s->predict_mentions[j];

				m->word_features.clear();
				string key = Utils::toLower(m->head_lemma) + "/" + m->head_pos[0];
				Dictionary::getEventFeatures(key, m->word_features);

				if (m->word_features.size() == 0) {
					m->word_features.push_back(m->head_lemma);
				}
				if (m->head_lemma != s->tokens[m->head_index]->second_lemma) {
					m->word_features.push_back(s->tokens[m->head_index]->second_lemma);
				}
			}
		}
	}
}

void CorefCorpus::BuildMentionFeatures() {
	map<string, int> srl_role_map;

	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		Document *local_doc = it->second;
		//cout<<"process doc "<<local_doc->doc_id<<endl;

		// compute within-doc salience
		map<int, int> headword_count;
		for (int i = 0; i < local_doc->SentNum(); ++i) {
			for (int j = 0; j < local_doc->GetSentence(i)->predict_mentions.size(); ++j) {
				Mention *m = local_doc->GetSentence(i)->predict_mentions[j];
				int word = vocabulary[m->head_lemma];
				if (headword_count.find(word) == headword_count.end()) {
					headword_count[word] = 1;
				} else {
					headword_count[word] += 1;
				}
			}
		}

		for (int i = 0; i < local_doc->SentNum(); ++i) {
			// build sentence level features
			vector<int> sent_args;
			for (int j = 0; j < local_doc->GetSentence(i)->predict_participant_mentions.size(); ++j) {
				sent_args.push_back(vocabulary[local_doc->GetSentence(i)->predict_participant_mentions[j]->head_lemma]);
			}
			if (sent_args.size() > 0) {
				sort(sent_args.begin(), sent_args.end());
				int *p = new int[sent_args.size()+1];
				std::copy(sent_args.begin(), sent_args.end(), p);
				p[sent_args.size()] = -1;
				local_doc->GetSentence(i)->srl_participant_vec = p;
			}

			sent_args.clear();
			for (int j = 0; j < local_doc->GetSentence(i)->predict_time_mentions.size(); ++j) {
				sent_args.push_back(vocabulary[local_doc->GetSentence(i)->predict_time_mentions[j]->head_lemma]);
			}
			if (sent_args.size() > 0) {
				sort(sent_args.begin(), sent_args.end());
				int *p = new int[sent_args.size()+1];
				std::copy(sent_args.begin(), sent_args.end(), p);
				p[sent_args.size()] = -1;
				local_doc->GetSentence(i)->srl_time_vec = p;
			}

			sent_args.clear();
			for (int j = 0; j < local_doc->GetSentence(i)->predict_loc_mentions.size(); ++j) {
				sent_args.push_back(vocabulary[local_doc->GetSentence(i)->predict_loc_mentions[j]->head_lemma]);
			}
			if (sent_args.size() > 0) {
				sort(sent_args.begin(), sent_args.end());
				int *p = new int[sent_args.size()+1];
				std::copy(sent_args.begin(), sent_args.end(), p);
				p[sent_args.size()] = -1;
				local_doc->GetSentence(i)->srl_loc_vec = p;
			}

			for (int j = 0; j < local_doc->GetSentence(i)->predict_mentions.size(); ++j) {
				Sentence *s = local_doc->GetSentence(i);
				Mention *m = local_doc->GetSentence(i)->predict_mentions[j];
				string key = m->head_lemma + "/" + m->head_pos[0];

				//m->doc_salience = (double)headword_count[vocabulary[m->head_lemma]]/local_doc->predict_mentions.size();

				m->doc_salience = (double)headword_count[vocabulary[m->head_lemma]];

				// mention words
				map<int, int> word_map;
				for (int k = m->start_offset; k <= m->end_offset; ++k) {
					string word = local_doc->GetSentence(i)->tokens[k]->lemma;

					//if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

					if(word_map.find(vocabulary[word]) == word_map.end()) {
						word_map[vocabulary[word]] = 1;
					} else {
						word_map[vocabulary[word]] += 1;
					}
				}
				m->word_vec = WordsToITEMVec(word_map, m->word_vec_norm);

				// context words
				word_map.clear();
				for (int k = max(0, m->start_offset-3); k < m->start_offset; ++k) {
					string word = local_doc->GetSentence(i)->tokens[k]->lemma;

					//if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

					if(word_map.find(vocabulary[word]) == word_map.end()) {
						word_map[vocabulary[word]] = 1;
					} else {
						word_map[vocabulary[word]] += 1;
					}
				}
				for (int k = m->end_offset+1; k < min(m->end_offset+3, local_doc->GetSentence(i)->TokenSize()); ++k) {
					string word = local_doc->GetSentence(i)->tokens[k]->lemma;

					//if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

					if(word_map.find(vocabulary[word]) == word_map.end()) {
						word_map[vocabulary[word]] = 1;
					} else {
						word_map[vocabulary[word]] += 1;
					}
				}
				m->context_vec = WordsToITEMVec(word_map, m->context_vec_norm);

				// useless...
			    //local_doc->BuildTFIDFMentionVec(m, idf_map);

				// synonym
				if (synonym_features.find(key) != synonym_features.end()) {
					m->head_synonym_vec = synonym_features[key];
				} else {
					if (Dictionary::HasSynonymFeatures(key)) {
						int *p = new int[Dictionary::synonym_features[key].size()+1];
						std::copy(Dictionary::synonym_features[key].begin(), Dictionary::synonym_features[key].end(), p);
						p[Dictionary::synonym_features[key].size()] = -1;
						m->head_synonym_vec = p;
						//m->head_synonym_norm = VecNorm(m->head_synonym_vec);
					}
				}

				// hypernym
				if (hypernym_features.find(key) != hypernym_features.end()) {
					m->head_hypernym_vec = hypernym_features[key];
				} else {
					if (Dictionary::HasHypernymFeatures(key)) {
						int *p = new int[Dictionary::hypernym_features[key].size()+1];
						std::copy(Dictionary::hypernym_features[key].begin(), Dictionary::hypernym_features[key].end(), p);
						p[Dictionary::hypernym_features[key].size()] = -1;
						m->head_hypernym_vec = p;
						//m->head_hypernym_norm = VecNorm(m->head_hypernym_vec);
					}
				}

				// verbnet
				if (verbnet_features.find(key) != verbnet_features.end()) {
					m->head_verbnet_vec = verbnet_features[key];
				} else {
					if (Dictionary::HasVerbnetFeatures(key)) {
						int *p = new int[Dictionary::verbnet_features[key].size()+1];
						std::copy(Dictionary::verbnet_features[key].begin(), Dictionary::verbnet_features[key].end(), p);
						p[Dictionary::verbnet_features[key].size()] = -1;
						m->head_verbnet_vec = p;
						//m->head_verbnet_norm = VecNorm(m->head_verbnet_vec);
					}
				}

				// framenet
				if (framenet_features.find(key) != framenet_features.end()) {
					m->head_framenet_vec = framenet_features[key];
				} else {
					if (Dictionary::HasFramenetFeatures(key)) {
						int *p = new int[Dictionary::framenet_features[key].size()+1];
						std::copy(Dictionary::framenet_features[key].begin(), Dictionary::framenet_features[key].end(), p);
						p[Dictionary::framenet_features[key].size()] = -1;
						m->head_framenet_vec = p;
						//m->head_framenet_norm = VecNorm(m->head_framenet_vec);
					}
				}

				// srl roles
				vector<int> srl_roles;
				for (map<string, vector<Mention*> >::iterator it = m->srl_args.begin(); it != m->srl_args.end(); ++it) {
					if (srl_role_map.find(it->first) == srl_role_map.end()) {
						srl_role_map[it->first] = srl_role_map.size();
					}
					srl_roles.push_back(srl_role_map[it->first]);
				}

				if (srl_roles.size() > 0) {
					sort(srl_roles.begin(), srl_roles.end());
					int *p = new int[srl_roles.size()+1];
					std::copy(srl_roles.begin(), srl_roles.end(), p);
					p[srl_roles.size()] = -1;
					m->srl_role_vec = p;
					//m->srl_role_norm = VecNorm(m->srl_role_vec);
				}

				// Add sentence level mention to srl_args
				vector<Mention*> parti;
				for (int k = 0; k < s->predict_participant_mentions.size(); ++k) {
					parti.push_back(s->predict_participant_mentions[k]);
				}
				/*int k = 0;
				// left participant
				for (; k < s->predict_participant_mentions.size(); ++k) {
					if (s->predict_participant_mentions[k]->head_index > m->head_index) {
						break;
					}
				}
				if (k > 0) {
					parti.push_back(s->predict_participant_mentions[k-1]);
				}
				k = s->predict_participant_mentions.size()-1;
				// right participant
				for (; k >=0; --k) {
					if (s->predict_participant_mentions[k]->head_index < m->head_index) {
						break;
					}
				}
				if (k < s->predict_participant_mentions.size()-1) {
					parti.push_back(s->predict_participant_mentions[k+1]);
				}*/

				if (parti.size() > 0) {
					m->srl_args["PARTICIPANT"] = parti;

					word_map.clear();
					for (int k = 0; k < parti.size(); ++k) {
						vector<string> words;
						Utils::Split(parti[k]->mention_str, ' ', words);
						for (int k1 = 0; k1 < words.size(); ++k1) {
							string word = words[k1];

							//if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

							if(word_map.find(vocabulary[word]) == word_map.end()) {
								word_map[vocabulary[word]] = 1;
							} else {
								word_map[vocabulary[word]] += 1;
							}
						}
					}
					m->srl_participant_vec = WordsToITEMVec(word_map, m->srl_participant_norm);
				}

				if (s->predict_time_mentions.size() > 0) {
					word_map.clear();
					for (int k = 0; k < s->predict_time_mentions.size(); ++k) {
						vector<string> words;
						Utils::Split(s->predict_time_mentions[k]->mention_str, ' ', words);
						for (int k1 = 0; k1 < words.size(); ++k1) {
							string word = words[k1];

							//if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

							if(word_map.find(vocabulary[word]) == word_map.end()) {
								word_map[vocabulary[word]] = 1;
							} else {
								word_map[vocabulary[word]] += 1;
							}
						}
					}
					m->srl_time_vec = WordsToITEMVec(word_map, m->srl_time_norm);
				}

				if (s->predict_loc_mentions.size() > 0) {
					word_map.clear();
					for (int k = 0; k < s->predict_loc_mentions.size(); ++k) {
						vector<string> words;
						Utils::Split(s->predict_loc_mentions[k]->mention_str, ' ', words);
						for (int k1 = 0; k1 < words.size(); ++k1) {
							string word = words[k1];

							//if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

							if(word_map.find(vocabulary[word]) == word_map.end()) {
								word_map[vocabulary[word]] = 1;
							} else {
								word_map[vocabulary[word]] += 1;
							}
						}
					}
					m->srl_loc_vec = WordsToITEMVec(word_map, m->srl_loc_norm);
				}

				Mention *arg = m->GetArg0Mention();
				if (arg != NULL) {
					word_map.clear();
					vector<string> words;
					Utils::Split(arg->mention_str, ' ', words);
					for (int k = 0; k < words.size(); ++k) {
						string word = words[k];

						//if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

						if(word_map.find(vocabulary[word]) == word_map.end()) {
							word_map[vocabulary[word]] = 1;
						} else {
							word_map[vocabulary[word]] += 1;
						}
					}
					m->srl_arg0_vec = WordsToITEMVec(word_map, m->srl_arg0_norm);
				}

				arg = m->GetArg1Mention();
				if (arg != NULL) {
					word_map.clear();
					vector<string> words;
					Utils::Split(arg->mention_str, ' ', words);
					for (int k = 0; k < words.size(); ++k) {
						string word = words[k];

						//if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

						if(word_map.find(vocabulary[word]) == word_map.end()) {
							word_map[vocabulary[word]] = 1;
						} else {
							word_map[vocabulary[word]] += 1;
						}
					}
					m->srl_arg1_vec = WordsToITEMVec(word_map, m->srl_arg1_norm);
				}

				arg = m->GetArg2Mention();
				if (arg != NULL) {
					word_map.clear();
					vector<string> words;
					Utils::Split(arg->mention_str, ' ', words);
					for (int k = 0; k < words.size(); ++k) {
						string word = words[k];

						//if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

						if(word_map.find(vocabulary[word]) == word_map.end()) {
							word_map[vocabulary[word]] = 1;
						} else {
							word_map[vocabulary[word]] += 1;
						}
					}
					m->srl_arg2_vec = WordsToITEMVec(word_map, m->srl_arg2_norm);
				}
			}
		}
	}
}

void CorefCorpus::BuildVocabulary() {
	// Build vocab and tf_idf info
	map<int, set<string> > word_in_doc;
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		Document *doc = it->second;
		int total_w;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);
			for (int j = 0; j < sent->TokenSize(); ++j) {
				if (vocabulary.find(sent->tokens[j]->lemma) == vocabulary.end()) {
					vocabulary[sent->tokens[j]->lemma] = vocabulary.size();
					id2word[vocabulary[sent->tokens[j]->lemma]] = sent->tokens[j]->lemma;
				}
				int w = vocabulary[sent->tokens[j]->lemma];
				if (doc->tf_map.find(w) == doc->tf_map.end()) {
					doc->tf_map[w] = 1;
				} else {
					doc->tf_map[w] += 1;
				}
				total_w += 1;

				if (word_in_doc.find(w) == word_in_doc.end()) {
					set<string> doclist;
					word_in_doc[w] = doclist;
				}
				word_in_doc[w].insert(doc->doc_id);
			}
		}

		for (map<int, float>::iterator dit = doc->tf_map.begin(); dit != doc->tf_map.end(); ++dit) {
			dit->second = (float)dit->second/total_w;
		}
	}

	int D = documents.size();
	for (map<int, set<string> >::iterator it = word_in_doc.begin(); it != word_in_doc.end(); ++it) {
		idf_map[it->first] = log((float)D/it->second.size());
	}
}

void CorefCorpus::AddVerbalMentions() {
	set<string> headlemmas;

	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		Document *doc = it->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);
			vector<int> tags(sent->TokenSize(), 0);
			for (int j = 0; j < sent->predict_mentions.size(); ++j) {
				Mention *men = sent->predict_mentions[j];
				tags[men->head_index] = 1;
				headlemmas.insert(men->head_lemma);
			}
			for (int j = 0; j < sent->predict_participant_mentions.size(); ++j) {
				Mention *men = sent->predict_participant_mentions[j];
				tags[men->head_index] = 1;
			}
			for (int j = 0; j < sent->predict_time_mentions.size(); ++j) {
				Mention *men = sent->predict_time_mentions[j];
				tags[men->head_index] = 1;
			}
			for (int j = 0; j < sent->predict_loc_mentions.size(); ++j) {
				Mention *men = sent->predict_loc_mentions[j];
				tags[men->head_index] = 1;
			}

			// add verbs
			for (int j = 0; j < sent->TokenSize(); ++j) {
				if (tags[j] == 0) {
				//if ((sent->tokens[j]->pos[0] == 'V') && tags[j] == 0) {
				//if ((sent->tokens[j]->pos[0] == 'N') && tags[j] == 0) {
				//if ((sent->tokens[j]->pos[0] == 'V' || sent->tokens[j]->pos[0] == 'N') && tags[j] == 0) {
					if (sent->tokens[j]->pos[0] == 'V') {
						tags[j] = 1;
						headlemmas.insert(sent->tokens[j]->lemma);

						Mention* m = new Mention();
						m->SetDocID(doc->doc_id);
						m->SetSentenceID(i);
						m->SetStartOffset(j);
						m->SetEndOffset(j);
						m->anno_prob = 1.0;
						m->anno_type = "EVENT";

						m->mention_length = m->end_offset - m->start_offset + 1;
						m->mention_str = sent->GetMentionStr(m);

						m->head_index = j;
						m->head_lemma = sent->tokens[j]->lemma;
						m->head_pos = sent->tokens[j]->pos;
						m->head_word = sent->tokens[j]->word;
						m->head_span = m->head_word;

						doc->GetSentence(i)->AddEventMention(m);
					}
				}
			}

			// add nouns
			for (int j = 0; j < sent->TokenSize(); ++j) {
				if (tags[j] == 0) {
				//if ((sent->tokens[j]->pos[0] == 'V') && tags[j] == 0) {
				//if ((sent->tokens[j]->pos[0] == 'N') && tags[j] == 0) {
				//if ((sent->tokens[j]->pos[0] == 'V' || sent->tokens[j]->pos[0] == 'N') && tags[j] == 0) {
					if (sent->tokens[j]->pos[0] == 'N') {
						//if (Dictionary::nom_to_verb.find(sent->tokens[j]->lemma) != Dictionary::nom_to_verb.end() ||
						//		headlemmas.find(sent->tokens[j]->lemma) != headlemmas.end()) {
							tags[j] = 1;
							headlemmas.insert(sent->tokens[j]->lemma);

							Mention* m = new Mention();
							m->SetDocID(doc->doc_id);
							m->SetSentenceID(i);
							m->SetStartOffset(j);
							m->SetEndOffset(j);
							m->anno_prob = 1.0;
							m->anno_type = "EVENT";

							m->mention_length = m->end_offset - m->start_offset + 1;
							m->mention_str = sent->GetMentionStr(m);

							m->head_index = j;
							m->head_lemma = sent->tokens[j]->lemma;
							m->head_pos = sent->tokens[j]->pos;
							m->head_word = sent->tokens[j]->word;
							m->head_span = m->head_word;

							doc->GetSentence(i)->AddEventMention(m);
						//}
					}
				}
			}

		}
	}
}

void CorefCorpus::SetPredictedMentionInfo(string filename) {
	vector<string> filenames;
	Utils::Split(filename, ',', filenames);
	for (int f = 0; f < filenames.size(); ++f) {
		ifstream infile(filenames[f].c_str(), ios::in);

		string line;
		while(getline(infile, line)) {
			if (line.find("#begin document") != string::npos) {
				// For each document.
				//int index = line.find_last_of(' ');
				//string doc_title = line.substr(index+1);

				int start_index = line.find('(');
				int end_index = line.find(')');
				string doc_title = line.substr(start_index+1, end_index-start_index-1);

				if (documents.find(doc_title) == documents.end()) continue;

				Document *doc = documents[doc_title];
				int old_sent = -1;
				map<int, int> head_dict;
				while (getline(infile, line) && line.find("#end document") == string::npos) {
					vector<string> fields;
					Utils::Split(line, '\t', fields);
					if (fields.size() < 4) continue;

					int sent_num = atoi(fields[0].c_str());
					int start_offset = atoi(fields[1].c_str());
					int end_offset = atoi(fields[2].c_str()) - 1;
					int head_index = atoi(fields[3].c_str());

					if (old_sent != -1 && sent_num != old_sent) {
						doc->GetSentence(old_sent)->SetHeadForPredictMentions(head_dict);
						head_dict.clear();
					}

					int token_size = doc->GetSentence(sent_num)->TokenSize();
					int key = start_offset * token_size + end_offset;
					head_dict[key] = head_index;
					old_sent = sent_num;
				}
				if (head_dict.size() > 0) {
					doc->GetSentence(old_sent)->SetHeadForPredictMentions(head_dict);
					head_dict.clear();
				}
			}
		}
		infile.close();
	}
}

void CorefCorpus::OutputCDPredictMentions(string topic_doc_file, string outputfile) {
	map<string, vector<string> > topic_to_docs;
	ifstream indexfile(topic_doc_file.c_str(), ios::in);
	string line;
	while (getline(indexfile, line)) {
		vector<string> fields;
		Utils::Split(line, '\t', fields);
		string topic = fields[0];
		string doc = fields[1];
		if (topic_to_docs.find(topic) == topic_to_docs.end()) {
			vector<string> p;
			topic_to_docs[topic] = p;
		}
		topic_to_docs[topic].push_back(doc);
	}
	indexfile.close();

	ofstream outfile(outputfile.c_str(), ios::out);
	for (map<string, vector<string> >::iterator it = topic_to_docs.begin(); it != topic_to_docs.end(); ++it) {
		string topic = it->first;
		int sentid = 0;
		for (int d = 0; d < it->second.size(); ++d) {
			string docid = it->second[d];
			if (documents.find(docid) == documents.end()) continue;

			Document *doc = documents[docid];
			for (int i = 0; i < doc->SentNum(); ++i) {
				for (int j = 0; j < doc->GetSentence(i)->predict_mentions.size(); ++j) {
					Mention *men = doc->GetSentence(i)->predict_mentions[j];
					outfile<<topic<<"\t"<<sentid<<"\t"
							<<men->start_offset<<"\t"<<men->end_offset+1<<"\t"<<men->head_index<<"\t"
							<<men->anno_type<<"\t"<<men->anno_prob<<"\t"<<men->mention_str<<"\t"<<men->head_lemma<<endl;
				}
				sentid++;
			}
		}
	}
	outfile.close();
}

void CorefCorpus::OutputPredictMentions(string filename) {
	ofstream outfile(filename.c_str(), ios::out);
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		Document *doc = it->second;

		/*for (int i = 0; i < doc->gold_mentions.size(); ++i) {
			Mention *men = doc->gold_mentions[i];
			//if (men->head_index < 0) {
				outfile<<doc->doc_id<<"\t"<<men->sent_id<<"\t"<<men->start_offset<<"\t"<<men->end_offset+1<<"\t"
					<<men->anno_type<<"\t"<<men->anno_prob<<"\t"<<men->mention_str<<"\t"<<men->head_lemma<<endl;
			//}
		}*/

		for (int i = 0; i < doc->predict_mentions.size(); ++i) {
			Mention *men = doc->predict_mentions[i];
			if (men->head_index < 0) {
				outfile<<doc->doc_id<<"\t"<<men->sent_id<<"\t"<<men->start_offset<<"\t"<<men->end_offset+1<<"\t"
					<<men->anno_type<<"\t"<<men->anno_prob<<"\t"<<men->mention_str<<"\t"<<men->head_lemma<<endl;
				men->head_index = 0;
			}
		}

		for (int i = 0; i < doc->predict_participant_mentions.size(); ++i) {
			Mention *men = doc->predict_participant_mentions[i];
			if (men->head_index < 0) {
				outfile<<doc->doc_id<<"\t"<<men->sent_id<<"\t"<<men->start_offset<<"\t"<<men->end_offset+1<<"\t"
					<<men->anno_type<<"\t"<<men->anno_prob<<"\t"<<men->mention_str<<"\t"<<men->head_lemma<<endl;
				men->head_index = 0;
			}
		}

		for (int i = 0; i < doc->predict_time_mentions.size(); ++i) {
			Mention *men = doc->predict_time_mentions[i];
			if (men->head_index < 0) {
				outfile<<doc->doc_id<<"\t"<<men->sent_id<<"\t"<<men->start_offset<<"\t"<<men->end_offset+1<<"\t"
					<<men->anno_type<<"\t"<<men->anno_prob<<"\t"<<men->mention_str<<"\t"<<men->head_lemma<<endl;
				men->head_index = 0;
			}
		}

		for (int i = 0; i < doc->predict_loc_mentions.size(); ++i) {
			Mention *men = doc->predict_loc_mentions[i];
			if (men->head_index < 0) {
				outfile<<doc->doc_id<<"\t"<<men->sent_id<<"\t"<<men->start_offset<<"\t"<<men->end_offset+1<<"\t"
					<<men->anno_type<<"\t"<<men->anno_prob<<"\t"<<men->mention_str<<"\t"<<men->head_lemma<<endl;
				men->head_index = 0;
			}
		}
	}
	outfile.close();
}

void CorefCorpus::BuildCorpusPredictEntitiesUsingGold() {
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		for (int i = 0; i < it->second->SentNum(); ++i) {
			Sentence *sent = it->second->GetSentence(i);
			sent->predict_entities.clear();
			for (int i = 0; i < sent->predict_mentions.size(); ++i) {
				Mention *m = sent->predict_mentions[i];
				// Invalid cluster. (e.g. pronouns)
				if (!m->valid || m->gold_entity_id < 0) continue;

				if (sent->predict_entities.find(m->gold_entity_id) == sent->predict_entities.end()) {
					Entity *en = new Entity();
					en->SetEntityID(m->gold_entity_id);
					sent->predict_entities[m->gold_entity_id] = en;
				}
				sent->predict_entities[m->gold_entity_id]->AddMention(m);
			}
		}
		it->second->BuildDocumentEntities(false);
	}

	for (map<string, Document*>::iterator it = topic_document.begin(); it != topic_document.end(); ++it) {
		//for (int i = 0; i < it->second->SentNum(); ++i) {
		//	it->second->GetSentence(i)->BuildSentenceEntities(gold);
		//}
		it->second->BuildDocumentEntities(false);
	}
}

// Merge document-level entities.
void CorefCorpus::BuildCorpusEntities(bool gold) {
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		for (int i = 0; i < it->second->SentNum(); ++i) {
			it->second->GetSentence(i)->BuildSentenceEntities(gold);
		}
		it->second->BuildDocumentEntities(gold);
	}

	for (map<string, Document*>::iterator it = topic_document.begin(); it != topic_document.end(); ++it) {
		//for (int i = 0; i < it->second->SentNum(); ++i) {
		//	it->second->GetSentence(i)->BuildSentenceEntities(gold);
		//}
		it->second->BuildDocumentEntities(gold);
	}
}

void CorefCorpus::BuildCorpusMentions(bool gold) {
	int n = 0;
	// within document
	for (map<string, Document*>::iterator it = documents.begin(); it!=documents.end(); ++it) {
		Document *doc = it->second;
		doc->BuildDocumentMentions(gold);
		//doc->BuildDocumentEntities(gold);

		// set mention id
		if (gold) {
			for (int i = 0; i < it->second->gold_mentions.size(); ++i) {
				it->second->gold_mentions[i]->SetMentionID(max_mention_id++);
			}
		} else {
			for (int i = 0; i < it->second->predict_mentions.size(); ++i) {
				it->second->predict_mentions[i]->SetMentionID(max_mention_id++);
			}
		}

		/*cout<<it->second->doc_id<<endl;
		for (int i = 0; i < it->second->SentNum(); ++i) {
			for (int j = 0; j < it->second->GetSentence(i)->gold_loc_mentions.size(); ++j) {
				Mention *m = it->second->GetSentence(i)->gold_loc_mentions[j];
				cout<<m->sent_id<<","<<m->start_offset<<","<<m->end_offset<<","<<m->mention_str<<"\t";
			}
		}
		cout<<endl;*/

		n += doc->sentences.size();
		//n += it->second->gold_mentions.size();
	}

	cout<<documents.size()<<" "<<n<<endl;

	// across documents
	for (map<string, Document*>::iterator it = topic_document.begin(); it != topic_document.end(); ++it) {
		it->second->BuildDocumentMentions(gold);
		//it->second->BuildDocumentEntities(gold);
	}
}

void CorefCorpus::RebuildDocumentEntities(bool gold) {
	for (map<string, Document*>::iterator iter = documents.begin(); iter!=documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			// Skip invalid coref clusters (id < 0).
			if (doc->GetSentence(i)->gold_mentions.size() == 0) continue;

			doc->GetSentence(i)->BuildSentenceEntities(gold);
		}
		doc->BuildDocumentEntities(gold);
	}
}

void CorefCorpus::DistanceBaseline() {
	for (map<string, Document*>::iterator iter = topic_document.begin();
				iter != topic_document.end(); ++iter) {
		for (int i = 0; i < iter->second->predict_mentions.size(); ++i) {
			Mention *m = iter->second->predict_mentions[i];
			double maxscore = 0;
			Mention *bestant = NULL;
			for (int j = 0; j < m->local_antecedents.size(); ++j) {
				if (m->local_antecedents[j].score > maxscore) {
					maxscore = m->local_antecedents[j].score;
					bestant = m->local_antecedents[j].m;
				}
			}
			m->antecedent = bestant;
		}

		int cid = 0;
		for (int i = 0; i < iter->second->predict_mentions.size(); ++i) {
			Mention *m = iter->second->predict_mentions[i];
			if (m->antecedent == NULL) {
				m->pred_entity_id = cid++;
			} else {
				m->pred_entity_id = m->antecedent->pred_entity_id;
			}
		}
	}
}

double CorefCorpus::EntitySimilarity(Entity *e1, Entity *e2) {
	double max_sim = 0.0;
	for (int i = 0; i < e1->Size(); ++i) {
		Mention *m1 = e1->GetCorefMention(i);
		for (int j = 0; j < e2->Size(); ++j) {
			Mention *m2 = e2->GetCorefMention(j);
			double sim = MentionGlobalSimilarity(m1, m2);
			if (sim > max_sim) {
				max_sim = sim;
			}
		}
	}
	return max_sim;
}

double CorefCorpus::MentionGlobalSimilarity(Mention *m1, Mention *m2) {
	int mid1 = m1->mention_id;
	int mid2 = m2->mention_id;

	if (global_pairwise_distance.find(mid1) != global_pairwise_distance.end()) {
		if (global_pairwise_distance[mid1].find(mid2) != global_pairwise_distance[mid1].end()) {
			double sim = global_pairwise_distance[mid1][mid2];
			return sim;
		}
	}

	if (global_pairwise_distance.find(mid2) != global_pairwise_distance.end()) {
		if (global_pairwise_distance[mid2].find(mid1) != global_pairwise_distance[mid2].end()) {
			double sim = global_pairwise_distance[mid2][mid1];
			return sim;
		}
	}
	return 0.0;
}

double CorefCorpus::MentionSimilarity(Mention *m1, Mention *m2) {
	int mid1 = m1->mention_id;
	int mid2 = m2->mention_id;

	if (global_pairwise_distance.find(mid1) != global_pairwise_distance.end()) {
		if (global_pairwise_distance[mid1].find(mid2) != global_pairwise_distance[mid1].end()) {
			double sim = global_pairwise_distance[mid1][mid2];
			return sim;
		}
	}

	if (global_pairwise_distance.find(mid2) != global_pairwise_distance.end()) {
		if (global_pairwise_distance[mid2].find(mid1) != global_pairwise_distance[mid2].end()) {
			double sim = global_pairwise_distance[mid2][mid1];
			return sim;
		}
	}
	return 0.0;
}

void CorefCorpus::BuildClusters(Document *doc, double threshold) {
	doc->predict_entities.clear();

	PairwiseClustering cl;
	cl.threshold = threshold;
	cl.maxNpID = doc->predict_mentions.size();

	// first map mentions with the same head word to one cluster.
	for (int i = 0; i < doc->predict_mentions.size(); ++i) {
		for (int j = 0; j < i; ++j) {
			Mention *m1 = doc->predict_mentions[i];
			Mention *m2 = doc->predict_mentions[j];
			cl.setWeight(i,j, MentionSimilarity(m1,m2));
		}
	}

	cl.Clustering();

	for (int i = 0; i < cl.cluster_assignments.size(); ++i) {
		int cid = cl.cluster_assignments[i];
		if (doc->predict_entities.find(cid) == doc->predict_entities.end()) {
			Entity *en = new Entity();
			en->SetEntityID(cid);
			en->AddMention(doc->predict_mentions[i]);
			doc->predict_entities[cid] = en;
		} else {
			doc->predict_entities[cid]->AddMention(doc->predict_mentions[i]);
		}
		doc->predict_mentions[i]->pred_entity_id = cid;
	}
}

void CorefCorpus::PairwiseBaseline(string modelfile, string embeddingfile,
		double local_threshold, double global_threshold) {
	PairwiseModel model;
	model.LoadEmbeddings(embeddingfile);
	model.LoadModel(modelfile);

	ofstream outfile("pairwise.distance.txt", ios::out);

	for (map<string, vector<Document*> >::iterator iter = topic_to_documents.begin();
				iter != topic_to_documents.end(); ++iter) {
	//for (map<string, vector<Document*> >::iterator iter = predict_document_clusters.begin();
	//			iter != predict_document_clusters.end(); ++iter) {
		vector<Document*> p_documents = iter->second;
		cout<<"topic "<<iter->first<<" containing documents "<<p_documents.size()<<endl;

		// loading distance
		global_pairwise_distance.clear();
		CRFPP::TaggerImpl x;
		for (int d = 0; d < p_documents.size(); ++d) {
			Document *doc = p_documents[d];
			for (int j = 0; j < doc->predict_mentions.size(); ++j) {

				Mention *m = doc->predict_mentions[j];
				Sentence *sent = doc->GetSentence(m->sent_id);

				int mid1 = m->mention_id;
				if (global_pairwise_distance.find(mid1) == global_pairwise_distance.end()) {
					map<int, double> p;
					global_pairwise_distance[mid1] = p;
				}

				// within doc
				for (int j1 = 0; j1 < j; ++j1) {
					Mention *ant_m = doc->predict_mentions[j1];
					int mid2 = ant_m->mention_id;
					if (global_pairwise_distance[mid1].find(mid2) == global_pairwise_distance[mid1].end()) {
						global_pairwise_distance[mid1][mid2] = 0.0;
					}

					Sentence *ant_sent = doc->GetSentence(ant_m->sent_id);

					map<string, float> fvec;
					model.GenCDEventPairFeatures(m, ant_m, fvec);
					model.GenEventPairEmbeddingFeatures(m, sent, ant_m, ant_sent, fvec);

					vector<map<string, float> > fvecs;
					fvecs.push_back(fvec);
					x.x = ant_m->mention_id;

					int y = 0;
					if (m->gold_entity_id != -1 && m->gold_entity_id == ant_m->gold_entity_id) {
						y = 1;
					}

					x.InitLRTagger(y, model.decoder_feature_index);
					model.decoder_feature_index->buildFeatures(&x, fvecs, false);
					model.decoder_feature_index->buildTagger(&x);
					x.recomputeCost();
					x.node_[0]->calcLRProb();
					global_pairwise_distance[mid1][mid2] = x.node_[0]->prob;

					if (m->gold_entity_id != -1 && ant_m->gold_entity_id != -1) {
						int error = 0;
						if ((x.node_[0]->prob > 0.5 && y == 0) || (x.node_[0]->prob < 0.5 && y == 1)) error = 1;
						if (error == 1) {
						outfile<<doc->doc_id<<","
								<<m->head_word<<","<<sent->sent_id<<","<<m->gold_entity_id<<"\t"
								<<ant_m->doc_id<<","<<ant_m->head_word<<","<<ant_m->sent_id<<","<<ant_m->gold_entity_id<<"\t"
								<<x.node_[0]->prob<<"\t"<<error<<endl;
						}
//						double u = runiform();
//						if (u < 0.1) {
//							global_pairwise_distance[mid1][mid2] = y;
//						}
					}
				}

				// across doc
				//if (d > 0) {
				//	Document *ant_doc = p_documents[d-1];
				//if (doc->predict_doc_ant != "") {
					//Document *ant_doc = documents[doc->predict_doc_ant];
					//string pred_topic = Utils::int2string(ant_doc->predict_topic);
					//if (pred_topic != iter->first) continue;
				for (int d1 = d-1; d1 >= 0; --d1) {
					Document *ant_doc = p_documents[d1];
					for (int j1 = 0; j1 < ant_doc->predict_mentions.size(); ++j1) {
						Mention *ant_m = ant_doc->predict_mentions[j1];

						int mid2 = ant_m->mention_id;
						if (global_pairwise_distance[mid1].find(mid2) == global_pairwise_distance[mid1].end()) {
							global_pairwise_distance[mid1][mid2] = 0.0;
						}

						Sentence *ant_sent = ant_doc->GetSentence(ant_m->sent_id);

						map<string, float> fvec;
						model.GenCDEventPairFeatures(m, ant_m, fvec);
						model.GenEventPairEmbeddingFeatures(m, sent, ant_m, ant_sent, fvec);

						vector<map<string, float> > fvecs;
						fvecs.push_back(fvec);
						x.x = ant_m->mention_id;

						int y = 0;
						if (m->gold_entity_id != -1 && m->gold_entity_id == ant_m->gold_entity_id) {
							y = 1;
						}

						x.InitLRTagger(y, model.decoder_feature_index);
						model.decoder_feature_index->buildFeatures(&x, fvecs, false);
						model.decoder_feature_index->buildTagger(&x);
						x.recomputeCost();
						x.node_[0]->calcLRProb();
						global_pairwise_distance[mid1][mid2] = x.node_[0]->prob;

						if (m->gold_entity_id != -1 && ant_m->gold_entity_id != -1) {
							int error = 0;
							if ((x.node_[0]->prob > 0.5 && y == 0) || (x.node_[0]->prob < 0.5 && y == 1)) error = 1;
							if (error == 1) {
							outfile<<doc->doc_id<<","
									<<m->head_word<<","<<sent->sent_id<<","<<m->gold_entity_id<<"\t"
									<<ant_m->doc_id<<","<<ant_m->head_word<<","<<ant_m->sent_id<<","<<ant_m->gold_entity_id<<"\t"
									<<x.node_[0]->prob<<"\t"<<error<<endl;
							}
//							double u = runiform();
//							if (u < 0.1) {
//								global_pairwise_distance[mid1][mid2] = y;
//							}
						}
					}
				}
			}
		}
		cout<<"Finish global pairwise distance"<<endl;
		outfile.close();

		// =======Compute clusters within topic======
		/*vector<Entity*> entities;
		for (int i = 0; i < p_documents.size(); ++i) {
			Document *doc = p_documents[i];

			// within document clustering
			BuildClusters(doc, local_threshold);

			for (map<int, Entity*>::iterator it = doc->predict_entities.begin(); it !=
					doc->predict_entities.end(); ++it) {
				entities.push_back(it->second);
			}
		}

		cout<<"Number of entities: "<<entities.size()<<endl;

		// cross-document clustering
		PairwiseClustering cl;
		cl.threshold = global_threshold;
		cl.maxNpID = entities.size();
		for (int i = 0; i < cl.maxNpID; ++i) {
			for (int j = 0; j < i; ++j) {
				double dist = EntitySimilarity(entities[i], entities[j]);
				if (dist > 0) {
					cl.setWeight(i,j,dist);
				}
			}
		}

		cl.Clustering();

		int max_cid = 0;
		for (int i = 0; i < cl.maxNpID; ++i) {
			int cid = cl.cluster_assignments[i];
			if (cid > max_cid) max_cid = cid;
			//cout<<entities[i]->Size()<<": ";
			for (int j = 0; j < entities[i]->Size(); ++j) {
				entities[i]->GetCorefMention(j)->pred_entity_id = cid;
				//cout<<entities[i]->GetCorefMention(j)->pred_entity_id<<" ";
			}
			//cout<<endl;
		}

		cout<<"finish clustering for topic "<<iter->first<<endl;
		*/
	}
}

void CorefCorpus::LoadHDPResults(string filename) {
	ifstream infile(filename.c_str(), ios::in);
	map<int, map<int, vector<int> > > doc_word_assignments;
	// Read header.
	string str;
	getline(infile, str); //"d w z t"
	// Assume not shuffled!!!
	while(getline(infile, str)) {
		vector<string> fields;
		Utils::Split(str, ' ', fields);
		if (fields.size() < 4) continue;
		int doc_id = atoi(fields[0].c_str());
		int word_id = atoi(fields[1].c_str());
		int topic_id = atoi(fields[2].c_str());
		int table_id = atoi(fields[3].c_str());
		if (doc_word_assignments.find(doc_id) == doc_word_assignments.end()) {
			map<int, vector<int> > p;
			doc_word_assignments[doc_id] = p;
		}
		if (doc_word_assignments[doc_id].find(word_id) == doc_word_assignments[doc_id].end()) {
			vector<int> p;
			doc_word_assignments[doc_id][word_id] = p;
		}
		doc_word_assignments[doc_id][word_id].push_back(topic_id);
	}
	infile.close();

	map<string, int> vocabulary;
	int docid = 0;
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		map<int, int> wordcount;
		map<int, vector<int> > word_assignments = doc_word_assignments[docid];
		for (int i = 0; i <doc->predict_mentions.size(); ++i) {
			Mention *m = doc->predict_mentions[i];
			if (vocabulary.find(m->head_lemma) == vocabulary.end()) {
				vocabulary[m->head_lemma] = vocabulary.size();
			}
			int wordid = vocabulary[m->head_lemma];
			if (wordcount.find(wordid) == wordcount.end()) {
				wordcount[wordid] = 1;
			} else {
				wordcount[wordid] += 1;
			}
			int index = wordcount[wordid] - 1;
			m->pred_entity_id = word_assignments[wordid][index];
		}
		docid++;
	}
}

void CorefCorpus::OutputEvents(string filename) {
	map<string, int> vocab;
	for (map<string, Document* >::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);

			for (int j = 0; j < sent->predict_mentions.size(); ++j) {
				Mention *m = sent->predict_mentions[j];

				string lemma_lower = Utils::toLower(m->head_lemma);
				string key = lemma_lower + "/" + m->head_pos[0];
				if (vocab.find(key) == vocab.end())
					vocab[key] = 1;
				else
					vocab[key] += 1;

				lemma_lower = Utils::toLower(sent->tokens[m->head_index]->second_lemma);
				key = lemma_lower + "/" + sent->tokens[m->head_index]->lemma_pos;
				if (vocab.find(key) == vocab.end())
					vocab[key] = 1;
				else
					vocab[key] += 1;

				if (m->Length() > 1) {
					// unigrams
					for (int k = m->start_offset; k <= m->end_offset; ++k) {
						if (k == m->head_index) continue;

						string pos = sent->tokens[k]->pos;
						if (pos[0] == 'V' || pos[0] == 'N' || pos[0] == 'J' || pos[0] == 'R' || pos[0] == 'C') {
							key = Utils::toLower(sent->tokens[k]->lemma) + "/" + pos[0];
							if (vocab.find(key) == vocab.end())
								vocab[key] = 1;
							else
								vocab[key] += 1;

							key = Utils::toLower(sent->tokens[k]->second_lemma) + "/" + sent->tokens[k]->lemma_pos;
							if (vocab.find(key) == vocab.end())
								vocab[key] = 1;
							else
								vocab[key] += 1;
						}
					}
					// bigrams
					for (int k = m->start_offset; k < m->end_offset; ++k) {
						key = Utils::toLower(sent->tokens[k]->lemma) + "_" + Utils::toLower(sent->tokens[k+1]->lemma) + "/O";
						if (vocab.find(key) == vocab.end())
							vocab[key] = 1;
						else
							vocab[key] += 1;
					}
					// whole
					if (m->Length() > 2) {
						key = Utils::toLower(sent->tokens[m->start_offset]->lemma);
						for (int k = m->start_offset+1; k <= m->end_offset; ++k) {
							key += "_" + Utils::toLower(sent->tokens[k]->lemma);
						}
						key = key + "/O";
						if (vocab.find(key) == vocab.end())
							vocab[key] = 1;
						else
							vocab[key] += 1;
					}
				}
			}
		}
	}

	vector<pair<string, int> > sorted_vocab = Utils::sortMap<string, int>(vocab);
	cout<<"vocab size "<<sorted_vocab.size()<<endl;
	ofstream outfile(filename.c_str(), ios::out);
	for (int i = 0; i <sorted_vocab.size(); ++i) {
		outfile<<sorted_vocab[i].first<<"\t"<<sorted_vocab[i].second<<endl;
	}
	outfile.close();
}

void CorefCorpus::OutputEventVocab(string filename) {
	string event_file = filename + ".event";
	ofstream eventfile(event_file.c_str(), ios::out);
	string time_file = filename + ".time";
	ofstream timefile(time_file.c_str(), ios::out);
	string location_file = filename + ".location";
	ofstream locationfile(location_file.c_str(), ios::out);
	string dep_args_file = filename + ".args";
	ofstream depargfile(dep_args_file.c_str(), ios::out);
	string srl_args_file = filename + ".srl.args";
	ofstream srlargfile(srl_args_file.c_str(), ios::out);

	map<string, int> head_vocab;
	map<int, int> mention_length;
	int total_mention_count = 0;
	map<string, int> time_vocab;
	map<int, int> time_length;
	int total_time_count = 0;
	map<string, int> loc_vocab;
	map<int, int> loc_length;
	int total_loc_count = 0;
	map<string, int> a0_vocab;
	map<int, int> a0_length;
	int total_a0_count = 0;
	map<string, int> a1_vocab;
	map<int, int> a1_length;
	int total_a1_count = 0;

	for (map<string, Document* >::iterator iter = topic_document.begin(); iter != topic_document.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);
			total_mention_count += sent->predict_mentions.size();
			for (int j = 0; j < sent->predict_mentions.size(); ++j) {
				Mention *m = sent->predict_mentions[j];
				head_vocab[m->head_lemma] = 1;
				int len = m->Length();
				if (mention_length.find(len) == mention_length.end()) {
					mention_length[len] = 1;
				} else {
					mention_length[len] += 1;
				}

				eventfile<<i<<" "<<m->ToString()<<" "<<m->head_span<<endl;
				if (m->time != NULL) {
					timefile<<i<<" "<<m->time->ToString()<<endl;
					total_time_count++;
					time_vocab[m->time->head_lemma] = 1;
					int len = m->time->Length();
					if (time_length.find(len) == time_length.end()) {
						time_length[len] = 1;
					} else {
						time_length[len] += 1;
					}
				}
				if (m->location != NULL) {
					locationfile<<i<<" "<<m->location->ToString()<<endl;
					total_loc_count++;
					loc_vocab[m->location->head_lemma] = 1;
					int len = m->location->Length();
					if (loc_length.find(len) == loc_length.end()) {
						loc_length[len] = 1;
					} else {
						loc_length[len] += 1;
					}
				}
				if (m->arguments.size() > 0) {
					for (map<string, Argument*>::iterator it = m->arguments.begin(); it != m->arguments.end(); ++it) {
						depargfile<<i<<" "<<j<<" "<<sent->GetSpanStr(it->second->word_id, it->second->word_id)
								<<":"<<it->second->toRelStr()<<endl;
					}
				}
				if (m->srl_args.size() > 0) {
					/*for (map<string, Mention*>::iterator it = m->srl_args.begin(); it != m->srl_args.end(); ++it) {
						if (it->first == "A0") {
							total_a0_count++;
							a0_vocab[it->second->head_lemma] = 1;
							int len = it->second->Length();
							if (a0_length.find(len) == a0_length.end()) {
								a0_length[len] = 1;
							} else {
								a0_length[len] += 1;
							}
							srlargfile<<"A0: "<<m->ToString()<<" "<<m->head_span<<"\t"<<it->second->head_lemma<<"\t"<<it->second->ToString()<<endl;
						} else if (it->first == "A1") {
							total_a1_count++;
							a1_vocab[it->second->head_lemma] = 1;
							int len = it->second->Length();
							if (a1_length.find(len) == a1_length.end()) {
								a1_length[len] = 1;
							} else {
								a1_length[len] += 1;
							}
							srlargfile<<"A1: "<<m->ToString()<<" "<<m->head_span<<"\t"<<it->second->head_lemma<<"\t"<<it->second->ToString()<<endl;
						}
					}*/
				}
			}
		}
	}
	eventfile.close();
	timefile.close();
	locationfile.close();
	depargfile.close();
	srlargfile.close();

	cout<<"mention: "<<total_mention_count<<" "<<head_vocab.size()<<endl;
	cout<<"time: "<<total_time_count<<" "<<time_vocab.size()<<endl;
	cout<<"loc: "<<total_loc_count<<" "<<loc_vocab.size()<<endl;
	cout<<"a0: "<<total_a0_count<<" "<<a0_vocab.size()<<endl;
	cout<<"a1: "<<total_a1_count<<" "<<a1_vocab.size()<<endl;

	cout<<"mention: ";
	for (map<int, int>::iterator it = mention_length.begin(); it != mention_length.end(); ++it) {
		cout<<it->first<<":"<<it->second<<" ";
	}
	cout<<endl;
	cout<<"time: ";
	for (map<int, int>::iterator it = time_length.begin(); it != time_length.end(); ++it) {
		cout<<it->first<<":"<<it->second<<" ";
	}
	cout<<endl;
	cout<<"loc: ";
	for (map<int, int>::iterator it = loc_length.begin(); it != loc_length.end(); ++it) {
		cout<<it->first<<":"<<it->second<<" ";
	}
	cout<<endl;
	cout<<"a0: ";
	for (map<int, int>::iterator it = a0_length.begin(); it != a0_length.end(); ++it) {
		cout<<it->first<<":"<<it->second<<" ";
	}
	cout<<endl;
	cout<<"a1: ";
	for (map<int, int>::iterator it = a1_length.begin(); it != a1_length.end(); ++it) {
		cout<<it->first<<":"<<it->second<<" ";
	}
	cout<<endl;
}

void CorefCorpus::BuildGoldSentClusters() {
	for (map<string, vector<Document*> >::iterator it = topic_to_documents.begin();
			it != topic_to_documents.end(); ++it) {
		int sent_num = 0;
		int pred_num = 0;
		for (int d = 0; d < it->second.size(); ++d) {
			Document *doc = it->second[d];
			for (int i = 0; i < doc->SentNum(); ++i) {
				Sentence *sent = doc->GetSentence(i);
				string sent_key = sent->sent_key;
				for (int j = 0; j < sent->gold_mentions.size(); ++j) {
					int gold_en = sent->gold_mentions[j]->gold_entity_id;
					Entity *en = topic_document[doc->topic_id]->gold_entities[gold_en];
					for (int k = 0; k < en->coref_mentions.size(); ++k) {
						string key = en->coref_mentions[k]->doc_id + "\t" + Utils::int2string(en->coref_mentions[k]->sent_id);
						if (sent_key == key) continue;
						sent->gold_cand_sents[key] = documents[en->coref_mentions[k]->doc_id]->GetSentence(en->coref_mentions[k]->sent_id);
					}
				}
			}
			sent_num += doc->SentNum();
			pred_num += doc->predict_mentions.size();
		}
		cout<<"Topic "<<it->first<<" has sentences "<<sent_num<<" average #pred/#sent is "<<pred_num/sent_num<<endl;
	}
}

void CorefCorpus::NearestSentences(int topK) {
	for (map<string, vector<Document*> >::iterator it = topic_to_documents.begin(); it != topic_to_documents.end(); ++it) {
		for (int d = 0; d < it->second.size(); ++d) {
			for (int i = 0; i < it->second[d]->SentNum(); ++i) {
				Sentence *sent = it->second[d]->GetSentence(i);
				string sent1 = sent->sent_key;
				if (sentence_pairwise_distance.find(sent1) == sentence_pairwise_distance.end()) {
					map<string, double> simmap;
					sentence_pairwise_distance[sent1] = simmap;
				}
				vector<pair<Sentence*, double> > cands;
				for (int d1 = 0; d1 < it->second.size(); ++d1) {
					for (int j = 0; j < it->second[d1]->SentNum(); ++j) {
						if (d1 == d && i == j) continue;
						Sentence *cand_sent = it->second[d1]->GetSentence(j);

						//if (!sent->similarity_check(cand_sent)) continue;

						string sent2 = cand_sent->sent_key;
						double sim = 0.0;
						assert (sentence2idx.find(sent1) != sentence2idx.end() &&
										sentence2idx.find(sent2) != sentence2idx.end());

						vector<double> v1 = sentence_vecs[sentence2idx[sent1]];
						vector<double> v2 = sentence_vecs[sentence2idx[sent2]];
						for (int i = 0; i < v1.size(); ++i) {
							sim += v1[i] * v2[i];
						}

						if (sentence_pairwise_distance[sent1].find(sent2) == sentence_pairwise_distance[sent1].end()) {
							sentence_pairwise_distance[sent1][sent2] = sim;
						}

						cands.push_back(make_pair(cand_sent, sim));
					}
				}
				sort(cands.begin(), cands.end(), Utils::decrease_second<Sentence*, double>);
				for (int j = 0; j < min((int)cands.size(), topK); ++j) {
					sent->cand_sents[cands[j].first->sent_key] = cands[j].first;
				}
			}
		}
	}

/*	ofstream outfile("sent.cluster.txt", ios::out);

	// Evaluate
	double acc_sum = 0;
	int total = 0;
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		Document *doc = it->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);

			// Debugging
			outfile<<"For sentence : "<<sent->toString()<<endl;
			outfile<<"gold:"<<endl;
			for (map<string, Sentence*>::iterator it1 = sent->gold_cand_sents.begin();
								it1 != sent->gold_cand_sents.end(); ++it1) {
				outfile<<it1->second->toString()<<endl;
			}
			outfile<<endl;
			outfile<<"predict:"<<endl;
			for (map<string, Sentence*>::iterator it1 = sent->cand_sents.begin();
								it1 != sent->cand_sents.end(); ++it1) {
				outfile<<it1->second->toString()<<endl;
			}
			outfile<<endl;

			if (sent->gold_cand_sents.size() == 0) continue;
			int n = sent->gold_cand_sents.size();
			int correct = 0;
			for (map<string, Sentence*>::iterator it1 = sent->gold_cand_sents.begin();
					it1 != sent->gold_cand_sents.end(); ++it1) {
				if (sent->cand_sents.find(it1->first) != sent->cand_sents.end()) {
					correct++;
				}
			}
			double rec = (double)correct/n;
			acc_sum += rec;
			total += 1;
		}
	}
	cout<<"average sent cluster rec: "<<(double)acc_sum/total<<endl;
	outfile.close();
	*/
}

void CorefCorpus::LoadSentVec(string filename, string indexfilename) {
	sentence2idx.clear();

	ifstream indexfile(indexfilename.c_str(), ios::in);
	string str;
	int idx = 0;
	while (getline(indexfile, str)) {
		vector<string> fields;
		Utils::Split(str, '\t', fields);

		int i = fields[0].find(')');
		string docid = fields[0].substr(1, i-1);

		string sentid = fields[1];
		sentence2idx[docid + "\t" + sentid] = idx++;
	}
	indexfile.close();

	cout<<"total sentences: "<<sentence2idx.size()<<endl;

	ifstream infile(filename.c_str(), ios::in);
	while(getline(infile, str)) {
		vector<string> splits;
		Utils::Split(str, ',', splits);
		vector<double> vec;
		double sum = 0;
		for (int i = 0; i < splits.size(); ++i) {
			if (splits[i] == "") continue;
			double v = atof(splits[i].c_str());
			vec.push_back(v);
			sum += v * v;
		}
		sum = sqrt(sum);
		// scale vector to unit vec
		for (int i = 0; i < vec.size(); ++i) {
			vec[i] = vec[i]/sum;
		}
		sentence_vecs.push_back(vec);
	}
	infile.close();
}

void CorefCorpus::OutputVocab(string filename) {
	map<string, int> vocab;
	for (map<string, Document* >::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			for (int j = 0; j < doc->GetSentence(i)->TokenSize(); ++j) {
				//string token = doc->GetSentence(i)->GetToken(j)->word;
				string lemma = doc->GetSentence(i)->GetToken(j)->lemma + "/" + doc->GetSentence(i)->GetToken(j)->pos.substr(0,1);
				//vocab[token] = 1;
				vocab[lemma] = 1;
			}
//			for (int j = 0; j < doc->predict_mentions.size(); ++j) {
//				string span = doc->predict_mentions[j]->head_span;
//				vocab[span] = 1;
//			}
		}
	}

	ofstream outfile(filename.c_str(), ios::out);
	for (map<string, int>::iterator it = vocab.begin(); it != vocab.end(); ++it) {
		outfile<<it->first<<endl;
	}
	outfile.close();
}

// datafile: word_count word_id:count word_id:count ...
// wordmapfile: word word_id
// indexfile: doc_id doc_title mention_id:word_id mention_id:word_id ...
void CorefCorpus::GenHDPData(string datapath) {
	//string filename = datapath + "debug.hdp.data";
	string filename = datapath + "hdp.data";
	ofstream outfile(filename.c_str(), ios::out);
	int docid = 0;
	map<string, int> vocabulary;
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		//cout<<"process "<<iter->second->doc_id<<endl;
		Document *doc = iter->second;
		map<int, int> wordcount;
		for (int i = 0; i <doc->predict_mentions.size(); ++i) {
			Mention *m = doc->predict_mentions[i];
			if (vocabulary.find(m->head_lemma) == vocabulary.end()) {
				vocabulary[m->head_lemma] = vocabulary.size();
			}
			int wordid = vocabulary[m->head_lemma];
			if (wordcount.find(wordid) == wordcount.end()) {
				wordcount[wordid] = 1;
			} else {
				wordcount[wordid] += 1;
			}
		}
		outfile<<wordcount.size()<<" ";
		for (map<int, int>::iterator it = wordcount.begin(); it != wordcount.end(); ++it) {
			outfile<<it->first<<":"<<it->second<<" ";
		}
		outfile<<endl;
		docid++;

		//if (docid == 2) break;
	}
	outfile.close();
	cout<<"vocabulary size: "<<vocabulary.size()<<endl;
	cout<<"document size: "<<documents.size()<<endl;
}

// datafile: word_count word_id:count word_id:count ...
// wordmapfile: word word_id
// indexfile: doc_id doc_title mention_id:word_id mention_id:word_id ...
void CorefCorpus::GenLDAdata(string indexfilename, string datafilename, string wordmapfile) {
	ofstream outfile(datafilename.c_str(), ios::out);
	ofstream indexfile(indexfilename.c_str(), ios::out);
	string line;
	int doc_id = 0;
	map<string, int> wordmap;
	for (map<string, Document *>::iterator iter = documents.begin();
			iter != documents.end(); ++iter) {
		Document *doc = iter->second;

		vector<int> word_vec;
		// lda_index
		indexfile<<doc_id<<" "<<doc->DocID()<<" ";

		for (int i = 0; i <doc->predict_mentions.size(); ++i) {
			Mention *m = doc->predict_mentions[i];
			// Skip pronouns!!!!!
			//if (m->IsPronoun()) continue;

		    // Lemma.
			string headword = m->head_lemma;
			if (wordmap.find(headword) == wordmap.end()) {
				wordmap[headword] = wordmap.size();
			}

			indexfile<<i<<":"<<wordmap[headword]<<" ";
			word_vec.push_back(wordmap[headword]);
		}

		outfile<<word_vec.size()<<" ";
		for (int i = 0; i < word_vec.size(); ++i) {
			outfile<<word_vec[i]<<":1 ";
		}
		outfile<<endl;

		indexfile<<endl;

		doc_id ++;
	}
	outfile.close();
	indexfile.close();

	//WriteWordMap(wordmap, wordmapfile);
}

void CorefCorpus::LoadCoNLLResults(string filename) {
	CoNLLDocumentReader read(filename);

	Document *newdoc = NULL;
	while ( (newdoc = read.ReadDocument()) != NULL) {
		if (topic_document.find(newdoc->DocID()) == topic_document.end()) continue;
		Document *doc = topic_document[newdoc->DocID()];
		for (int i = 0; i < doc->SentNum(); ++i) {
			if (doc->GetSentence(i)->gold_mentions.size() == 0) continue;
			Sentence *sent = doc->GetSentence(i);
			Sentence *newsent = NULL;
			for (int j = 0; j < newdoc->SentNum(); ++j) {
				newsent = newdoc->GetSentence(j);
				if (sent->Equal(newsent)) break;
			}
			if (newsent != NULL) { // prediction skip non-annotated sentences.
				for (int k = 0; k < sent->predict_mentions.size(); ++k) {
					// seting the predict_entity_id for each predict mention.
					int k1;
					for (k1 = 0; k1 < newsent->gold_mentions.size(); ++k1) {
						if(sent->predict_mentions[k]->StartOffset() == newsent->gold_mentions[k1]->StartOffset()
								&& sent->predict_mentions[k]->EndOffset() == newsent->gold_mentions[k1]->EndOffset()) {
							sent->predict_mentions[k]->pred_entity_id = newsent->gold_mentions[k1]->gold_entity_id;
							break;
						}
					}
					if (k1 == newsent->gold_mentions.size()) {
						cout<<sent->predict_mentions[k]->ToString() + " mention does not match!!!"<<endl;
						cout<<sent->toString()<<endl;
					}
				}
			} else {
				cout<<"sentence does not match!!!"<<endl;
			}
		}

		delete newdoc;
		newdoc = NULL;
	}
}

void CorefCorpus::LoadLDAResults(string indexfilename, string datafilename, string wordmapfile) {
	//map<string, int> wordmap;
	//ReadWordMap(wordmap, wordmapfile);

	ifstream infile(datafilename.c_str(), ios::in);
	map<int, vector<pair<int, int> > > doc_to_words;
	// Read header.
	string str;
	getline(infile, str); //"d w z t"
	// Assume not shuffled!!!
	while(getline(infile, str)) {
		vector<string> fields;
		Utils::Split(str, ' ', fields);
		if (fields.size() < 4) continue;
		int doc_id = atoi(fields[0].c_str());
		int word_id = atoi(fields[1].c_str());
		int topic_id = atoi(fields[2].c_str());
		int table_id = atoi(fields[3].c_str());
		if (doc_to_words.find(doc_id) == doc_to_words.end()) {
			vector<pair<int, int> > p;
			doc_to_words[doc_id] = p;
		}
		doc_to_words[doc_id].push_back(make_pair(word_id, topic_id));
	}
	infile.close();

	ifstream indexfile(indexfilename.c_str(), ios::in);
	while(getline(indexfile, str)) {
		vector<string> fields;
		Utils::Split(str, ' ', fields);
		int doc_id = atoi(fields[0].c_str());
		string doc_title = fields[1];
		Document *doc = documents[doc_title];
		for (int i = 2; i < fields.size(); ++i) {
			vector<string> splits;
			Utils::Split(fields[i], ':', splits);
			int mention_id = atoi(splits[0].c_str());
			int word_id = atoi(splits[1].c_str());
			assert(word_id == doc_to_words[doc_id][i-2].first);

			int cluster_id = doc_to_words[doc_id][i-2].second;
			if (cluster_id >= 0) {
				// !!! assume the mention_id aligns with the vector index of predict_mentions
				doc->predict_mentions[mention_id]->pred_entity_id = cluster_id;
			}
		}
		doc_id++;
	}
	indexfile.close();

	RebuildDocumentEntities(false);
}

// Corpus-level singletons
void CorefCorpus::FilterSingletonMentions(bool gold) {
	for (map<string, Document *>::iterator iter = topic_document.begin(); iter != topic_document.end(); ++iter) {
		Document *doc = iter->second;
		if (gold) {
			for (map<int, Entity *>::iterator iter = doc->gold_entities.begin(); iter != doc->gold_entities.end(); ++iter) {
				Entity *en = iter->second;
				int en_size = 0;
				for (int i = 0; i <en->Size(); ++i) {
					if (en->GetCorefMention(i)->valid) en_size++;
				}
				if (en_size <= 1) {
					for (int i = 0; i <en->Size(); ++i) {
						en->GetCorefMention(i)->valid = false;
					}
				}
			}
		} else {
			for (map<int, Entity *>::iterator iter = doc->predict_entities.begin(); iter != doc->predict_entities.end(); ++iter) {
				Entity *en = iter->second;
				int en_size = 0;
				for (int i = 0; i <en->Size(); ++i) {
					if (en->GetCorefMention(i)->valid) en_size++;
				}
				if (en_size <= 1) {
					for (int i = 0; i <en->Size(); ++i) {
						en->GetCorefMention(i)->valid = false;
					}
				}
			}
		}
	}
}

void CorefCorpus::FilterNotAnnotated() {
	for (map<string, Document *>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			if (doc->GetSentence(i)->gold_mentions.size() == 0) {
				for (int j = 0; j < doc->GetSentence(i)->predict_mentions.size(); ++j) {
					doc->GetSentence(i)->predict_mentions[j]->valid = false;
				}
			}
		}
	}
}

void CorefCorpus::FilterTwinless() {
	for (map<string, Document *>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		for (int j = 0; j < doc->predict_mentions.size(); ++j) {
			if (doc->predict_mentions[j]->twinless) {
				doc->predict_mentions[j]->valid = false;
			}
		}
	}
}

void CorefCorpus::FilterPronouns(bool gold) {
	for (map<string, Document *>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		if (gold) {
			for (int j = 0; j < doc->gold_mentions.size(); ++j) {
				if (doc->gold_mentions[j]->IsPronoun()) {
					doc->gold_mentions[j]->valid = false;
				}
			}
		} else {
			for (int j = 0; j < doc->predict_mentions.size(); ++j) {
				if (doc->predict_mentions[j]->IsPronoun()) {
					doc->predict_mentions[j]->valid = false;
				}
			}
		}
	}
}

string CorefCorpus::SentenceToSemevalStr(Sentence *sent, string topic_id, string part_id, bool gold) {
	map<int, Entity *> entities;
	if (gold) entities = sent->gold_entities;
	else entities = sent->predict_entities;

	vector<string> coref_tags(sent->TokenSize());
	fill(coref_tags.begin(), coref_tags.end(), "-");

	map<pair<int, int>, int> check_repeated;
	for (map<int, Entity *>::iterator iter = entities.begin(); iter != entities.end(); ++iter) {
		Entity *en = iter->second;
		for (int i = 0; i < en->Size(); ++i) {
			Mention *m = en->GetCorefMention(i);
			int entity_id = gold ? m->gold_entity_id : m->pred_entity_id;
			if (!m->valid || entity_id == -1) { // No predicton results.
				//cout<<"no prediction results!"<<endl;
				continue;
			}
			if (check_repeated.find(make_pair(m->StartOffset(), m->EndOffset()))!=check_repeated.end()) {
				cout<<"repeated!!!"<<endl;
			}
			check_repeated[make_pair(m->StartOffset(), m->EndOffset())] = 1;

			if (m->StartOffset() == m->EndOffset()) {
				if (coref_tags[m->StartOffset()] != "-") {
					coref_tags[m->StartOffset()] += "|";
				} else {
					coref_tags[m->StartOffset()] = "";
				}
				coref_tags[m->StartOffset()] += "(" + Utils::int2string(entity_id) + ")";
			} else {
				if (coref_tags[m->StartOffset()] != "-") {
					coref_tags[m->StartOffset()] += "|";
				} else {
					coref_tags[m->StartOffset()] = "";
				}
				coref_tags[m->StartOffset()] += "(" + Utils::int2string(entity_id);

				if (coref_tags[m->EndOffset()] != "-") {
					coref_tags[m->EndOffset()] += "|";
				} else {
					coref_tags[m->EndOffset()] = "";
				}
				coref_tags[m->EndOffset()] += Utils::int2string(entity_id) + ")";
			}
		}
	}

	string out_str = "";
	for (int i = 0; i < sent->TokenSize(); ++i) {
		out_str += topic_id + "\t"
				+ part_id + "\t"
				+ Utils::int2string(i) + "\t"
				+ sent->GetToken(i)->word + "\t"
				+ coref_tags[i] + "\n";
	}
	return out_str;
}

void CorefCorpus::FilterMentions(SystemOptions option) {
	// Filter singletons.
	if (option.filter_singleton) {
		FilterSingletonMentions(true);
		FilterSingletonMentions(false);
	}
	if (option.filter_pronoun) {
		FilterPronouns(true);
		FilterPronouns(false);
	}
	if (option.filter_not_annotated) {
		FilterNotAnnotated();
	}
	if (option.filter_twinless) {
		FilterTwinless(); //Filter twinless predict mentions
	}
}

void CorefCorpus::OutputSRLResults(string filename) {
	ofstream outfile(filename.c_str(), ios::out);
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		Document *doc = it->second;
		outfile<<"#begin document("<<doc->doc_id<<"); part 0"<<endl;
		outfile<<"==== Sentence level"<<endl;
		for (int i = 0; i < doc->SentNum(); ++i) {
			outfile<<i<<"\t";
			Sentence *s = doc->GetSentence(i);

			outfile<<"PARTICIPANT:";
			for (int j = 0; j < s->predict_participant_mentions.size(); ++j) {
				outfile<<s->predict_participant_mentions[j]->mention_str<<",";
			}
			outfile<<"\t";

			outfile<<"LOC:";
			for (int j = 0; j < s->predict_loc_mentions.size(); ++j) {
				outfile<<s->predict_loc_mentions[j]->mention_str<<",";
			}
			outfile<<"\t";

			outfile<<"TIME:";
			for (int j = 0; j < s->predict_time_mentions.size(); ++j) {
				outfile<<s->predict_time_mentions[j]->mention_str<<",";
			}
			outfile<<endl;
		}
		outfile<<endl;

		outfile<<"==== Mention level"<<endl;
		for (int i = 0; i < doc->predict_mentions.size(); ++i) {
			Mention *m = doc->predict_mentions[i];
			outfile<<m->sent_id<<"\t";
			outfile<<m->gold_entity_id<<"\t";
			outfile<<m->mention_str<<"\t";

			Mention *arg = m->GetArg0Mention();
			outfile<<"A0:";
			if (arg == NULL) outfile<<"NULL"<<"\t";
			else outfile<<arg->head_index<<","<<arg->mention_str<<"\t";

			arg = m->GetArg1Mention();
			outfile<<"A1:";
			if (arg == NULL) outfile<<"NULL"<<"\t";
			else outfile<<arg->head_index<<","<<arg->mention_str<<"\t";

			arg = m->GetArg2Mention();
			outfile<<"A2:";
			if (arg == NULL) outfile<<"NULL"<<"\t";
			else outfile<<arg->head_index<<","<<arg->mention_str<<"\t";

			arg = m->GetTimeMention();
			outfile<<"TIME:";
			if (arg == NULL) outfile<<"NULL"<<"\t";
			else outfile<<arg->head_index<<","<<arg->mention_str<<"\t";

			arg = m->GetLocMention();
			outfile<<"LOC:";
			if (arg == NULL) outfile<<"NULL"<<"\t";
			else outfile<<arg->head_index<<","<<arg->mention_str<<"\t";

			outfile<<endl;
		}
		outfile<<"#end document"<<endl;
	}
	outfile.close();
}

void CorefCorpus::OutputCorefResults(string filename) {
	ofstream outfile(filename.c_str(), ios::out);
	int total_pred_mentions = 0;
	int srl_pred_mentions = 0;

	for (map<string, Document*>::iterator it = topic_document.begin(); it != topic_document.end(); ++it) {
		Document *doc = it->second;
		// how many entities have identified semantic roles

		for (int i = 0; i < doc->predict_mentions.size(); ++i) {
			if (doc->predict_mentions[i]->gold_entity_id < 0) continue;
			total_pred_mentions++;

			if (doc->predict_mentions[i]->srl_args.find("A0") != doc->predict_mentions[i]->srl_args.end()) srl_pred_mentions++;
		}

		// For MUC, predict entities vs. gold entities
		outfile<<"#begin document("<<doc->doc_id<<"); part 0"<<endl;
		for (map<int, Entity*>::iterator eit = doc->gold_entities.begin(); eit != doc->gold_entities.end(); ++eit) {
			outfile<<"Gold Entity "<<eit->first<<" ";
			for (int i = 0; i < eit->second->coref_mentions.size(); ++i) {
				outfile<<eit->second->coref_mentions[i]->mention_str<<"; ";
			}
			outfile<<endl;
		}

		for (map<int, Entity*>::iterator eit = doc->predict_entities.begin(); eit != doc->predict_entities.end(); ++eit) {
			outfile<<"Predict Entity "<<eit->first<<" ";
			for (int i = 0; i < eit->second->coref_mentions.size(); ++i) {
				outfile<<eit->second->coref_mentions[i]->mention_str<<"; ";
			}
			outfile<<endl;
		}

		for (map<int, Entity*>::iterator eit = doc->gold_entities.begin(); eit != doc->gold_entities.end(); ++eit) {
			map<string, vector<int> > head_clusters;
			for (int i = 0; i < eit->second->coref_mentions.size(); ++i) {
				string head = eit->second->coref_mentions[i]->head_lemma;
				if (head_clusters.find(head) != head_clusters.end()) {
					vector<int> p;
					p.push_back(i);
					head_clusters[head] = p;
				} else {
					head_clusters[head].push_back(i);
				}
			}
			if (head_clusters.size() == 1) continue;

			outfile<<"Head Mismatch: ";
			for (map<string, vector<int> >::iterator cit = head_clusters.begin(); cit != head_clusters.end(); ++cit) {
				Mention *pm = eit->second->coref_mentions[cit->second[0]];
				Sentence *s = documents[pm->doc_id]->GetSentence(pm->sent_id);
				outfile<<pm->mention_str<<","<<s->OutputArguments(pm)<<"\t";
			}
			outfile<<endl;
		}
		outfile<<"#end document"<<endl;

		// For Bcube, per mention cluster
		/*outfile<<"#begin document("<<doc->doc_id<<"); part 0"<<endl;
		for (int i = 0; i < doc->predict_mentions.size(); ++i) {
			Mention *m = doc->predict_mentions[i];
			if (m->gold_entity_id == -1) continue;
			outfile<<"===="<<m->mention_str<<endl;
			Entity *gold_en = doc->gold_entities[m->gold_entity_id];
			outfile<<"Gold:\t";
			for (int j = 0; j < gold_en->coref_mentions.size(); ++j) {
				outfile<<gold_en->coref_mentions[j]->mention_str<<"\t";
			}
			outfile<<endl;
			Entity *pred_en = doc->predict_entities[m->pred_entity_id];
			outfile<<"Predict:\t";
			for (int j = 0; j < pred_en->coref_mentions.size(); ++j) {
				outfile<<pred_en->coref_mentions[j]->mention_str<<"\t";
			}
			outfile<<endl;
		}
		outfile<<"#end document"<<endl;
		*/
	}
	outfile.close();

	//cout<<"srl ratio: "<<(double)srl_pred_mentions/total_pred_mentions<<" ("<<srl_pred_mentions<<"/"<<total_pred_mentions<<")"<<endl;


	filename += ".WD.txt";
	outfile.open(filename.c_str(), ios::out);
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		Document *doc = it->second;
		// For MUC, predict entities vs. gold entities
		outfile<<"#begin document("<<doc->doc_id<<"); part 0"<<endl;
		for (map<int, Entity*>::iterator eit = doc->gold_entities.begin(); eit != doc->gold_entities.end(); ++eit) {
			outfile<<"Gold Entity "<<eit->first<<" ";
			for (int i = 0; i < eit->second->coref_mentions.size(); ++i) {
				outfile<<eit->second->coref_mentions[i]->mention_str<<"; ";
			}
			outfile<<endl;
		}
		for (map<int, Entity*>::iterator eit = doc->predict_entities.begin(); eit != doc->predict_entities.end(); ++eit) {
			outfile<<"Predict Entity "<<eit->first<<" ";
			for (int i = 0; i < eit->second->coref_mentions.size(); ++i) {
				outfile<<eit->second->coref_mentions[i]->mention_str<<"; ";
			}
			outfile<<endl;
		}
		outfile<<"#end document"<<endl;
	}
	outfile.close();
}

void CorefCorpus::OutputWDSemEvalFiles(string filename, bool gold) {
	ofstream outfile(filename.c_str(), ios::out);

	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		string topic_id = iter->first;
		outfile<<"#begin document ("<<topic_id<<"); part 000"<<endl;

		Document *doc = iter->second;

		string doc_str = "";
		for (int i = 0; i < doc->SentNum(); ++i) {
			doc_str += SentenceToSemevalStr(doc->GetSentence(i), topic_id, "0", gold);
			doc_str += "\n";
		}
		outfile<<doc_str;

		outfile<<"#end document"<<endl;
	}
	outfile.close();
}

void CorefCorpus::OutputCDSemEvalFiles(string filename, bool gold) {
	ofstream outfile(filename.c_str(), ios::out);
	for (map<string, Document*>::iterator iter = topic_document.begin();
			iter != topic_document.end(); ++iter) {
		string topic_id = iter->first;
		outfile<<"#begin document ("<<topic_id<<"); part 000"<<endl;

		Document *doc = iter->second;
		string doc_str = "";
		for (int i = 0; i < doc->SentNum(); ++i) {
			doc_str += SentenceToSemevalStr(doc->GetSentence(i), topic_id, "0", gold);
			doc_str += "\n";
		}
		outfile<<doc_str;

		outfile<<"#end document"<<endl;
	}
	outfile.close();
}

void CorefCorpus::LoadWDSemEvalFiles(string filename) {
	ifstream infile(filename.c_str(), ios::in);
	vector<Document *> read_docs;
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		read_docs.push_back(iter->second);
	}

//	string str;
//	vector<string> lines;
//	while(getline(infile, str)) {
//		if (str.find("#begin document") != string::npos) {
//			lines.clear();
//		} else if (str.find("#end document") != string::npos) {
//			// parse document
//		}
//	}
	infile.close();
}

void CorefCorpus::LoadDDCRPResults(string path) {
	for (map<string, vector<Document*> >::iterator it = topic_to_documents.begin();
				it != topic_to_documents.end(); ++it) {
		string filename = path + "/" + it->first + "/00700-word-assignments.dat";
		ifstream infile(filename.c_str(), ios::in);
		map<int, vector<pair<int, int> > > doc_to_words;
		// Read header.
		string str;
		getline(infile, str); //"d w z t"
		// Assume not shuffled!!!
		while(getline(infile, str)) {
			vector<string> fields;
			Utils::Split(str, ' ', fields);
			if (fields.size() < 4) continue;
			int doc_id = atoi(fields[0].c_str());
			int word_id = atoi(fields[1].c_str());
			int topic_id = atoi(fields[2].c_str());
			int table_id = atoi(fields[3].c_str());
			if (doc_to_words.find(doc_id) == doc_to_words.end()) {
				vector<pair<int, int> > p;
				doc_to_words[doc_id] = p;
			}
			doc_to_words[doc_id].push_back(make_pair(word_id, topic_id));
		}
		infile.close();

		if (doc_to_words.size() == 0) continue;

		int doc_id = 0;
		for (int i = 0; i < it->second.size(); ++i) {
			Document *doc = it->second[i];
			int length = doc->predict_mentions.size();
			vector<pair<int, int> > p = doc_to_words[doc_id];
			assert((int)p.size() == length);
			for (int n = 0; n < length; n++) {
				Mention *m = doc->predict_mentions[n];
				m->pred_entity_id = p[n].second;
			}
			doc_id++;
		}
	}

	RebuildDocumentEntities(false);
}

void CorefCorpus::OutputClusterResults(string filename) {
	ofstream outfile(filename.c_str(), ios::out);
/*	outfile<<"Gold entities: "<<endl;
	for (map<int, Entity*>::iterator iter = gold_entities.begin(); iter != gold_entities.end(); ++iter) {
		Entity *en = iter->second;
		outfile<<"Entity "<<iter->first<<" : "<<en->ToString()<<endl;
		outfile<<endl;
	}
	outfile<<endl;
	outfile<<"Predicted entities: "<<endl;
	for (map<int, Entity*>::iterator iter = predict_entities.begin(); iter != predict_entities.end(); ++iter) {
		Entity *en = iter->second;
		outfile<<"Entity "<<iter->first<<" : "<<en->ToString()<<endl;
		outfile<<endl;
	}
*/
	for (map<string, Document *>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;

		outfile<<"#Document "<<iter->first<<endl;
		outfile<<"Gold_entities: ";
		for (map<int, Entity *>::iterator it = doc->gold_entities.begin(); it != doc->gold_entities.end(); ++it) {
			Entity *en = it->second;
			if (en->EntityID() < 0) continue;
			outfile<<it->first<<"   ### "<<en->ToString()<<endl;
		}
		outfile<<endl;
		outfile<<"Predict_entities: ";
		for (map<int, Entity *>::iterator it = doc->predict_entities.begin(); it != doc->predict_entities.end(); ++it) {
			Entity *en = it->second;
			if (en->EntityID() < 0) continue;
			//outfile<<it->first<<"   ### "<<en->ToString()<<endl;
			// Only output mentions in annotated sentences.
			outfile<<it->first<<"   ### ";
			for (int i = 0; i < en->Size(); ++i) {
				Mention *m = en->GetCorefMention(i);
				if (doc->GetSentence(m->SentenceID())->gold_mentions.size() == 0) continue;
				outfile<<m->ToString()<<", ";
			}
			outfile<<endl;
		}
		outfile<<endl;
		outfile<<endl;
	}
	outfile.close();
}

void CorefCorpus::OutputVerbalMentions(string filename) {
	// output verbal mention spans.
	ofstream outfile(filename.c_str(), ios::out);
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		Document *doc = iter->second;
		outfile<<"#begin document ("<<doc->DocID()<<")"<<endl;
		for (int i = 0; i < doc->SentNum(); ++i) {
			vector<Mention*> mentions;
			//doc->GetSentence(i)->ExtractVerbalMentions(mentions);
			doc->GetSentence(i)->ExtractGoldVerbalMentions(mentions);
			for (int j = 0; j < mentions.size(); ++j) {
				outfile<<i<<"\t"<<mentions[j]->StartOffset()<<"\t"<<mentions[j]->EndOffset()<<"\t"
						<<doc->GetSentence(i)->GetSpanStr(mentions[j]->StartOffset(), mentions[j]->EndOffset())<<endl;
			}
			for (int j = 0; j < mentions.size(); ++j) {
				delete mentions[j];
			}
		}
		outfile<<"#end document"<<endl;
	}
	outfile.close();
}

void CorefCorpus::MakeSwirlInput(string path) {
	for (map<string, Document*>::iterator iter = documents.begin(); iter != documents.end(); ++iter) {
		string str = "";
		Document *doc = iter->second;
		string filename = path + "/SWIRL_INPUT." + doc->DocID() + ".txt";
		ofstream outfile(filename.c_str(), ios::out);
		for (int i = 0; i < doc->SentNum(); ++i) {
			str += "3";
			for (int j = 0; j < doc->GetSentence(i)->TokenSize(); ++j) {
				str += " " + doc->GetSentence(i)->GetToken(j)->word;
			}
			str += "\n";
		}
		outfile<<str;
		outfile.close();
	}
}

bool CorefCorpus::ParseSrlArguments(string sent_str, Sentence *sent) {
	vector<string> lines;
	Utils::Split(sent_str, '\n', lines);
	if (lines.size() != sent->TokenSize()) return false;

	// pred_index => map<span, role>
	map<int, map<pair<int, int>, string> > arg_tags; // arguments for each predicate, intpair->role
	vector<int> verbIdx;

	vector<vector<string> > data;
	for(int i = 0; i < lines.size(); ++i) {
		vector<string> parts;
		Utils::Split(lines[i], '\t', parts); //\t\t??
		data.push_back(parts);
	}

	int ntokens = data.size();
    for(int col = 1 ; col < data[0].size() ; col++) {
      string argTag = "";
      int argBeginIdx = -1;
      int argEndIdx = -1;
      for(int idx = 0 ; idx < ntokens ; idx++) {
        string str = data[idx][col];
        if(col==1) {
        	// predicate
        	if(str != "-") {
        		verbIdx.push_back(idx);
        		map<pair<int, int>, string> p;
        		arg_tags[idx] = p;
        	}
        } else { //argument
          if(str == "O") continue;
          if(str[0] == 'B') {
            argTag = str.substr(2,str.size()-2);
            argBeginIdx = idx;
          }
          if(str[0] == 'E') {
            argEndIdx = idx;
            arg_tags[verbIdx[col-2]][make_pair(argBeginIdx, argEndIdx)] = argTag;
          }
        }
      }
    }

	// Swirl format
	/*
	int npredicates = data[0].size();
	int ntokens = data.size();
    for(int col = 0 ; col < data[0].size() ; col++) {
      string argTag = "";
      int argBeginIdx = -1;
      int argEndIdx = -1;
      for(int idx = 0 ; idx < ntokens ; idx++) {
        string str = data[idx][col];
        if(col==0) {
          if(str != "-") {
        	  verbIdx.push_back(idx);
        	  map<pair<int, int>, string> p;
        	  arg_tags[idx] = p;
          }
        } else { //argument
          if(str == "*") continue;
          if(str[0] == '(') {
            argTag = (str[str.size()-1] == ')')? str.substr(1,str.size()-3) : str.substr(1,str.size()-2);
            argBeginIdx = idx;
          }
          if(str[str.size()-1] == ')') {
            argEndIdx = idx+1;
            arg_tags[verbIdx[col-1]][make_pair(argBeginIdx, argEndIdx)] = argTag;
          }
        }
      }
    }*/

    sent->BuildSrlArguments(arg_tags);
    return true;
}
bool CorefCorpus::ParseSwirlSrlArguments(string sent_str, Sentence *sent) {
	vector<string> lines;
	Utils::Split(sent_str, '\n', lines);
	if (lines.size() != sent->TokenSize()) return false;

	// pred_index => map<span, role>
	map<int, map<pair<int, int>, string> > arg_tags; // arguments for each predicate, intpair->role
	vector<int> verbIdx;

	vector<vector<string> > data;
	for(int i = 0; i < lines.size(); ++i) {
		vector<string> parts;
		Utils::Split(lines[i], '\t', parts); //\t\t??
		data.push_back(parts);
	}

	// Swirl format
	int npredicates = data[0].size();
	int ntokens = data.size();
    for(int col = 0 ; col < data[0].size() ; col++) {
      string argTag = "";
      int argBeginIdx = -1;
      int argEndIdx = -1;
      for(int idx = 0 ; idx < ntokens ; idx++) {
        string str = data[idx][col];
        if(col==0) {
          if(str != "-") {
        	  verbIdx.push_back(idx);
        	  map<pair<int, int>, string> p;
        	  arg_tags[idx] = p;
          }
        } else { //argument
          if(str == "*") continue;
          if(str[0] == '(') {
            argTag = (str[str.size()-1] == ')')? str.substr(1,str.size()-3) : str.substr(1,str.size()-2);
            argBeginIdx = idx;
          }
          if(str[str.size()-1] == ')') {
            argEndIdx = idx;
            arg_tags[verbIdx[col-1]][make_pair(argBeginIdx, argEndIdx)] = argTag;
          }
        }
      }
    }

    sent->BuildSrlArguments(arg_tags);
    return true;
}

void CorefCorpus::OutputSwirlInfo(Logger &logger) {
	std::stringstream ss;
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		Document *doc = it->second;
		ss<<"Document "<<it->first<<endl;
		for (int i = 0; i < doc->predict_mentions.size(); ++i) {
			Mention *m = doc->predict_mentions[i];
			if (m->srl_args.size() == 0) continue;
			ss<<doc->GetSentence(m->SentenceID())->GetMentionContext(m)<<"->";
			/*for (map<string, Mention*>::iterator arg_it = m->srl_args.begin(); arg_it != m->srl_args.end(); ++arg_it) {
				ss<<arg_it->first<<":"<<arg_it->second->mention_str<<","
						<<doc->GetSentence(m->SentenceID())->GetMentionContext(arg_it->second)<<endl;
			}*/
			ss<<endl;
		}
	}
	logger.Write(ss.str());
}

void CorefCorpus::ReadSwirlOutput(string path) {
	vector<string> filenames;
	Utils::ReadFileInDir(path, "SWIRL_OUTPUT", filenames);
	for (int i = 0; i < filenames.size(); ++i) {
		ifstream infile(filenames[i].c_str(), ios::in);
		int start = filenames[i].find_first_of(".");
		int end = filenames[i].find_last_of(".");
		string doc_id = filenames[i].substr(start+1, end-start-1);
		if (documents.find(doc_id) == documents.end()) continue;

		Document *doc = documents[doc_id];
		int sent_id = 0;
		string str;
		string sent_str = "";
		while(getline(infile, str)) {
			if (str == "") {
				if (!ParseSwirlSrlArguments(sent_str, doc->GetSentence(sent_id))) {
					//cout<<"srl mismatch "<<doc->doc_id<<" "<<sent_id<<endl;
					break;
				}
				sent_str = "";
				sent_id++;
			} else {
				sent_str += str + "\n";
			}
		}
		infile.close();
	}
}

void CorefCorpus::RunSwirl(string path, string swirl_home) {
	string swirlInput = "SWIRL_INPUT";
	string swirlOutput = "SWIRL_OUPUT";
	vector<string> files;
	Utils::ReadFileInDir(path, swirlInput, files);

	string swirlCommand = swirl_home + "/src/bin/swirl_parse_classify";
	string swirlSRLModel = swirl_home + "/model_swirl";
	string swirlCharniakModel = swirl_home + "/model_charniak";

	for(int i = 0; i < files.size(); ++i) {
	    string inputFile = files[i];
	    int index = inputFile.find(swirlInput);
	    string outputFile = inputFile.replace(index, swirlInput.size(), swirlOutput);
		string command = swirlCommand+" "+swirlSRLModel+" "+swirlCharniakModel+" "+inputFile + " > " + outputFile;
		system(command.c_str());
	}
}

void CorefCorpus::OutputMultinomial(string filename, bool gold) {
	ofstream outfile(filename.c_str(), ios::out);
	for (map<string, Document*>::iterator it = topic_document.begin(); it != topic_document.end(); ++it) {
		string topic = it->first;
		Document *doc = it->second;
		if (gold) {
			for (map<int, Entity *>::iterator it1 = doc->gold_entities.begin(); it1 != doc->gold_entities.end(); ++it1) {
				map<string, int> word2count;
				Entity *en = it1->second;
				for (int j = 0; j < en->Size(); ++j) {
					string word = en->GetCorefMention(j)->head_span;
					if (word2count.find(word) == word2count.end()) {
						word2count[word] = 1;
					} else {
						word2count[word] += 1;
					}
				}
				outfile<<it->first<<" "<<en->EntityTypeStr()<<" "<<it1->first<<" "<<en->Size()<<" ";
				for (map<string, int>::iterator it = word2count.begin(); it != word2count.end(); ++it) {
					outfile<<it->first<<":"<<it->second<<" ";
				}
				outfile<<endl;
			}
		} else {
			for (map<int, Entity *>::iterator it1 = doc->predict_entities.begin(); it1 != doc->predict_entities.end(); ++it1) {
				map<string, int> word2count;
				Entity *en = it1->second;
				for (int j = 0; j < en->Size(); ++j) {
					string word = en->GetCorefMention(j)->head_span;
					if (word2count.find(word) == word2count.end()) {
						word2count[word] = 1;
					} else {
						word2count[word] += 1;
					}
				}
				outfile<<it->first<<" "<<en->EntityTypeStr()<<" "<<it1->first<<" "<<en->Size()<<" ";
				for (map<string, int>::iterator it = word2count.begin(); it != word2count.end(); ++it) {
					outfile<<it->first<<":"<<it->second<<" ";
				}
				outfile<<endl;
			}
		}
	}
	outfile.close();
}

/*void CorefCorpus::loadNonanarphoricInfo(string filename) {
	nonanarphoric_local_prob.clear();
	nonanarphoric_global_prob.clear();
	ifstream infile(filename.c_str(), ios::in);
	string str;
	while(getline(infile, str)) {
		vector<string> fields;
		Utils::Split(str, '\t', fields);
		double v = atof(fields[1].c_str());
		if (fields[0] == "PROPER") {
			nonanarphoric_local_prob[PROPER] = v;
		} else if (fields[0] == "PRONOMINAL") {
			nonanarphoric_local_prob[PRONOMINAL] = v;
		} else if (fields[0] == "NOMINAL") {
			nonanarphoric_local_prob[NOMINAL] = v;
		}

		v = atof(fields[2].c_str());
		if (fields[0] == "PROPER") {
			nonanarphoric_global_prob[PROPER] = v;
		} else if (fields[0] == "PRONOMINAL") {
			nonanarphoric_global_prob[PRONOMINAL] = v;
		} else if (fields[0] == "NOMINAL") {
			nonanarphoric_global_prob[NOMINAL] = v;
		}
	}
	infile.close();
}
*/

/*void CorefCorpus::loadMentionDistance(string filename) {
	ifstream infile(filename.c_str(), ios::in);
	string str;
	while(getline(infile, str)) {
		int i = str.find("\t");
		string field_0 = str.substr(0, i);
		int j = str.rfind("\t");
		string field_1 = str.substr(i+1, j-1-i);
		string field_2 = str.substr(j+1);

		i = field_0.find(",");
		int mid = atoi(field_0.substr(0, i).c_str());
		Mention *m = mention_dict[mid];

		if (field_1 != "") {
			vector<string> splits;
			Utils::Split(field_1, ' ', splits);
			for (int k = 0; k < splits.size(); ++k) {
				i = splits[k].find(":");
				int ant_mid = atoi(splits[k].substr(0, i).c_str());
				double score = atof(splits[k].substr(i+1).c_str());
				Mention *ant_m = mention_dict[ant_mid];
				AntecedentLink link(ant_m, score);
				m->local_antecedents.push_back(link);
			}
			//m->transform_local_prob();
		}

		if (field_2 != "") {
			vector<string> splits;
			Utils::Split(field_2, ' ', splits);
			for (int k = 0; k < splits.size(); ++k) {
				i = splits[k].find(":");
				int ant_mid = atoi(splits[k].substr(0, i).c_str());
				double score = atof(splits[k].substr(i+1).c_str());
				Mention *ant_m = mention_dict[ant_mid];
				AntecedentLink link(ant_m, score);
				m->global_antecedents.push_back(link);
			}
			//m->transform_global_prob();
		}
	}
	infile.close();
}
*/

void CorefCorpus::SetupMentionDict() {
	mention_dict.clear();
	for (map<string, vector<Document*> >::iterator it = topic_to_documents.begin(); it != topic_to_documents.end(); ++it) {
		for (int d = 0; d < it->second.size(); ++d) {
			// within document mention
			Document *doc = it->second[d];
			for (int i = 0; i < doc->predict_mentions.size(); ++i) {
				mention_dict[doc->predict_mentions[i]->mention_id] = doc->predict_mentions[i];
			}
		}
	}
}

void CorefCorpus::SetupMentionDistance() {
	// within document: link in sentence order
	// across document : only link to the first sentence
	for (map<string, vector<Document*> >::iterator it = topic_to_documents.begin(); it != topic_to_documents.end(); ++it) {
		for (int d = 0; d < it->second.size(); ++d) {
			// within document mention
			Document *doc = it->second[d];
			for (int i = 0; i < doc->predict_mentions.size(); ++i) {
				Mention *m = doc->predict_mentions[i];
				MentionType type = m->mention_type;
				m->non_anarphoric_local_prob = 0.1;
				m->non_anarphoric_global_prob = 0.1;

				for (int j = 0; j < i; ++j) {
					double score = m->Similarity(doc->predict_mentions[j]);
					if (score > 0) {
						AntecedentLink link(doc->predict_mentions[j], score);
						m->local_antecedents.push_back(link);
					}
				}

				for (int d1 = 0; d1 < d; ++d1) { // documents are also ordered!!!
					if (d1 == d) continue;
					Document *ant_doc = it->second[d1];
					// only consider the first sentence!!!
					for (int j = 0; j < ant_doc->GetSentence(0)->predict_mentions.size(); ++j) {
						double score = m->Similarity(ant_doc->GetSentence(0)->predict_mentions[j]);
						if (score > 0) {
							AntecedentLink link(ant_doc->GetSentence(0)->predict_mentions[j], score);
							m->global_antecedents.push_back(link);
						}
					}
				}
			}
		}
	}
}

void CorefCorpus::loadMentionID(string filename) {
	// clear the mention id field
	max_mention_id = 0;
	mention_dict.clear();

	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		for (int i = 0; i < it->second->predict_mentions.size(); ++i) {
			it->second->predict_mentions[i]->mention_id = -1;
		}
		for (int i = 0; i < it->second->gold_mentions.size(); ++i) {
			it->second->gold_mentions[i]->mention_id = -1;
		}
	}

	ifstream infile(filename.c_str(), ios::in);
	string str;
	while(getline(infile, str)) {
		vector<string> fields;
		Utils::Split(str, '\t', fields);
		vector<string> splits;
		Utils::Split(fields[0], ',', splits);
		string docid = splits[0];
		int sentid = atoi(splits[1].c_str());
		int start = atoi(splits[2].c_str());
		int end = atoi(splits[3].c_str()) - 1;
		Document * doc = documents[docid];
		Mention* m = doc->sentences[sentid]->FindPredMentionBySpan(start, end);
		if (m == NULL) {
			cout<<"does not find mention!!!"<<endl;
		}
		m->mention_id = atoi(fields[2].c_str());

		mention_dict[m->mention_id] = m;

		if (m->mention_id > max_mention_id) {
			max_mention_id = m->mention_id + 1;
		}
	}
	infile.close();

	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		for (int i = 0; i < it->second->predict_mentions.size(); ++i) {
			if (it->second->predict_mentions[i]->mention_id == -1) {
				cout<<"Miss predict mentions!!!"<<endl;
			}
		}
		for (int i = 0; i < it->second->gold_mentions.size(); ++i) {
			if (it->second->gold_mentions[i]->mention_id == -1) {
				cout<<"Miss gold mentions!!!"<<endl;
				it->second->gold_mentions[i]->mention_id = max_mention_id++;
				mention_dict[it->second->gold_mentions[i]->mention_id] = it->second->gold_mentions[i];
			}
		}
	}
}

void CorefCorpus::FilterPredictMentions() {
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		Document *doc = it->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *sent = doc->GetSentence(i);
			set<int> headset;
//			for (vector<Mention*>::iterator it=sent->predict_mentions.begin();
//					it!=sent->predict_mentions.end();) {
//				Mention *men = *it;
//				if (headset.find(men->head_index) != headset.end()) {
//					it = sent->predict_mentions.erase(it);
//				} else {
//					++it;
//					headset.insert(men->head_index);
//				}
//			}

			headset.clear();
			for (vector<Mention*>::iterator it=sent->predict_participant_mentions.begin();
					it!=sent->predict_participant_mentions.end();) {
				Mention *men = *it;
				if (headset.find(men->head_index) != headset.end()) {
					it = sent->predict_participant_mentions.erase(it);
				} else {
					++it;
					headset.insert(men->head_index);
				}
			}

			headset.clear();
			for (vector<Mention*>::iterator it=sent->predict_time_mentions.begin();
					it!=sent->predict_time_mentions.end();) {
				Mention *men = *it;
				if (headset.find(men->head_index) != headset.end()) {
					it = sent->predict_time_mentions.erase(it);
				} else {
					++it;
					headset.insert(men->head_index);
				}
			}

			headset.clear();
			for (vector<Mention*>::iterator it=sent->predict_loc_mentions.begin();
					it!=sent->predict_loc_mentions.end();) {
				Mention *men = *it;
				if (headset.find(men->head_index) != headset.end()) {
					it = sent->predict_loc_mentions.erase(it);
				} else {
					++it;
					headset.insert(men->head_index);
				}
			}
		}
	}
}

void CorefCorpus::OutputStats() {
	cout<<"#documents: "<<documents.size()<<endl;
	int sentno = 0;
	int eventno = 0;
	int WDchain = 0;
	int CDchain = 0;
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		Document *doc = it->second;
		//cout<<doc->doc_id<<endl;
		sentno += doc->SentNum();
		eventno += doc->gold_mentions.size();
		WDchain += doc->gold_entities.size();
	}
	cout<<"#sentences: "<<sentno<<endl;
	cout<<"#event mentions: "<<eventno<<endl;
	cout<<"#wd chain: "<<WDchain<<endl;
	for (map<string, Document*>::iterator it = topic_document.begin(); it != topic_document.end(); ++it) {
		Document *doc = it->second;
		CDchain += doc->gold_entities.size();
	}
	cout<<"#cd chain: "<<CDchain<<endl;
}

double VecDistance(map<int, double> &fvec1, map<int, double> &fvec2) {
	double dist = 0.0;
	for (map<int, double>::iterator it = fvec1.begin(); it != fvec1.end(); ++it) {
		if (fvec2.find(it->first) != fvec2.end()) {
			dist += it->second * fvec2[it->first];
		}
	}
	return dist;
}

void CorefCorpus::DocumentClustering() {
	// Compute pairwise document distance
	map<string, int> feature_dict;
	vector<Document*> docs;
	for (map<string, Document*>::iterator it = documents.begin(); it != documents.end(); ++it) {
		it->second->BuildDocumentVec(feature_dict);
		it->second->doc_index = docs.size();
		docs.push_back(it->second);
	}

	vector<map<int, double> > pairwise_doc_dist;
	for (int i = 0; i < docs.size(); ++i) {
		Document *doc = docs[i];
		map<int, double> p;
		pairwise_doc_dist.push_back(p);

		for (int j = 0; j < i; ++j) {
			Document *ant_doc = docs[j];
			double dist = VecDistance(doc->doc_fvec, ant_doc->doc_fvec);
			pairwise_doc_dist[i][j] = (double)dist/(doc->doc_norm * ant_doc->doc_norm);
		}
	}

	// set label
	int err = 0;
	int acc = 0;
	int count = 0;

	double threshold = 0.0;
	int topK = docs.size();
	for (int i = 1; i < docs.size(); ++i) {
		vector<pair<int, double> > neighbors_info = Utils::sortMap(pairwise_doc_dist[i], false);
		int best_ant = neighbors_info[0].first;
		double best_score = neighbors_info[0].second;
		if (best_score > threshold) {
			docs[i]->predict_doc_ant = docs[best_ant]->doc_id;
		}

		bool found = false;
		for (int j = 0; j < min((int)neighbors_info.size(), topK); ++j) {
			int ant_doc = neighbors_info[j].first;
			if (neighbors_info[j].second > threshold) {
				docs[i]->nearest_neighbors.push_back(docs[ant_doc]);
				docs[i]->doc_distance[docs[ant_doc]->doc_id] = neighbors_info[j].second;
			}

			if (docs[i]->topic == docs[ant_doc]->topic) {
				found = true;
			}
		}
		if (found) acc++;

		// For debugging
		if (best_score > threshold && docs[i]->topic != docs[best_ant]->topic) {
			err++;
			//cout<<"MissLink: "<<docs[i]->doc_id<<" to "<<docs[best_ant]->doc_id<<" "<<best_score<<endl;
		}

		count ++;
	}
	//cout<<"Total error: "<<err<<" out of "<<count<<endl;
	//cout<<"Total acc: "<<acc<<" out of "<<count<<endl;

	int pos_count = 0;
	int pos_err = 0;
	int neg_count = 0;
	int neg_err = 0;
	threshold = 0.4;
	for (int i = 1; i < docs.size(); ++i) {
		for (int j = 0; j < i; ++j) {
			if (docs[i]->topic == docs[j]->topic) {
				pos_count++;
				if (docs[i]->doc_distance[docs[j]->doc_id] < threshold) pos_err++;
			}
			if (docs[i]->topic != docs[j]->topic) {
				neg_count++;
				if (docs[i]->doc_distance[docs[j]->doc_id] > threshold) neg_err++;
			}
		}
	}

	cout<<"pos_error: "<<(double)pos_err/pos_count<<" neg_error: "<<(double)neg_err/neg_count<<endl;

	// cross-document clustering
/*	PairwiseClustering cl;
	cl.threshold = threshold;
	cl.maxNpID = docs.size();
	for (int i = 0; i < cl.maxNpID; ++i) {
		for (int j = 0; j < i; ++j) {
			double dist = pairwise_doc_dist[i][j];
			if (dist > 0) {
				cl.setWeight(i,j,dist);
			}
		}
	}

	cl.Clustering();

	map<int, vector<int> > clusters;
	predict_document_clusters.clear();
	int max_cid = 0;
	for (int i = 0; i < cl.maxNpID; ++i) {
		int cid = cl.cluster_assignments[i];
		if (cid > max_cid) max_cid = cid;

		docs[i]->predict_topic = cid;

		if (clusters.find(cid) == clusters.end()) {
			vector<int> p;
			clusters[cid] = p;
			vector<Document*> tmp;
			predict_document_clusters[Utils::int2string(cid)] = tmp;
		}
		clusters[cid].push_back(i);
		predict_document_clusters[Utils::int2string(cid)].push_back(docs[i]);
	}

	cout<<"finish clustering"<<endl;

	ofstream outfile("doc.clusters.txt", ios::out);
	for (map<int, vector<int> >::iterator it = clusters.begin(); it != clusters.end(); ++it) {
		outfile<<"Cluster "<<it->first<<endl;
		for (int i = 0; i < it->second.size(); ++i) {
			outfile<<docs[it->second[i]]->doc_id<<endl;
		}
		outfile<<endl;
	}
	outfile.close();
	*/
}
