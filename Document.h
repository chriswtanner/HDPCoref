/*
 * Document.h
 *
 *  Created on: Mar 19, 2013
 *      Author: bishan
 */

#ifndef DOCUMENT_H_
#define DOCUMENT_H_

#include <map>
#include "Entity.h"
#include "Constraints.h"

#include "./Parsing/PennTree.h"
#include "./Parsing/GraphNode.h"

using namespace std;

class Token {
public:
	Token() {
		word_id = 0;
		word = "";
		lowercase = "";
		pos = "";
		lemma = "";
		second_lemma = "";
		lemma_pos = "";
	}
	~Token() {}
public:
	int word_id;
	string word;
	string lowercase;
	string pos;
	string lemma;

	string second_lemma;
	string lemma_pos;
};

class Sentence
{
public:
	Sentence(void) : penntree(0), G(0), valid(true), sent_id(-1), srl_participant_vec(NULL), srl_time_vec(NULL), srl_loc_vec(NULL) {}
	~Sentence(void);

	void AddEventMention(Mention *m);
	void AddParticipantMention(Mention *m);
	void AddTimeMention(Mention *m);
	void AddLocMention(Mention *m);

	string OutputArguments(Mention *m);
	bool Overlap(Sentence *sent);

	void GetContextWords(int start, int end, vector<string> &words) {
		words.clear();
		for (int i = max(0, start); i <= min(end, TokenSize()-1); ++i) {
			words.push_back(tokens[i]->lemma);
		}
	}

	void SetSentenceID(int i) {sent_id = i;}
	int SentenceID() {return sent_id;}
	void SetDocID(string str) {doc_id = str;}
	string DocID() {return doc_id;}
	void AddToken(Token *t) { tokens.push_back(t); }
	int TokenSize() { return tokens.size();}
	Token *GetToken(int i) { return tokens[i]; }

	int MentionSize(bool gold) {
		if (gold) return gold_mentions.size();
		else return predict_mentions.size();
	}

	int EntitySize(bool gold) {
		if (gold) return gold_entities.size();
		else return predict_entities.size();
	}

	void SortMentionsInLinearOrder(bool gold);
	void BuildSentenceEntities(bool gold);

	void buildPennTree(string parse_tree);
	void buildDepGraph(string dep_graph);
	void buildDiscourseTree(string discourse_tree);

	bool similarity_check(Sentence *sent);

	string GetParseTag(Mention *m);
	int GetHeadIndex(Mention *m);
	string GetHeadWord(Mention *m);
	string GetHeadPOS(Mention *m);

	string toString();

	// for word2vec
	string GetSpanLemmaStr(int start, int end);
	string GetOrigSpanStr(int start, int end);

	string GetSpanWordStr(int start, int end);

	string GetSpanLowerCase(int start, int end);

	string GetSpanStr(int start, int end);
	string GetSpanContext(int start, int end);

	void ParseMention(vector<string> fields, Mention *m);
	void ParseMentionFeatures(vector<string> fields, Mention *m);
	string GetMentionStr(Mention *m);
	string GetMentionContext(Mention *m);
	string GetMentionSRLArguments(Mention *m);

	Mention* FetchGoldMention(int start_offset, int end_offset);
	Mention* FetchPredictMention(int start, int end);

	void ExtractVerbalMentions(vector<Mention*> &mentions);
	void ExtractGoldVerbalMentions(vector<Mention*> &mentions);

	void FindEventArguments(int index, vector<Argument> &args);

	void GetDepArguments(int word, vector<Argument> &args);
	void BuildSrlArguments(map<int, map<pair<int, int>, string> > &args);

	Mention *FindPredMentionBySpan(int start, int end);
	Mention *FindMentionByHead(int headindex);
	int FindHeadIndexOfArg(Span argSpan);

	void AddCoNLLTag(int start, int end, string coref_id, vector<string> &coref_tags);
	void GetCoNLLTags(vector<string> &coref_tags);
	void GetCoNLLPredictMentions(vector<string> & coref_tags);

	bool Equal(Sentence *sent);

	int FindVerbalHead(Span argSpan);

	void  SetHeadForPredictMentions(map<int, int> &head_dict);
	void  SetHeadIndex(Mention *m, int hindex);

public:
	int *srl_participant_vec;
	int *srl_time_vec;
	int *srl_loc_vec;

	vector<Mention *> gold_mentions;
	vector<Mention *> gold_participant_mentions;
	vector<Mention *> gold_time_mentions;
	vector<Mention *> gold_loc_mentions;

	vector<Mention *> predict_mentions;
	vector<Mention *> predict_participant_mentions;
	vector<Mention *> predict_time_mentions;
	vector<Mention *> predict_loc_mentions;
	//vector<Mention *> all_predicted_mentions;

	// Sentence-level coref entities not document-level!
	map<int, Entity *> gold_entities;
	map<int, Entity *> gold_participant_entities;
	map<int, Entity *> gold_time_entities;
	map<int, Entity *> gold_loc_entities;

	map<int, Entity *> predict_entities;
	map<int, Entity *> predict_participant_entities;
	map<int, Entity *> predict_time_entities;
	map<int, Entity *> predict_loc_entities;

	map<string, Sentence*> gold_cand_sents;
	map<string, Sentence*> cand_sents;

	string sent_key; //local doc_id + sent_id

	string conll_str;
	vector<Span> phrases;
	vector<Token*> tokens;

	string doc_id;
	int sent_id;
	string sent; //original sentence

	bool valid;


private:
    PennTree *penntree;
	DependencyGraph *G;
};

class Document{
public:
	Document();
	virtual ~Document();

	void BuildSimpleDocumentVec(map<string, int> &features);

	int SentNum() { return sentences.size(); }
	void SetTopicID(string id) { topic_id = id; }
    string TopicID() { return topic_id; }
	void SetDocID(string id) { doc_id = id; }
	string DocID() { return doc_id; }

	void GroupSentencesByDist();
	void GroupSentences(map<string, int> &sentence2idx, vector<vector<double> > &sentence_vecs);

	void BuildCorefClusters(vector<string> coref_tags, Sentence *sent, EntityType type);
	void ReadAnnotation(vector<string> coref_tags, int sent_id);

	void AddSentence(Sentence *sent) { sentences.push_back(sent); }
	void AddGoldEntity(int en_id, Entity *en) { gold_entities[en_id] = en; }
	void AddGoldMention(Mention *m) { gold_mentions.push_back(m); }

	// Collect entities and mentions (in the order they appear in text).
	void BuildDocumentMentions(bool gold);
	void BuildDocumentEntities(bool gold);

	Sentence *GetSentence(int i) { return sentences[i]; }

	bool RulebasedCoreferent(Entity *e, Entity *ant,
			Mention *m, Mention *ant_m, map<string, double> &distance);
	void GetOrderedAntecedents(
		  Sentence *ant_sent,
		  Sentence *m_sent,
		  Mention *m,
		  map<int, Entity *> &coref_cluster,
		  vector <Mention*> &antecedent_mentions);
	void SortMentionsForPronoun(vector<Mention *> l, Mention *m, bool sameSentence);

	void FreeEntities(map<int, Entity*> &entities);

	void BuildTFIDFMentionVec(Mention *m, map<int, float> &idf_map);

	void BuildDocumentVec(map<string, int> &features);

	void LoadEventCorefClusters(vector<string> coref_tags, Sentence *sent);

public:
	vector<Mention *> gold_mentions;
	vector<Mention *> gold_participant_mentions;
	vector<Mention *> gold_time_mentions;
	vector<Mention *> gold_loc_mentions;

	vector<Mention *> predict_mentions;
	vector<Mention *> predict_participant_mentions;
	vector<Mention *> predict_time_mentions;
	vector<Mention *> predict_loc_mentions;

	map<int, Entity *> gold_entities;
	map<int, Entity *> gold_participant_entities;
	map<int, Entity *> gold_time_entities;
	map<int, Entity *> gold_loc_entities;

	map<int, Entity *> predict_entities;
	map<int, Entity *> predict_participant_entities;
	map<int, Entity *> predict_time_entities;
	map<int, Entity *> predict_loc_entities;

	string topic_id;
	string topic;
	vector<Sentence *> sentences;

	map<int, float> tf_map;

	string doc_id;
	int gold_entity_size;
	int predict_entity_size;

	int doc_index;
	map<int, double> doc_fvec;
	double doc_norm;

	string predict_doc_ant;
	int predict_topic;
	vector<Document*> nearest_neighbors;
	map<string, double> doc_distance;
};

#endif /* DOCUMENT_H_ */
