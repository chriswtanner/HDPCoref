/*
 * Entity.h
 *
 *  Created on: Mar 19, 2013
 *      Author: bishan
 */

#ifndef ENTITY_H_
#define ENTITY_H_

#include "./Parsing/Utils.h"
#include "./Parsing/Dictionary.h"
#include <string>
#include <vector>

using namespace std;

enum EntityType { EVENT, PARTICIPANT, EN_TIME, EN_LOC };

typedef struct ITEM {
  int    wid;	               /* word number */
  float  weight;              /* word weight */
};

class Argument {
public:
	int sent_id;
	int word_id;
	string word_str;
	vector<string> rel;
public:
	Argument() : sent_id(-1), word_id(-1) {}

	bool Equal(Argument *arg) {
		return (sent_id == arg->sent_id && word_id == arg->word_id);
	}

	void CommonRels(Argument *arg, vector<string> &c_rels) {
		for (int i = 0; i < rel.size(); ++i) {
			for (int j = 0; j < arg->rel.size(); ++j) {
				if (rel[i] == arg->rel[j]) {
					c_rels.push_back(rel[i]);
				}
			}
		}
	}

	string toRelStr() {
		string str = "";
		for (int i = 0; i < rel.size(); ++i) str += rel[i]+",";
		return str;
	}
};

class Mention;

class AntecedentLink {
public:
	AntecedentLink(Mention* pm, double s) {
		m = pm;
		score = s;
	}
public:
	Mention *m;
	double score;
};

class Mention {
public:
	Mention() : sent_id(-1), mention_id(-1), start_offset(-1), end_offset(-1), mention_length(-1),
	ant_mention_id(-1), head_index(-1), twinless(true), antecedent(NULL), doc_salience(0),
	    gold_entity_id(-1), pred_entity_id(-1), valid(true),
	    time(NULL), location(NULL), non_anarphoric_local_prob(0.0),
	    non_anarphoric_global_prob(0.0), entity_type(EVENT),
	    mention_type(NOMINAL), anno_prob(0.0),
	    head_synonym_vec(NULL), head_hypernym_vec(NULL), head_verbnet_vec(NULL), head_framenet_vec(NULL), word_vec(NULL), context_vec(NULL), tfidf_vec(NULL),
	    srl_role_vec(NULL), srl_arg0_vec(NULL), srl_arg1_vec(NULL), srl_participant_vec(NULL), srl_time_vec(NULL), srl_loc_vec(NULL), srl_arg2_vec(NULL) {

		head_synonym_norm = 0.0;
		head_hypernym_norm = 0.0;
		head_verbnet_norm = 0.0;
		head_framenet_norm = 0.0;

		word_vec_norm = 0.0;
		context_vec_norm = 0.0;
		srl_role_norm = 0.0;
		srl_arg0_norm = 0.0;
		srl_arg1_norm = 0.0;
		srl_arg2_norm = 0.0;
		srl_participant_norm = 0.0;
		srl_time_norm = 0.0;
		srl_loc_norm = 0.0;
	}

	virtual ~Mention();

	int Length() { return end_offset - start_offset + 1; }
	void SetMentionID(int i) { mention_id = i; }
	int MentionID() { return mention_id; }
	void SetSentenceID(int i) { sent_id = i; }
	int SentenceID() { return sent_id; }
	void SetDocID(string s) { doc_id = s; }
	string DocID() { return doc_id; }
	void SetStartOffset(int s) { start_offset = s; }
	void SetEndOffset(int e) { end_offset = e; }
	int StartOffset() { return start_offset; }
	int EndOffset() { return end_offset; }

	Mention* GetArg0Mention();
	Mention* GetArg1Mention();
	Mention* GetArg2Mention();
	Mention* GetTimeMention();
	Mention* GetLocMention();
	void GetParticipantMentions(vector<Mention*> &args);

	void CopyFeatures(Mention *m) {
		head_synonym_vec = m->head_synonym_vec;
		head_hypernym_vec = m->head_hypernym_vec;
		head_verbnet_vec = m->head_verbnet_vec;
		head_framenet_vec = m->head_framenet_vec;

		word_vec = m->word_vec; // whole mentions
		context_vec = m->context_vec;
		srl_participant_vec = m->srl_participant_vec;

		srl_role_vec = m->srl_role_vec;
		srl_arg0_vec = m->srl_arg0_vec;
		srl_arg1_vec = m->srl_arg1_vec;
		srl_arg2_vec = m->srl_arg2_vec;
		srl_time_vec = m->srl_time_vec;
		srl_loc_vec = m->srl_loc_vec;
	}

	void Copy(Mention *m) {
		doc_id = m->DocID();
		sent_id = m->SentenceID();
		mention_id = m->MentionID();
		start_offset = m->StartOffset();
		end_offset = m->EndOffset();

		gold_entity_id = m->gold_entity_id;

		head_index = m->head_index;
		head_word = m->head_word;
		head_span = m->head_span;
		head_pos = m->head_pos;
		head_lemma = m->head_lemma;

		mention_str = m->mention_str;
		mention_type = m->mention_type;

	    animate = m->animate;
	    dep_verb = m->dep_verb;
	    gender = m->gender;
	    is_dirobj = m->is_dirobj;
	    is_indirobj = m->is_indirobj;
	    is_prepobj = m->is_prepobj;
	    is_subj = m->is_subj;
	    ner = m->ner;
	    number = m->number;
	    person = m->person;
	}

	Mention *Copy() {
		Mention *m = new Mention();
		m->Copy(this);
		return m;
	}

	bool Overlap(Mention *m) {
		if ((start_offset >= m->start_offset && start_offset <= m->end_offset) ||
				(end_offset >= m->start_offset && end_offset <= m->end_offset)) return true;
		return false;
	}

	bool AddMention(vector<Mention*> &mlist, bool repeated) {
		// Make sure there are no duplicate mentions.
		for (int i = 0; i < mlist.size(); ++i) {
			// head duplicate
			//if (!repeated) {
				//if (mlist[i]->head_index >= 0 && mlist[i]->head_index == head_index) return false;
				//if (mlist[i]->Overlap(this)) return false;
			//}
			if (mlist[i]->Equal(this)) return false;
		}
		mlist.push_back(this);

		return true;
	}

    void SetMentionType(string type) {
    		if (type == "NOMINAL") {
			mention_type = NOMINAL;
		} else if (type == "PROPER") {
			mention_type = PROPER;
		} else if (type == "PRONOMINAL") {
			mention_type = PRONOMINAL;
		} else if (type == "VERBAL") {
			mention_type = VERBAL;
		}
    }

    string ToString() {
    		//return doc_id+":"+Utils::int2string(sent_id)+":"+mention_str+":"+head_word;
    		return mention_str;
    }

    string ToKeyStr() {
    	return Utils::int2string(sent_id)+","+Utils::int2string(start_offset)+","+Utils::int2string(end_offset);
    }

    bool IsPronoun() { return mention_type == PRONOMINAL; }

    bool MoreRepresentativeThan(Mention *m);

	static bool increase_start(Mention * m1, Mention *m2) {
		if (m1->StartOffset() == m2->StartOffset()) {
			return (m1->EndOffset() < m2->EndOffset());
		} else {
			return (m1->StartOffset() < m2->StartOffset());
		}
	}

	bool Equal(Mention *m) {
		return (m->DocID() == doc_id && m->SentenceID() == sent_id
				&& m->StartOffset() == start_offset
				&& m->EndOffset() == end_offset);
				//&& m->entity_type == entity_type);
	}

	bool Contain(Mention *m) {
		return (start_offset <= m->StartOffset() && end_offset >= m->EndOffset());
	}

	bool ContainWord(string word) {
		vector<string> words;
		Utils::Split(mention_str, ' ', words);
		for (int i = 0; i < words.size(); ++i) {
			if (words[i] == word) return true;
		}
		return false;
	}

	bool MatchSrlArguments(Mention *m);
	bool HeadSynMatch(Mention *m);

	string GetHeadLemma();
	bool HeadMatch(Mention *m);
	double Similarity(Mention *m);

	string GetHeadSpan();

	void transform_local_prob();
	void transform_global_prob();

	string MentionTypeStr() {
		if (mention_type == PRONOMINAL) {
			return "PRONOMIAL";
		} else if (mention_type == NOMINAL) {
			return "NOMINAL";
		} else if (mention_type == PROPER) {
			return "PROPER";
		} else if (mention_type == VERBAL) {
			return "VERBAL";
		}
		return "";
	}

	bool SameSentence(Mention *m) {
		if (doc_id == m->doc_id && sent_id == m->sent_id) return true;
		return false;
	}

public:
	bool valid;
	bool twinless;

	int gold_entity_id;
	int pred_entity_id;

	int ant_mention_id;
	Mention *antecedent;

	double doc_salience; // #times its head word appear in the document

	double non_anarphoric_local_prob;
	double non_anarphoric_global_prob;
	vector<AntecedentLink> local_antecedents;
	vector<AntecedentLink> global_antecedents;

	EntityType entity_type;

	string anno_type;
	double anno_prob;

	// External resource.
	MentionType mention_type;

	string doc_id;
	int sent_id;

	// Unique id in the system, use for identify antecedents.
	int mention_id;
	int start_offset;
	int end_offset;
	int mention_length;

	string head_pos;
	// Headword is lemmatized.
	string head_word;
	string head_lemma;
	string head_span;
	int head_index;

	int *head_synonym_vec;
	int *head_hypernym_vec;
	int *head_verbnet_vec;
	int *head_framenet_vec;

	vector<string> word_features;

	ITEM *word_vec;

	ITEM *context_vec;
	ITEM *srl_arg0_vec;
	ITEM *srl_arg1_vec;
	ITEM *srl_arg2_vec;
	ITEM *srl_participant_vec;
	ITEM *srl_time_vec;
	ITEM *srl_loc_vec;

	double word_vec_norm;
	double context_vec_norm;
	double srl_arg0_norm;
	double srl_arg1_norm;
	double srl_arg2_norm;
	double srl_participant_norm;
	double srl_time_norm;
	double srl_loc_norm;

	int *srl_role_vec;
	double srl_role_norm;

	double head_synonym_norm;
	double head_hypernym_norm;
	double head_verbnet_norm;
	double head_framenet_norm;

	ITEM *tfidf_vec; // whole mentions

	string mention_str;

	string animate;
	string number;
	string gender;
	string person;
	string ner;

	string is_subj;
	string is_dirobj;
	string is_indirobj;
	string is_prepobj;
	string dep_verb;

	// Event structure
	map<string, Argument*> arguments; // A0, nsub, nobj, ...

	// E.g. A attacked B. Each role maps to one mention.
	map<string, vector<Mention *> > srl_args; // args (A, B) for mention attacked (A0, A1, LOC, TIME, ...)
	map<string, vector<Mention *> > srl_predicates; // attacked for mention A or B.
	Mention *time;
	Mention *location;
};

class Entity {
public:
	Entity() : entity_type(EVENT), entity_id(-1), representative(NULL) {
		doc_id = 0;
	}

	virtual ~Entity();

	void SetEntityID(int i) { entity_id = i; }
	int EntityID(){ return entity_id; }

	vector<Mention *> *GetCorefMentions() { return &coref_mentions; }
	Mention *GetCorefMention(int i) { return coref_mentions[i]; }
	void AddMention(Mention *m) {
		//entity_type = m->entity_type;
		coref_mentions.push_back(m);
	}

	int Size() { return coref_mentions.size(); }

	// Merge the source to the target by adding mentions.
	void MergeWithEntity(Entity *source);
	void CopyEntity(Entity *en);
	void SortMentionInLinearOder() {
		sort(coref_mentions.begin(), coref_mentions.end(), Mention::increase_start);
	}

	string EntityTypeStr() {
		if (entity_type == EVENT) {
			return "Event";
		} else {
			return "Arg";
		}
	}

	bool IsSinglePronounCluster();

	string ToString();

	EntityType entity_type;
	Mention *representative;

	int entity_id;
	int doc_id;
	vector<Mention *> coref_mentions;
};

#endif /* ENTITY_H_ */
