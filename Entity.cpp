/*
 * Entity.cpp
 *
 *  Created on: Mar 19, 2013
 *      Author: bishan
 */

#include "Entity.h"

Entity::~Entity() {
	// TODO Auto-generated destructor stub
}

void Entity::MergeWithEntity(Entity *source) {
	vector<Mention *> *src_mentions = source->GetCorefMentions();
	for (int i = 0; i < src_mentions->size(); ++i) {
		coref_mentions.push_back((*src_mentions)[i]);
	}
}

void Entity::CopyEntity(Entity *en) {
	entity_id = en->EntityID();
	entity_type = en->entity_type;
	vector<Mention *> *mentions = en->GetCorefMentions();
	coref_mentions.resize(mentions->size());
	for (int i = 0; i < mentions->size(); ++i) {
		coref_mentions[i] = (*mentions)[i];
	}
}

string Entity::ToString() {
	string str;
	for (int i = 0; i < coref_mentions.size(); ++i) {
		str += coref_mentions[i]->ToString() + ", ";
	}
	return str.substr(0, str.size() - 2);
}


Mention::~Mention() {
	// TODO Auto-generated destructor stub
}

bool Mention::MoreRepresentativeThan(Mention *m){
    if(m==NULL) return true;
    if(mention_type!=m->mention_type) {
      if ((mention_type == PROPER && m->mention_type != PROPER)
           || (mention_type == NOMINAL && mention_type == PRONOMINAL)) {
        return true;
      } else {
        return false;
      }
    } else {
      if (head_index - start_offset > m->head_index - m->start_offset) {
        return true;
      } else if (sent_id < m->sent_id || (sent_id == m->sent_id && head_index < m->head_index)) {
        return true;
      } else {
        return false;
      }
    }
  }

bool isOverlap(int* fvector1, int* fvector2) {
	if (fvector1 == NULL || fvector2 == NULL) return false;

	int i = 0;
	int j = 0;
	double score = 0.0;
	while (fvector1[i] != -1 && fvector2[j] != -1) {
		if (fvector1[i] == fvector2[j]) {
			score += 1;
			++i;
			++j;
		} else if (fvector1[i] < fvector2[j]) {
			++i;
		} else {
			++j;
		}
	}
	if (score > 0) return true;
	return false;
}

double Mention::Similarity(Mention *m) {
	//if (HeadSynMatch(m)) {
	/*if (HeadMatch(m)) {
		//if (!MatchDepArguments(m))
		//	return 0;
		//if (!MatchSrlArguments(m) && !MatchDepArguments(m))
		if (!MatchSrlArguments(m))
			return 0;
		return 1;
	} else
		return 0;*/
	if (HeadMatch(m)) {
		// if there is no argument info
		//if ((srl_arg0_vec == 0 && srl_arg1_vec == 0) || (m->srl_arg0_vec == 0 && m->srl_arg1_vec == 0)) return 1;
		/*bool match = false;
		// if any of the argument overlap??
		if (srl_arg0_vec != NULL && m->srl_arg0_vec != NULL) {
			match = true;
			if(isOverlap(srl_arg0_vec, m->srl_arg0_vec)) return 1;
		} else if (srl_arg1_vec != NULL && m->srl_arg1_vec != NULL) {
			match = true;
			if(isOverlap(srl_arg1_vec, m->srl_arg1_vec)) return 1;
		} else if (srl_time_vec != NULL && m->srl_time_vec != NULL) {
			match = true;
			if(isOverlap(srl_time_vec, m->srl_time_vec)) return 1;
		} else if (srl_loc_vec != NULL && m->srl_loc_vec != NULL) {
			match = true;
			if(isOverlap(srl_loc_vec, m->srl_loc_vec)) return 1;
		}

		//return 0;
		if (match) return 0;*/
		return 1;
	} else {
		/*bool arg0_match = false;
		bool arg1_match = false;
		bool time_match = false;
		bool loc_match = false;
		// if any of the argument overlap??
		if (srl_arg0_vec != NULL && m->srl_arg0_vec != NULL) {
			if(isOverlap(srl_arg0_vec, m->srl_arg0_vec)) arg0_match = true;
		}
		if (srl_arg1_vec != NULL && m->srl_arg1_vec != NULL) {
			if(isOverlap(srl_arg1_vec, m->srl_arg1_vec)) arg1_match = true;
		}
		if (srl_time_vec != NULL && m->srl_time_vec != NULL) {
			if(isOverlap(srl_time_vec, m->srl_time_vec)) time_match = true;
		}
		if (srl_loc_vec != NULL && m->srl_loc_vec != NULL) {
			if(isOverlap(srl_loc_vec, m->srl_loc_vec)) loc_match = true;
		}

		//if (arg0_match && arg1_match) return 1;
		//if (time_match && loc_match) return 1;
		if (arg0_match && arg1_match && time_match && loc_match) return 1;
*/
		return 0;
	}
}

string Mention::GetHeadLemma() {
	return head_lemma;
}

string Mention::GetHeadSpan() {
	return head_span;
}

bool Mention::HeadMatch(Mention *m) {
	return m->head_lemma == head_lemma;
}

bool Mention::HeadSynMatch(Mention *m) {
	return Dictionary::IsWordnetSynonym(head_word, head_pos, m->head_word, m->head_pos);
}

bool Mention::MatchSrlArguments(Mention *m) {
	//if (srl_args.size() == 0 || m->srl_args.size() == 0) return true;
	bool match = false;
	/*for (map<string, Mention*>::iterator it = srl_args.begin(); it != srl_args.end(); ++it) {
		string role = it->first;
		Mention *m1 = it->second;
		for (map<string, Mention*>::iterator it2 = m->srl_args.begin();
				it2 != m->srl_args.end(); ++it2) {
			if (role == it2->first
					&& m1->HeadMatch(it2->second)) {
				match = true;
				break;
			}
		}
	}*/

	return match;
}

Mention* Mention::GetArg0Mention() {
	if (srl_args.find("A0") != srl_args.end()) {
		return srl_args["A0"][0];
	}
	return NULL;
}

Mention* Mention::GetArg1Mention() {
	if (srl_args.find("A1") != srl_args.end()) {
		return srl_args["A1"][0];
	}
	return NULL;
}

Mention* Mention::GetArg2Mention() {
	if (srl_args.find("A2") != srl_args.end()) {
		return srl_args["A2"][0];
	}
	return NULL;
}

void Mention::GetParticipantMentions(vector<Mention*> &args) {

}

Mention* Mention::GetTimeMention() {
	if (srl_args.find("TIME") != srl_args.end()) {
		return srl_args["TIME"][0];
	}
	return NULL;
}

Mention* Mention::GetLocMention() {
	if (srl_args.find("LOC") != srl_args.end()) {
		return srl_args["LOC"][0];
	}
	return NULL;
}

void Mention::transform_local_prob() {
	double sum = 0;
	for (int i = 0; i < local_antecedents.size(); ++i) {
		sum += local_antecedents[i].score;
	}
	if (sum == 0.0) return;
	for (int i = 0; i < local_antecedents.size(); ++i) {
		local_antecedents[i].score = (1.0 - non_anarphoric_local_prob) * (local_antecedents[i].score / sum);
	}
}

void Mention::transform_global_prob() {
	double sum = 0;
	for (int i = 0; i < global_antecedents.size(); ++i) {
		sum += global_antecedents[i].score;
	}
	if (sum == 0.0) return;
	for (int i = 0; i < global_antecedents.size(); ++i) {
		global_antecedents[i].score = (1.0 - non_anarphoric_global_prob) * (global_antecedents[i].score / sum);
	}
}
