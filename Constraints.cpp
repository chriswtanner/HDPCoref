/*
 * Constraints.cpp
 *
 *  Created on: Oct 1, 2013
 *      Author: bishan
 */

#include "Constraints.h"

Constraints::Constraints() {
	// TODO Auto-generated constructor stub

}

Constraints::~Constraints() {
	// TODO Auto-generated destructor stub
}

int Constraints::DependencyAgreement(Mention *m1, Mention *m2) {
	if (m1->dep_verb == "-" || m2->dep_verb == "-") {
		return -1;
	} else {
		vector<string> parts;
		Utils::Split(m1->dep_verb, '-', parts);
		string dep_word_a = parts[0];
		string dep_pos_a = parts[1];
		parts.clear();
		Utils::Split(m2->dep_verb, '-', parts);
		string dep_word_b = parts[0];
		string dep_pos_b = parts[1];
		if ((dep_word_a == dep_word_b ||
				Dictionary::IsWordnetSynonym(dep_word_a, dep_pos_a, dep_word_b, dep_pos_b)) &&
				DependencyRoleAgreement(m1, m2)) {
			return 1;
		}
	}
	return 0;
}

bool Constraints::EntityDisagreement(Mention *m1, Mention *m2) {
	// Must-not-link constraints, return 0.
	if (NumberAgreement(m1, m2) == 0 ||
		GenderAgreement(m1, m2) == 0 ||
		AnimateAgreement(m1, m2) == 0 ||
		PersonAgreement(m1, m2) == 0 ||
		!EntityTypesAgree(m1, m2))
	return true;

	return false;
}

bool Constraints::EntityAgreement(Mention *m1, Mention *m2) {
	// Must-link constraints, return 1.
	if (NumberAgreement(m1, m2) == 1 &&
		GenderAgreement(m1, m2) == 1 &&
		AnimateAgreement(m1, m2) == 1 &&
		PersonAgreement(m1, m2) == 1 &&
		EntityTypesAgree(m1, m2))
	return true;

	return false;
}

bool Constraints::EntityTypesAgree(Mention *m1, Mention *m2) {
  if (m1->mention_type == PRONOMINAL) {
	  if (m2->ner == "O") {
		return true;
	  } else if (m2->ner == "MISC") {
		return true;
	  } else if (m2->ner == "ORGANIZATION") {
		return Dictionary::organizationPronouns(m1->head_word);
	  } else if (m2->ner == "PERSON") {
		return Dictionary::personPronouns(m1->head_word);
	  } else if (m2->ner == "LOCATION") {
		return Dictionary::locationPronouns(m1->head_word);
	  } else if (m2->ner == "DATE" || m2->ner == "TIME") {
		return Dictionary::dateTimePronouns(m1->head_word);
	  } else if (m2->ner == "MONEY" || m2->ner == "PERCENT" || m2->ner == "NUMBER") {
		return Dictionary::moneyPercentNumberPronouns(m1->head_word);
	  } else {
		return false;
	  }
  }
  return m1->ner == m2->ner;
}


// Must-not-link constraints: return 0.
// Must-link constraints: return 1.
// Otherwise return 0.5.
double Constraints::PairwiseDistance(Mention *m1, Mention *m2) {
	if (m1->mention_type == PRONOMINAL || m2->mention_type == PRONOMINAL) {
		return 0;
	}

	if (EntityDisagreement(m1, m2)) return 0;

	// Must-link constraints, return 1.
	//if (Dictionary::IsWordnetSynonym(m1->head_word, m1->head_pos, m2->head_word, m2->head_pos)
	//	&& DependencyAgreement(m1, m2) == 1)
	//	return 1;
	if (m1->HeadSynMatch(m2)) {
		if (m1->srl_args.size() == 0 && m2->srl_args.size() == 0) return 1;

		// Make sure srl arguments match
		if (m1->srl_args.size() > 0 && m2->srl_args.size() > 0) {
			/*for (map<string, Mention*>::iterator it = m1->srl_args.begin(); it != m1->srl_args.end(); ++it) {
				string role = it->first;
				Mention *m = it->second;
				for (map<string, Mention*>::iterator it2 = m2->srl_args.begin();
						it2 != m2->srl_args.end(); ++it2) {
					if (m->HeadSynMatch(it2->second)) {
						if (role != it2->first) { // cannot-link: head word equal but role doesn't equal
							return 0;
						} else if (EntityDisagreement(m, it2->second)) { //cannot-link
							return 0;
						}
					}
				}
			}*/
			return 1;
		} else {
			return 1;
		}
	}

	// Head synonym doesn't match, return 0.5.
	return 0.5;
/*	if (m1->pred_entity_id == m2->pred_entity_id) return 1;
	else return 0;*/
}

bool Constraints::MentionTypeAgreement(Mention *m1, Mention *m2) {
	return (m1->mention_type == m2->mention_type);
}

int Constraints::NumberAgreement(Mention *m1, Mention *m2) {
	if (m1->number == "UNKNOWN" || m2->number == "UNKNOWN") {
		return -1;
	} else {
		if (m1->number == m2->number) return 1;
		else return 0;
	}
}

int Constraints::GenderAgreement(Mention *m1, Mention *m2) {
	if (m1->gender == "UNKNOWN" || m2->gender == "UNKNOWN") {
		return -1;
	} else {
		if (m1->gender == m2->gender) return 1;
		else return 0;
	}
}

int Constraints::AnimateAgreement(Mention *m1, Mention *m2) {
	if (m1->animate == "UNKNOWN" || m2->animate == "UNKNOWN") {
		return -1;
	} else {
		if (m1->animate == m2->animate) return 1;
		else return 0;
	}
}

int Constraints::PersonAgreement(Mention *m1, Mention *m2) {
	if (m1->person == "UNKNOWN" || m2->person == "UNKNOWN") {
		return -1;
	} else {
		if (m1->person == m2->person) return 1;
		else return 0;
	}
}

bool Constraints::DependencyRoleAgreement(Mention *m1, Mention *m2) {
	return (m1->is_dirobj == m2->is_dirobj &&
			m1->is_indirobj == m2->is_indirobj &&
			m1->is_prepobj == m2->is_prepobj &&
			m1->is_subj == m2->is_subj);
}

