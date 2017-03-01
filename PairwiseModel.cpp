/*
 * PairwiseModel.cpp
 *
 *  Created on: Nov 9, 2014
 *      Author: bishan
 */

#include "PairwiseModel.h"
#include <assert.h>

PairwiseModel::PairwiseModel() {
	// TODO Auto-generated constructor stub
	embed_dim = 0;
	word_embeddings = NULL;

	singleton_local_prob = 0.1;
	singleton_global_prob = 0.1;
}

PairwiseModel::~PairwiseModel() {
	// TODO Auto-generated destructor stub
}

void PairwiseModel::GenPairwiseEntityFeatures(Mention *m1, Mention *m2, map<string, float> &fvec) {
	fvec.clear();

	//fvec[MentionTypePair(m1, m2)] = 1.0;

	// headword match (lemma match)
	if (m1->head_lemma == m2->head_lemma) fvec["head_match"] = 1.0;

	//fvec["Head_" + m1->head_lemma + "-" + m2->head_lemma] = 1.0;

	if (m1->mention_type != VERBAL && m2->mention_type != VERBAL) {
		fvec[m1->ner + "-" + m2->ner] = 1.0;
		fvec[m1->number + "-" + m2->number] = 1.0;
		fvec[m1->gender + "-" + m2->gender] = 1.0;
		fvec[m1->animate + "-" + m2->animate] = 1.0;
	}

	// srl predicate match

	fvec["pair_bias"] = 1.0;
}

int PairwiseModel::MentionGroupAlign(vector<Mention*> &g1, vector<Mention*> &g2) {
	if (g1.size() == 0 || g2.size() == 0) return -1;

	int match = 0;
	for (int i = 0; i < g1.size(); ++i) {
		for (int j = 0; j < g2.size(); ++j) {
			if (g1[i]->HeadMatch(g2[j])) {
				//fvec["time_match"] = 1;
				//return;
				match++;
			}
		}
	}
	//if (match > 0) return 1;
	return match;
}

void PairwiseModel::RightFeatures(Mention *m1, Sentence *s1, Mention *m2, Sentence *s2, map<string, float> &fvec) {
	//if (m1->RightMention() != NULL && m2->RightMention() != NULL) {
	//	fvec["Right_"+m1->RightMention()->head_lemma + "-" + m2->RightMention()->head_lemma] = 1.0;
	//}
	if (s1->doc_id == s2->doc_id && s1->sent_id == s2->sent_id) return;

	vector<string> context_1;
	vector<string> context_2;
	for (int i = m1->end_offset + 1; i < min(m1->end_offset + 4, s1->TokenSize()); ++i) {
		context_1.push_back(s1->tokens[i]->lemma);
	}
	for (int i = m2->end_offset + 1; i < min(m2->end_offset + 4, s2->TokenSize()); ++i) {
		context_2.push_back(s2->tokens[i]->lemma);
	}
	int overlap = 0;
	for (int i = 0; i < context_1.size(); ++i) {
		for (int j = 0; j < context_2.size(); ++j) {
			if (context_1[i] == context_2[j]) {
				overlap++;
			}
		}
	}
	fvec["Right_overlap_" + Utils::int2string(overlap)] = 1;
}

void PairwiseModel::LeftFeatures(Mention *m1, Sentence *s1, Mention *m2, Sentence *s2, map<string, float> &fvec) {
	//if (m1->LeftMention() != NULL && m2->LeftMention() != NULL) {
	//	fvec["Left_"+m1->LeftMention()->head_lemma + "-" + m2->LeftMention()->head_lemma] = 1.0;
	//}
	if (s1->doc_id == s2->doc_id && s1->sent_id == s2->sent_id) return;

	vector<string> context_1;
	vector<string> context_2;
	for (int i = max(m1->start_offset - 4, 0); i < min(m1->start_offset - 1, s1->TokenSize()); ++i) {
		context_1.push_back(s1->tokens[i]->lemma);
	}
	for (int i = max(m1->start_offset - 4, 0); i < min(m2->start_offset - 1, s2->TokenSize()); ++i) {
		context_2.push_back(s2->tokens[i]->lemma);
	}
	int overlap = 0;
	for (int i = 0; i < context_1.size(); ++i) {
		for (int j = 0; j < context_2.size(); ++j) {
			if (context_1[i] == context_2[j]) {
				overlap++;
			}
		}
	}
	fvec["Left_overlap_" + Utils::int2string(overlap)] = 1;
}

// only for mentions with the same head ???
void PairwiseModel::DepArgumentFeatures(Mention *m1, Mention *m2, map<string, float> &fvec) {
	if (m1->head_lemma != m2->head_lemma) return;
	if (m1->arguments.size() == 0 || m2->arguments.size() == 0) return;

	for (map<string, Argument*>::iterator it = m1->arguments.begin();
			it != m1->arguments.end(); ++it) {
		for (map<string, Argument*>::iterator it2 = m2->arguments.begin();
			it2 != m2->arguments.end(); ++it2) {
			// check if argument info conflict
			Argument *arg1 = it->second;
			Argument *arg2 = it2->second;

			vector<string> rels;
			arg1->CommonRels(arg2, rels);
			if (rels.size() == 0) {
				fvec["dep_conflict"] = 1.0;
			} else {
				if (arg1->word_str == arg2->word_str) {
					for (int i = 0; i < rels.size(); ++i) {
						//fvec[m1->head_lemma + "_dep_"+rels[i] + "_" + arg1->word_str] = 1.0;
						fvec["dep_"+rels[i]] = 1.0;
					}
				}
			}
		}
	}
}

void PairwiseModel::SRLPredicateFeatures(Mention *m1, Mention *m2, map<string, float> &fvec) {
	if (m1->srl_predicates.size() == 0 || m2->srl_predicates.size() == 0) return;

	for (map<string, vector<Mention*> >::iterator it = m1->srl_predicates.begin(); it != m1->srl_predicates.end(); ++it) {
		string mPredicate = it->first;
		if (m2->srl_predicates.find(mPredicate) != m2->srl_predicates.end()) {
			for (int i = 0; i < it->second.size(); ++i) {
				for (int j = 0; j < m2->srl_predicates[mPredicate].size(); ++j) {
					if (it->second[i]->HeadMatch(m2->srl_predicates[mPredicate][j])) {
						fvec["share_predicate"] = 1.0;
					}
				}
			}
		}
	}
}

// only for mentions in different sentences
void PairwiseModel::SRLArgumentFeatures(Mention *m1, Mention *m2, map<string, float> &fvec) {
/*	if (m1->srl_args.size() == 0 || m2->srl_args.size() == 0) return;
	// should be in different sentences ???
	if (m1->sent_id == m2->sent_id) return;

	// A0 && A1 matching
	if(m1->srl_args.find("A0") != m1->srl_args.end() && m2->srl_args.find("A0") != m2->srl_args.end() ) {
		Mention *mArg = m1->srl_args["A0"];
		Mention *aArg = m2->srl_args["A0"];
		if(mArg->HeadMatch(aArg)) {
			fvec["a0_head_match"] = 1.0;
		} else {
			fvec["a0_"+mArg->head_lemma+"-"+aArg->head_lemma] = 1.0;
		}
		if(Constraints::EntityAgreement(mArg, aArg)) {
			fvec["a0_match"] = 1.0;
		} else {
			fvec["a0_mismatch"] = 1.0;
		}
	}

	if (m1->srl_args.find("A1") != m1->srl_args.end() && m2->srl_args.find("A1") != m2->srl_args.end()) {
		Mention *mArg = m1->srl_args["A1"];
		Mention *aArg = m2->srl_args["A1"];
		if(mArg->HeadMatch(aArg)) {
			fvec["a1_head_match"] = 1.0;
		} else {
			fvec["a1_"+mArg->head_lemma+"-"+aArg->head_lemma] = 1.0;
		}
		if(Constraints::EntityAgreement(mArg, aArg)) {
			fvec["a1_match"] = 1.0;
		} else {
			fvec["a1_mismatch"] = 1.0;
		}
	}

	// check if other arguments overlap...
	set<string> roleCollapsed1;
	roleCollapsed1.insert("A2");
	roleCollapsed1.insert("A3");
	roleCollapsed1.insert("A4");
	roleCollapsed1.insert("A5");

	set<string> roleCollapsed2;
	roleCollapsed2.insert("AM-DIR");
	roleCollapsed2.insert("AM-LOC");

	for (map<string, Mention*>::iterator it = m1->srl_args.begin(); it != m1->srl_args.end(); ++it) {
		string mRole = it->first;
		for (map<string, Mention*>::iterator ait = m2->srl_args.begin(); ait != m2->srl_args.end(); ait++) {
			string aRole = ait->first;
			if(mRole == aRole || (roleCollapsed1.find(mRole) != roleCollapsed1.end()
					&& roleCollapsed1.find(aRole) != roleCollapsed1.end())
					|| (roleCollapsed2.find(mRole) != roleCollapsed2.end()
							&& roleCollapsed2.find(aRole) != roleCollapsed2.end())) {
			  Mention *mArg = it->second;
			  Mention *aArg = it->second;
			  if(mArg->HeadMatch(aArg)) {
				  fvec["srlarg_head_match"] = 1.0;
			  }
			  if(Constraints::EntityAgreement(mArg, aArg)) {
				  fvec["srlarg_match"] = 1.0;
			  } else {
				  fvec["srlarg_mismatch"] = 1.0;
			  }
		   }
		}
	}*/
}

void PairwiseModel::GenCDEventPairFeatures(Entity *e1, Entity *e2, map<string, float> &fvec) {
	// aggregate features
	// head-match
/*	int match = 0;
	for (int i = 0; i < e1->coref_mentions.size(); ++i) {
		for (int j = 0; j < e2->coref_mentions.size(); ++j) {
			Mention *m1 = e1->coref_mentions[i];
			Mention *m2 = e2->coref_mentions[j];
			if (m1->head_lemma == m2->head_lemma) {
				match++;
			}
		}
	}
	if (match > 0) fvec["agg_head_match"] = match;
	//else fvec["agg_head_mismatch"] = 1.0;

	// doc salience
//	int salience_1 = 0, salience_2 = 0;
//	for (int i = 0; i < e1->coref_mentions.size(); ++i) {
//		Mention *m1 = e1->coref_mentions[i];
//		salience_1 += m1->doc_salience;
//	}
//	for (int j = 0; j < e2->coref_mentions.size(); ++j) {
//		Mention *m2 = e2->coref_mentions[j];
//		salience_2 += m2->doc_salience;
//	}
//	fvec["agg_salience"] = abs(salience_1 - salience_2);

	// aggregate word_vec features
	map<int, int> fea_vec_1;
	for (int i = 0; i < e1->coref_mentions.size(); ++i) {
		int *wv = e1->coref_mentions[i]->word_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_1.find(wid) == fea_vec_1.end()) {
				fea_vec_1[wid] = 1;
			} else {
				fea_vec_1[wid] += 1;
			}
		}
	}
	map<int, int> fea_vec_2;
	for (int i = 0; i < e2->coref_mentions.size(); ++i) {
		int *wv = e2->coref_mentions[i]->word_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_2.find(wid) == fea_vec_2.end()) {
				fea_vec_2[wid] = 1;
			} else {
				fea_vec_2[wid] += 1;
			}
		}
	}
	double wv_sim = GetFeatureMapSim(fea_vec_1, fea_vec_2);
	if (wv_sim > 0)
		//fvec["wv_sim_"+Utils::int2string(wv_sim)] = 1;
		fvec["agg_wv_sim"] = wv_sim;
	//else
	//	fvec["no_wv_sim"] = 1;

	// head synonym vectors
	fea_vec_1.clear();
	for (int i = 0; i < e1->coref_mentions.size(); ++i) {
		int *wv = e1->coref_mentions[i]->head_synonym_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_1.find(wid) == fea_vec_1.end()) {
				fea_vec_1[wid] = 1;
			} else {
				fea_vec_1[wid] += 1;
			}
		}
	}
	fea_vec_2.clear();
	for (int i = 0; i < e2->coref_mentions.size(); ++i) {
		int *wv = e2->coref_mentions[i]->head_synonym_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_2.find(wid) == fea_vec_2.end()) {
				fea_vec_2[wid] = 1;
			} else {
				fea_vec_2[wid] += 1;
			}
		}
	}
	double syn_sim = GetFeatureMapSim(fea_vec_1, fea_vec_2);
	if (syn_sim > 0)
		//fvec["syn_sim_"+Utils::int2string(syn_sim)] = 1;
		fvec["agg_syn_sim"] = syn_sim;

	// head hypernym vectors
	fea_vec_1.clear();
	for (int i = 0; i < e1->coref_mentions.size(); ++i) {
		int *wv = e1->coref_mentions[i]->head_hypernym_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_1.find(wid) == fea_vec_1.end()) {
				fea_vec_1[wid] = 1;
			} else {
				fea_vec_1[wid] += 1;
			}
		}
	}
	fea_vec_2.clear();
	for (int i = 0; i < e2->coref_mentions.size(); ++i) {
		int *wv = e2->coref_mentions[i]->head_hypernym_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_2.find(wid) == fea_vec_2.end()) {
				fea_vec_2[wid] = 1;
			} else {
				fea_vec_2[wid] += 1;
			}
		}
	}
	double hyp_sim = GetFeatureMapSim(fea_vec_1, fea_vec_2);
	if (hyp_sim > 0)
		//fvec["hyp_sim_"+Utils::int2string(hyp_sim)] = 1;
		fvec["agg_hyp_sim"] = hyp_sim;

	// head verbnet vectors
	fea_vec_1.clear();
	for (int i = 0; i < e1->coref_mentions.size(); ++i) {
		int *wv = e1->coref_mentions[i]->head_verbnet_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_1.find(wid) == fea_vec_1.end()) {
				fea_vec_1[wid] = 1;
			} else {
				fea_vec_1[wid] += 1;
			}
		}
	}
	fea_vec_2.clear();
	for (int i = 0; i < e2->coref_mentions.size(); ++i) {
		int *wv = e2->coref_mentions[i]->head_verbnet_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_2.find(wid) == fea_vec_2.end()) {
				fea_vec_2[wid] = 1;
			} else {
				fea_vec_2[wid] += 1;
			}
		}
	}
	double vn_sim = GetFeatureMapSim(fea_vec_1, fea_vec_2);
	if (vn_sim > 0)
		//fvec["vn_sim_"+Utils::int2string(vn_sim)] = 1;
		fvec["agg_vn_sim"] = vn_sim;

	// head framenet vectors
	fea_vec_1.clear();
	for (int i = 0; i < e1->coref_mentions.size(); ++i) {
		int *wv = e1->coref_mentions[i]->head_framenet_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_1.find(wid) == fea_vec_1.end()) {
				fea_vec_1[wid] = 1;
			} else {
				fea_vec_1[wid] += 1;
			}
		}
	}
	fea_vec_2.clear();
	for (int i = 0; i < e2->coref_mentions.size(); ++i) {
		int *wv = e2->coref_mentions[i]->head_framenet_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_2.find(wid) == fea_vec_2.end()) {
				fea_vec_2[wid] = 1;
			} else {
				fea_vec_2[wid] += 1;
			}
		}
	}
	double fn_sim = GetFeatureMapSim(fea_vec_1, fea_vec_2);
	if (fn_sim > 0)
		//fvec["fn_sim_"+Utils::int2string(fn_sim)] = 1;
		fvec["agg_fn_sim"] = fn_sim;

	// SRL role match
	fea_vec_1.clear();
	for (int i = 0; i < e1->coref_mentions.size(); ++i) {
		int *wv = e1->coref_mentions[i]->srl_role_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_1.find(wid) == fea_vec_1.end()) {
				fea_vec_1[wid] = 1;
			} else {
				fea_vec_1[wid] += 1;
			}
		}
	}
	fea_vec_2.clear();
	for (int i = 0; i < e2->coref_mentions.size(); ++i) {
		int *wv = e2->coref_mentions[i]->srl_role_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_2.find(wid) == fea_vec_2.end()) {
				fea_vec_2[wid] = 1;
			} else {
				fea_vec_2[wid] += 1;
			}
		}
	}
	double role_sim = GetFeatureMapSim(fea_vec_1, fea_vec_2);
	if (role_sim > 0)
		//fvec["role_sim_"+Utils::int2string(role_sim)] = 1;
		fvec["agg_role_sim"] = role_sim;

	// SRL argument match
	fea_vec_1.clear();
	for (int i = 0; i < e1->coref_mentions.size(); ++i) {
		int *wv = e1->coref_mentions[i]->srl_arg0_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_1.find(wid) == fea_vec_1.end()) {
				fea_vec_1[wid] = 1;
			} else {
				fea_vec_1[wid] += 1;
			}
		}
	}
	fea_vec_2.clear();
	for (int i = 0; i < e2->coref_mentions.size(); ++i) {
		int *wv = e2->coref_mentions[i]->srl_arg0_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_2.find(wid) == fea_vec_2.end()) {
				fea_vec_2[wid] = 1;
			} else {
				fea_vec_2[wid] += 1;
			}
		}
	}
	double arg_sim = GetFeatureMapSim(fea_vec_1, fea_vec_2);
	if (arg_sim > 0)
		//fvec["arg_sim_"+Utils::int2string(arg_sim)] = 1;
		fvec["agg_arg0_sim"] = arg_sim;

	fea_vec_1.clear();
	for (int i = 0; i < e1->coref_mentions.size(); ++i) {
		int *wv = e1->coref_mentions[i]->srl_arg1_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_1.find(wid) == fea_vec_1.end()) {
				fea_vec_1[wid] = 1;
			} else {
				fea_vec_1[wid] += 1;
			}
		}
	}
	fea_vec_2.clear();
	for (int i = 0; i < e2->coref_mentions.size(); ++i) {
		int *wv = e2->coref_mentions[i]->srl_arg1_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_2.find(wid) == fea_vec_2.end()) {
				fea_vec_2[wid] = 1;
			} else {
				fea_vec_2[wid] += 1;
			}
		}
	}
	arg_sim = GetFeatureMapSim(fea_vec_1, fea_vec_2);
	if (arg_sim > 0)
		//fvec["arg_sim_"+Utils::int2string(arg_sim)] = 1;
		fvec["agg_arg1_sim"] = arg_sim;

	// time
	fea_vec_1.clear();
	for (int i = 0; i < e1->coref_mentions.size(); ++i) {
		int *wv = e1->coref_mentions[i]->srl_time_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_1.find(wid) == fea_vec_1.end()) {
				fea_vec_1[wid] = 1;
			} else {
				fea_vec_1[wid] += 1;
			}
		}
	}
	fea_vec_2.clear();
	for (int i = 0; i < e2->coref_mentions.size(); ++i) {
		int *wv = e2->coref_mentions[i]->srl_time_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_2.find(wid) == fea_vec_2.end()) {
				fea_vec_2[wid] = 1;
			} else {
				fea_vec_2[wid] += 1;
			}
		}
	}
	double time_sim = GetFeatureMapSim(fea_vec_1, fea_vec_2);
	if (time_sim > 0)
		//fvec["time_sim" + Utils::int2string(time_sim)] = 1;
		fvec["agg_time_sim"] = time_sim;

	// loc
	fea_vec_1.clear();
	for (int i = 0; i < e1->coref_mentions.size(); ++i) {
		int *wv = e1->coref_mentions[i]->srl_loc_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_1.find(wid) == fea_vec_1.end()) {
				fea_vec_1[wid] = 1;
			} else {
				fea_vec_1[wid] += 1;
			}
		}
	}
	fea_vec_2.clear();
	for (int i = 0; i < e2->coref_mentions.size(); ++i) {
		int *wv = e2->coref_mentions[i]->srl_loc_vec;
		if (wv == NULL) continue;
		for (int *f = wv; *f != -1; ++f) {
			int wid = *f;
			if (fea_vec_2.find(wid) == fea_vec_2.end()) {
				fea_vec_2[wid] = 1;
			} else {
				fea_vec_2[wid] += 1;
			}
		}
	}
	double loc_sim = GetFeatureMapSim(fea_vec_1, fea_vec_2);
	if (loc_sim > 0)
		//fvec["loc_sim" + Utils::int2string(loc_sim)] = 1;
		fvec["agg_loc_sim"] = loc_sim;
*/
}

double PairwiseModel::GetFeatureMapSim(map<int, int> &fvec1, map<int, int> &fvec2) {
	double score = 0.0;
	for (map<int, int>::iterator it = fvec1.begin(); it != fvec1.end(); ++it) {
		if (fvec2.find(it->first) != fvec2.end()) {
			//score += it->second * fvec2[it->first];
			score += 1;
		}
	}
	return (double)score/(fvec1.size()+fvec2.size()-score);
}

double PairwiseModel::GetBOWSim(ITEM *fvec1, ITEM *fvec2, double norm1, double norm2) {
	if (fvec1 == NULL || fvec2 == NULL) return 0.0;

	double sum=0;
	while (fvec1->wid != -1 && fvec2->wid != -1) {
		if(fvec1->wid > fvec2->wid) {
			fvec2++;
		} else if (fvec1->wid < fvec2->wid) {
			fvec1++;
		}
		else {
			sum+=(fvec1->weight) * (fvec2->weight);
			fvec1++;
			fvec2++;
		}
	}
	if (sum == 0.0) return -1;
	return sum / (norm1 * norm2);
}

double PairwiseModel::GetFeatureVecSim(int* fvector1, int* fvector2) {
	if (fvector1 == NULL || fvector2 == NULL) return 0.0;
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
	while (fvector1[i] != -1) ++i;
	while (fvector1[j] != -1) ++j;

	double norm1 = sqrt((double)i);
	double norm2 = sqrt((double)j);
	//return (double)score/(i+j-score);
	return (double)score/(norm1*norm2);
}

void PairwiseModel::MergeFeatureVec(int* fvector1, int* fvector2, string key, map<string, float> &fvec) {
	if (fvector1 == NULL || fvector2 == NULL) return;
	int i = 0;
	int j = 0;
	while (fvector1[i] != -1 && fvector2[j] != -1) {
		if (fvector1[i] == fvector2[j]) {
			string fstr = key + "_" + Utils::int2string(fvector1[i]);
			fvec[fstr] = 1;
			++i;
			++j;
		} else if (fvector1[i] < fvector2[j]) {
			++i;
		} else {
			++j;
		}
	}
}

void PairwiseModel::GetSentPairFeatures(Sentence *s1, Sentence *s2, map<string, float>&fvec) {
	double sim_count = GetFeatureVecSim(s1->srl_participant_vec, s2->srl_participant_vec);
	if (sim_count > 0)
		fvec["sent_participant_sim_count"] = sim_count;

	sim_count = GetFeatureVecSim(s1->srl_time_vec, s2->srl_time_vec);
	if (sim_count > 0)
		fvec["sent_time_sim_count"] = sim_count;

	sim_count = GetFeatureVecSim(s1->srl_loc_vec, s2->srl_loc_vec);
	if (sim_count > 0)
		fvec["sent_loc_sim_count"] = sim_count;
}

void PairwiseModel::GenWDEventPairFeatures(Mention *m1, Mention *m2, map<string, float> &fvec) {
	// headword match (lemma match)
	//if (m1->head_lemma == m2->head_lemma) fvec["head_match"] = 1.0;
	//else fvec["head_mismatch"] = 1.0;

	fvec["Head_" + m1->head_lemma + "-" + m2->head_lemma] = 1.0;
	fvec["Head_" + m1->head_pos + "-" + m2->head_pos] = 1.0;

	// whole mention overlap
	double wv_sim_count = GetBOWSim(m1->word_vec, m2->word_vec, m1->word_vec_norm, m2->word_vec_norm);
	if (wv_sim_count > 0)
		fvec["wv_sim_count"] = wv_sim_count;

	if (m1->doc_id == m2->doc_id) {
		int sent_dist = abs(m1->sent_id - m2->sent_id);
		fvec["sent_dist"] = sent_dist;
	}

	// head synonym vectors
	double syn_sim = GetFeatureVecSim(m1->head_synonym_vec, m2->head_synonym_vec);
	if (syn_sim > 0)
		//fvec["syn_sim_"+Utils::int2string(syn_sim)] = 1;
		fvec["syn_sim"] = syn_sim;
	//else
	//	fvec["zero_syn_sim"] = 1;

	// head hypernym vectors
	double hyp_sim = GetFeatureVecSim(m1->head_hypernym_vec, m2->head_hypernym_vec);
	if (hyp_sim > 0)
		//fvec["hyp_sim_"+Utils::int2string(hyp_sim)] = 1;
		fvec["hyp_sim"] = hyp_sim;
	//else
	//	fvec["zero_hyp_sim"] = 1;

	// head verbnet vectors
	double vn_sim = GetFeatureVecSim(m1->head_verbnet_vec, m2->head_verbnet_vec);
	if (vn_sim > 0)
		//fvec["vn_sim_"+Utils::int2string(vn_sim)] = 1;
		fvec["vn_sim"] = vn_sim;
	//else
	//	fvec["zero_vn_sim"] = 1;

	// head framenet vectors
	double fn_sim = GetFeatureVecSim(m1->head_framenet_vec, m2->head_framenet_vec);
	if (fn_sim > 0)
		//fvec["fn_sim_"+Utils::int2string(fn_sim)] = 1;
		fvec["fn_sim"] = fn_sim;
	//else
	//	fvec["zero_fn_sim"] = 1;

	// SRL role match
	double role_sim = GetFeatureVecSim(m1->srl_role_vec, m2->srl_role_vec);
	if (role_sim > 0)
		//fvec["role_sim_"+Utils::int2string(role_sim)] = 1;
		fvec["role_sim"] = role_sim;
	//else
	//	fvec["zero_role_sim"] = 1;

	// ARG0 argument match
/*	double arg0_sim = GetFeatureVecSim(m1->srl_arg0_vec, m2->srl_arg0_vec);
	if (arg0_sim > 0)
		//fvec["arg_sim_"+Utils::int2string(arg_sim)] = 1;
		fvec["arg0_sim"] = arg0_sim;
	//else
	//	fvec["zero_arg_sim"] = 1;

	// ARG1 argument match
	double arg1_sim = GetFeatureVecSim(m1->srl_arg1_vec, m2->srl_arg1_vec);
	if (arg1_sim > 0)
		//fvec["arg_sim_"+Utils::int2string(arg_sim)] = 1;
		fvec["arg1_sim"] = arg1_sim;

	// time
	double time_sim = GetFeatureVecSim(m1->srl_time_vec, m2->srl_time_vec);
	if (time_sim > 0)
		//fvec["time_sim" + Utils::int2string(time_sim)] = 1;
		fvec["time_sim"] = time_sim;
	//else
	//	fvec["zero_time_sim"] = 1;

	// loc
	double loc_sim = GetFeatureVecSim(m1->srl_loc_vec, m2->srl_loc_vec);
	if (loc_sim > 0)
		//fvec["loc_sim" + Utils::int2string(loc_sim)] = 1;
		fvec["loc_sim"] = loc_sim;
	//else
	//	fvec["zero_loc_sim"] = 1;
	 */
}

void PairwiseModel::GenCDEventPairFeatures(Mention *m1, Mention *m2, map<string, float> &fvec) {
	// string match (lemma match)
	if (m1->head_lemma == m2->head_lemma) fvec["head_match"] = 1.0;
	else if (m1->head_lemma.find(m2->head_lemma) != string::npos ||
			m2->head_lemma.find(m1->head_lemma) != string::npos) {
		fvec["string_match"] = 1.0;
	}

	double pmi = Dictionary::getPMIscore(m1->head_lemma, m2->head_lemma);
	if (pmi > 0.0)
		fvec["pmi_score"] = pmi;

	string key1 = m1->head_lemma + "/" + m1->head_pos[0];
	string key2 = m2->head_lemma + "/" + m2->head_pos[0];
	double event_sim = Dictionary::getEventFeatureSim(key1, key2);
	if (event_sim > 0.0)
		fvec["event_sim"] = event_sim;

	//else fvec["head_mismatch"] = 1.0;

	//fvec["Head_" + m1->head_lemma + "-" + m2->head_lemma] = 1.0;

	string pos1 = m1->head_pos.substr(0,1);
	string pos2 = m2->head_pos.substr(0,1);
	string head_poss = "Head_" + pos1 + "-" + pos2;
	fvec[head_poss] = 1.0;

	if (m1->doc_id != m2->doc_id) {
		//int salience = abs(m1->doc_salience - m2->doc_salience);
		//fvec["salience_" + Utils::int2string(salience)] = 1;
		//fvec["salience"] = (double)salience/max(m1->doc_salience, m2->doc_salience);
		//fvec["salience"] = salience;
	} else {
		int sent_dist = abs(m1->sent_id - m2->sent_id);
		if (sent_dist <= 3)
			fvec["sent_dist_"+Utils::int2string(sent_dist)] = 1;
	}

	// whole mention overlap
	if (m1->Length() > 1 && m2->Length() > 1) {
		double wv_sim_count = GetBOWSim(m1->word_vec, m2->word_vec, m1->word_vec_norm, m2->word_vec_norm);
		if (wv_sim_count > 0)
			fvec["wv_sim_count"] = wv_sim_count;
	}

	if (!m1->SameSentence(m2)) {
		double context_sim_count = GetBOWSim(m1->context_vec, m2->context_vec, m1->context_vec_norm, m2->context_vec_norm);
		if (context_sim_count > 0)
			fvec["context_sim_count"] = context_sim_count;
	}

	// head synonym vectors
	double syn_sim = GetFeatureVecSim(m1->head_synonym_vec, m2->head_synonym_vec);
	if (syn_sim != 0)
		//fvec["syn_sim_"+Utils::int2string(syn_sim)] = 1;
		fvec["syn_sim"] = syn_sim;
	//else
	//	fvec["zero_syn_sim"] = 1;

	// head hypernym vectors
	double hyp_sim = GetFeatureVecSim(m1->head_hypernym_vec, m2->head_hypernym_vec);
	if (hyp_sim != 0)
		//fvec["hyp_sim_"+Utils::int2string(hyp_sim)] = 1;
		fvec["hyp_sim"] = hyp_sim;
	//else
	//	fvec["zero_hyp_sim"] = 1;

	// head verbnet vectors
	double vn_sim = GetFeatureVecSim(m1->head_verbnet_vec, m2->head_verbnet_vec);
	if (vn_sim != 0)
		//fvec["vn_sim_"+Utils::int2string(vn_sim)] = 1;
		fvec["vn_sim"] = vn_sim;
	//else
	//	fvec["zero_vn_sim"] = 1;

	// head framenet vectors
	double fn_sim = GetFeatureVecSim(m1->head_framenet_vec, m2->head_framenet_vec);
	if (fn_sim != 0)
		//fvec["fn_sim_"+Utils::int2string(fn_sim)] = 1;
		fvec["fn_sim"] = fn_sim;
	//else
	//	fvec["zero_fn_sim"] = 1;

	// SRL role match
/*	double role_sim = GetFeatureVecSim(m1->srl_role_vec, m2->srl_role_vec);
	if (role_sim != 0)
		//fvec["role_sim_"+Utils::int2string(role_sim)] = 1;
		fvec["role_sim"] = role_sim;
	//else
	//	fvec["zero_role_sim"] = 1;

	// ARG0 argument match
	double arg0_sim = GetBOWSim(m1->srl_arg0_vec, m2->srl_arg0_vec, m1->srl_arg0_norm, m2->srl_arg0_norm);
	if (arg0_sim > 0) {
		//fvec["arg_sim_"+Utils::int2string(arg_sim)] = 1;
		fvec["arg0_sim"] = arg0_sim;
	}
	//else
	//	fvec["zero_arg_sim"] = 1;

	// ARG1 argument match
	double arg1_sim = GetBOWSim(m1->srl_arg1_vec, m2->srl_arg1_vec, m1->srl_arg1_norm, m2->srl_arg1_norm);
	if (arg1_sim > 0) {
		//fvec["arg_sim_"+Utils::int2string(arg_sim)] = 1;
		fvec["arg1_sim"] = arg1_sim;
	}

	double arg2_sim = GetBOWSim(m1->srl_arg2_vec, m2->srl_arg2_vec, m1->srl_arg2_norm, m2->srl_arg2_norm);
	if (arg2_sim > 0) {
		//fvec["arg_sim_"+Utils::int2string(arg_sim)] = 1;
		fvec["arg2_sim"] = arg2_sim;
	}

	if (!m1->SameSentence(m2)) {
//		double part_sim = GetBOWSim(m1->srl_participant_vec, m2->srl_participant_vec, m1->srl_participant_norm, m2->srl_participant_norm);
//		if (part_sim > 0)
//			fvec["participant_sim"] = part_sim;

		// time
		double time_sim = GetBOWSim(m1->srl_time_vec, m2->srl_time_vec, m1->srl_time_norm, m2->srl_time_norm);
		if (time_sim > 0)
			//fvec["time_sim" + Utils::int2string(time_sim)] = 1;
			fvec["time_sim"] = time_sim;
		//else
		//	fvec["zero_time_sim"] = 1;

		// loc
//		double loc_sim = GetBOWSim(m1->srl_loc_vec, m2->srl_loc_vec, m1->srl_loc_norm, m2->srl_loc_norm);
//		if (loc_sim > 0)
//			//fvec["loc_sim" + Utils::int2string(loc_sim)] = 1;
//			fvec["loc_sim"] = loc_sim;
		//else
		//	fvec["zero_loc_sim"] = 1;
	}*/
}

string PairwiseModel::MentionTypePair(Mention *m1, Mention *m2) {
	if (m1->mention_type == m2->mention_type) {
		return m1->MentionTypeStr() + "-" + m2->MentionTypeStr();
	} else {
		if (m1->mention_type == VERBAL) {
			return m1->MentionTypeStr() + "-" + m2->MentionTypeStr();
		} else if (m2->mention_type == VERBAL) {
			return m2->MentionTypeStr() + "-" + m1->MentionTypeStr();
		} else if (m1->mention_type == NOMINAL) {
			return m1->MentionTypeStr() + "-" + m2->MentionTypeStr();
		} else if (m2->mention_type == NOMINAL) {
			return m2->MentionTypeStr() + "-" + m1->MentionTypeStr();
		} else if (m1->mention_type == PROPER) {
			return m1->MentionTypeStr() + "-" + m2->MentionTypeStr();
		} else if (m2->mention_type == PROPER) {
			return m2->MentionTypeStr() + "-" + m1->MentionTypeStr();
		} else {
			return "";
		}
	}
}

void PairwiseModel::GenSingletonFeatures(Sentence *sent, Mention *m, map<string, float> &fvec) {
	// determiner
	string pos = sent->tokens[m->start_offset]->pos;
	if (pos == "DT") {
		fvec["singleton_DT"] = 1.0;
		//fvec["singleton_"+sent->tokens[m->start_offset]->word] = 1.0;
	}

	// sentence position
	if (m->start_offset < 3) {
		fvec["sent_begin"] = 1.0;
	} else if (m->end_offset >= sent->TokenSize()-3) {
		fvec["sent_end"] = 1.0;
	} else {
		fvec["sent_middle"] = 1.0;
	}

	if (m->srl_args.size() > 0) {
		fvec["srl_args"] = 1.0;
	}

	if (m->srl_predicates.size() > 0) {
		fvec["srl_predicates"] = 1.0;
	}

	// start pos + word
	//fvec["singleton_start_"+pos] = 1.0;
	//fvec["singleton_start_"+sent->tokens[m->start_offset]->lemma] = 1.0;

	// lexical
	//fvec["singleton_"+m->head_lemma] = 1.0;

	// mention type
	fvec["singleton_"+m->MentionTypeStr()] = 1.0;

	if (m->mention_type != VERBAL) {
		fvec[m->ner] = 1.0;
		fvec[m->number] = 1.0;
		fvec[m->animate] = 1.0;
		fvec[m->gender] = 1.0;
	}

	fvec["singleton_bias"] = 1.0;
}

void PairwiseModel::GenEmbeddingFeatures(string word, map<string, float> &fvec) {
	if (word_lookup_table.find(word) != word_lookup_table.end()) {
		int wid = word_lookup_table[word];
		for (int i = 0; i < embed_dim; ++i) {
			fvec["Embed_"+Utils::int2string(i)] = word_embeddings[wid][i];
		}
	}
}

bool PairwiseModel::GetHeadEmbedding(Sentence *s, Mention *m, vector<double> &embed) {
	embed.resize(embed_dim, 0.0);
	int wid = -1;
	// multi-word?
//	if (m->Length() > 1) {
//		string phrase = s->GetOrigSpanStr(m->start_offset, m->end_offset);
//		string phrase_lemma = s->GetSpanLemmaStr(m->start_offset, m->end_offset);
//		string phrase_lower = Utils::toLower(phrase_lemma);
//		if (word_lookup_table.find(phrase_lower) != word_lookup_table.end()) {
//			wid = word_lookup_table[phrase_lower];
//		} else if (word_lookup_table.find(phrase_lemma) != word_lookup_table.end()) {
//			wid = word_lookup_table[phrase_lemma];
//		} else if (word_lookup_table.find(phrase) != word_lookup_table.end()) {
//			wid = word_lookup_table[phrase];
//		}
//	}

	// lemma_lower -> lemma -> word
	if (wid < 0) {
		string lemma_lower = Utils::toLower(m->head_lemma);
		if (word_lookup_table.find(lemma_lower) != word_lookup_table.end()) {
			wid = word_lookup_table[lemma_lower];
		} else if (word_lookup_table.find(m->head_lemma) != word_lookup_table.end()) {
			wid = word_lookup_table[m->head_lemma];
		} else if (word_lookup_table.find(m->head_word) != word_lookup_table.end()) {
			wid = word_lookup_table[m->head_word];
		}
	}

	if (wid >= 0) {
		for (int i = 0; i < embed_dim; ++i) {
			embed[i] = word_embeddings[wid][i];
		}
		return true;
	}
	return false;
}

bool PairwiseModel::GetPhraseEmbedding(Sentence *s, int start, int end, vector<double> &embed) {
	embed.resize(embed_dim, 0.0);

	bool hit = false;
	int count = 0;
	for (int i = start; i <= end; ++i) {
		int wid = -1;
		string lemma_lower = Utils::toLower(s->tokens[i]->lemma);
		if (word_lookup_table.find(lemma_lower) != word_lookup_table.end()) {
			wid = word_lookup_table[lemma_lower];
		}
		if (wid >= 0) {
			hit = true;
			for (int i = 0; i < embed_dim; ++i) {
				embed[i] += word_embeddings[wid][i];
			}
			count++;
		}
	}

	// average
	if (hit) {
		for (int i = 0; i < embed.size(); ++i) {
			embed[i] = (double)embed[i]/count;
		}
		return true;
	} else {
		return false;
	}
}

double PairwiseModel::L2distance(vector<double> &v1, vector<double> &v2) {
	double norm1 = 0.0;
	double norm2 = 0.0;
	double sim = 0;
	for (int i = 0; i < v1.size(); ++i) {
		//sim += (v1[i]-v2[i]) * (v1[i]-v2[i]);
		sim += v1[i]*v2[i];
		norm1 += v1[i]*v1[i];
		norm2 += v2[i]*v2[i];
	}
	//return sqrt(sim);
	return sim/(sqrt(norm1)*sqrt(norm2));
}

void PairwiseModel::GenEntityPairEmbeddingFeatures(Sentence *s1, Mention *m1, Sentence *s2, Mention *m2, map<string, float> &fvec) {
	if (embed_dim == 0) return;

	vector<double> head_embed_1;
	GetHeadEmbedding(s1, m1, head_embed_1);
	vector<double> head_embed_2;
	GetHeadEmbedding(s2, m2, head_embed_2);
	for (int i = 0; i < embed_dim; ++i) {
		//fvec["Embedding_pair_"+Utils::int2string(i)] = word_embeddings[wid1][i] * word_embeddings[wid2][i];
		//double v = head_embed_1[i] - head_embed_2[i];
		//fvec["Embedding_head_"+Utils::int2string(i)] = - v * v;
		fvec["Embedding_head_"+Utils::int2string(i)] = -head_embed_1[i] * head_embed_2[i];
	}
	//fvec["Embedding_head"] = CosineSimilarity(head_embed_1, head_embed_2);
}

void PairwiseModel::GenEventPairEmbeddingFeatures(Mention *m1, Sentence *s1, Mention *m2, Sentence *s2, map<string, float> &fvec) {
	if (embed_dim == 0) return;
	if (m1->mention_str == m2->mention_str) fvec["embed_phrase"] = 1.0;

	vector<double> head_embed_1;
	vector<double> head_embed_2;
	if (GetHeadEmbedding(s1, m1, head_embed_1) && GetHeadEmbedding(s2, m2, head_embed_2)) {
		//for (int i = 0; i < embed_dim; ++i) {
			//fvec["Embedding_pair_"+Utils::int2string(i)] = word_embeddings[wid1][i] * word_embeddings[wid2][i];
			//double v = head_embed_1[i] - head_embed_2[i];
			//fvec["Embedding_head_"+Utils::int2string(i)] = - v * v;
			//fvec["Embedding_head_"+Utils::int2string(i)] = head_embed_1[i] * head_embed_2[i];
		//}
		fvec["embed_head"] = L2distance(head_embed_1, head_embed_2);
	}


	vector<double> phrase_embed_1;
	vector<double> phrase_embed_2;
	if (GetPhraseEmbedding(s1, m1->start_offset, m1->end_offset, phrase_embed_1) && GetPhraseEmbedding(s2, m2->start_offset, m2->end_offset, phrase_embed_2)) {
		//for (int i = 0; i < embed_dim; ++i) {
			//fvec["Embedding_phrase_"+Utils::int2string(i)] = mention_embed_1[i] * mention_embed_2[i];
		//}
		fvec["embed_phrase"] = L2distance(phrase_embed_1, phrase_embed_2);
	}


/*	Mention *m1_arg0 = m1->GetArg0Mention();
	Mention *m2_arg0 = m2->GetArg0Mention();
	if (m1_arg0 != NULL && m2_arg0 != NULL) {
		words1.clear();
		words2.clear();
		head_embed_1.clear();
		head_embed_2.clear();

		if (GetHeadEmbedding(m1_arg0, head_embed_1) && GetHeadEmbedding(m2_arg0, head_embed_2)) {
			fvec["Embedding_srl_arg0_head"] = 1.0 - L2distance(head_embed_1, head_embed_2);
		}

//		phrase_embed_1.clear();
//		phrase_embed_2.clear();
//		if (m1_arg0->Length() > 1 || m2_arg0->Length() > 1) {
//			Utils::Split(m1_arg0->mention_str, ' ', words1);
//			Utils::Split(m2_arg0->mention_str, ' ', words2);
//			if (GetPhraseEmbedding(words1, phrase_embed_1) && GetPhraseEmbedding(words2, phrase_embed_2)) {
//				fvec["Embedding_srl_arg0"] = 1.0 - L2distance(phrase_embed_1, phrase_embed_2);
//			}
//		}
	}

	Mention *m1_arg1 = m1->GetArg1Mention();
	Mention *m2_arg1 = m2->GetArg1Mention();
	if (m1_arg1 != NULL && m2_arg1 != NULL) {
		words1.clear();
		words2.clear();
		head_embed_1.clear();
		head_embed_2.clear();
		if (GetHeadEmbedding(m1_arg1, head_embed_1) && GetHeadEmbedding(m2_arg1, head_embed_2)) {
			fvec["Embedding_srl_arg1_head"] = 1.0 - L2distance(head_embed_1, head_embed_2);
		}

//		phrase_embed_1.clear();
//		phrase_embed_2.clear();
//		if (m1_arg1->Length() > 1 || m2_arg1->Length() > 1) {
//			Utils::Split(m1_arg1->mention_str, ' ', words1);
//			Utils::Split(m2_arg1->mention_str, ' ', words2);
//			if (GetPhraseEmbedding(words1, phrase_embed_1) && GetPhraseEmbedding(words2, phrase_embed_2)) {
//				fvec["Embedding_srl_arg1"] = 1.0 - L2distance(phrase_embed_1, phrase_embed_2);
//			}
//		}
	}

	Mention *m1_arg2 = m1->GetArg2Mention();
	Mention *m2_arg2 = m2->GetArg2Mention();
	if (m1_arg2 != NULL && m2_arg2 != NULL) {
		words1.clear();
		words2.clear();
		head_embed_1.clear();
		head_embed_2.clear();
		if (GetHeadEmbedding(m1_arg2, head_embed_1) && GetHeadEmbedding(m2_arg2, head_embed_2)) {
			fvec["Embedding_srl_arg2_head"] = 1.0 - L2distance(head_embed_1, head_embed_2);
		}

//		phrase_embed_1.clear();
//		phrase_embed_2.clear();
//		if (m1_arg2->Length() > 1 || m2_arg2->Length() > 1) {
//			Utils::Split(m1_arg2->mention_str, ' ', words1);
//			Utils::Split(m2_arg2->mention_str, ' ', words2);
//			if (GetPhraseEmbedding(words1, phrase_embed_1) && GetPhraseEmbedding(words2, phrase_embed_2)) {
//				fvec["Embedding_srl_arg2"] = 1.0 - L2distance(phrase_embed_1, phrase_embed_2);
//			}
//		}
	}*/
/*
	Mention *m1_time = m1->GetTimeMention();
	Mention *m2_time = m2->GetTimeMention();
	if (m1_time != NULL && m2_time != NULL) {
		words1.clear();
		words2.clear();
		head_embed_1.clear();
		head_embed_2.clear();
		phrase_embed_1.clear();
		phrase_embed_2.clear();
		if (GetHeadEmbedding(m1_time, head_embed_1) && GetHeadEmbedding(m2_time, head_embed_2)) {
			fvec["Embedding_srl_time_head"] = 1.0 - L2distance(head_embed_1, head_embed_2);
		}

		if (m1_time->Length() > 1 || m2_time->Length() > 1) {
			Utils::Split(m1_time->mention_str, ' ', words1);
			Utils::Split(m2_time->mention_str, ' ', words2);
			if (GetPhraseEmbedding(words1, phrase_embed_1) && GetPhraseEmbedding(words2, phrase_embed_2)) {
				fvec["Embedding_srl_time"] = 1.0 - L2distance(phrase_embed_1, phrase_embed_2);
			}
		}
	}

	Mention *m1_loc = m1->GetLocMention();
	Mention *m2_loc = m2->GetLocMention();
	if (m1_loc != NULL && m2_loc != NULL) {
		words1.clear();
		words2.clear();
		head_embed_1.clear();
		head_embed_2.clear();
		phrase_embed_1.clear();
		phrase_embed_2.clear();
		if (GetHeadEmbedding(m1_loc, head_embed_1) && GetHeadEmbedding(m2_loc, head_embed_2)) {
			fvec["Embedding_srl_loc_head"] = 1.0 - L2distance(head_embed_1, head_embed_2);
		}

		if (m1_loc->Length() > 1 || m2_loc->Length() > 1) {
			Utils::Split(m1_loc->mention_str, ' ', words1);
			Utils::Split(m2_loc->mention_str, ' ', words2);
			if (GetPhraseEmbedding(words1, phrase_embed_1) && GetPhraseEmbedding(words2, phrase_embed_2)) {
				fvec["Embedding_srl_loc"] = 1.0 - L2distance(phrase_embed_1, phrase_embed_2);
			}
		}
	}*/
}

void PairwiseModel::TrainModel(string lrfile, string modelfile) {
	//encoder.model_type = MAXENT;
	encoder.model_type = LR;

	ifstream infile(lrfile.c_str(), ios::in);
	string line;
	while (getline(infile, line)) {
		map<string, float> fvec;
		int ans = 0;
		LoadLRInstance(line, ans, fvec);

		CRFPP::TaggerImpl* x = new CRFPP::TaggerImpl();
		x->MentionPairToTagger(fvec, ans, encoder.feature_index,  true);
		int xid = labelx.size();
		x->set_thread_id(xid % encoder.thread_num);

		labelx.push_back(x);
	}

	infile.close();

	cout<<"Feature size before shrinking "<<encoder.feature_index->size()<<endl;
	encoder.feature_index->shrink(encoder.freq);
	cout<<"Feature size after shrinking "<<encoder.feature_index->size()<<endl;

	//encoder.feature_index->BuildFeatureVec();

	// allocate nodes, paths, and fvectors
	encoder.RebuildAlpha();

	//encoder.SGD_learn(modelfile.c_str(), "log.txt");
	//encoder.LBFGS_learn(modelfile.c_str());

	encoder.learn(labelx, modelfile);
}

void PairwiseModel::calculateAcc(vector<int> &true_labels, vector<int> &pred_labels)
{
	int err = 0;
	int pp = 0;
	int tp = 0;
	int cp = 0;
	for(int i=0; i<true_labels.size(); i++) {
		if (pred_labels[i] == 1) {
			pp++;
			if (pred_labels[i] == true_labels[i]) cp++;
		}
		if (true_labels[i] == 1) tp++;

		if (pred_labels[i] != true_labels[i]) err++;
	}

	double dev_acc = 1 - (double)err/true_labels.size();
	double dev_pre = (pp == 0) ? 0.0 : (double)cp/pp;
	double dev_rec = (tp == 0) ? 0.0 : (double)cp/tp;
	cout<<"cp:"<<cp<<" tp:"<<tp<<" pp:"<<pp<<endl;
	cout<<"eval acc = "<< dev_acc <<" prec = "<< dev_pre <<" rec = "<< dev_rec <<endl;
}

void PairwiseModel::LoadModel(string modelfile) {
	decoder_feature_index = new CRFPP::DecoderFeatureIndex();
	decoder_feature_index->open(modelfile.c_str());
}

void PairwiseModel::Predict(string lrfile, string outputfile, bool eval) {
	vector<int> true_labels;
	vector<int> pred_labels;
	ofstream out(outputfile.c_str(), ios::out);

	CRFPP::TaggerImpl* x = new CRFPP::TaggerImpl();

	ifstream infile(lrfile.c_str(), ios::in);
	string line;
	while (getline(infile, line)) {
		map<string, float> fvec;
		int ans = 0;
		LoadLRInstance(line, ans, fvec);

		x->clear();
		x->MentionPairToTagger(fvec, ans, decoder_feature_index,  false);

		x->LRInference();

		out<<x->answer_<<"\t"<<x->result_<<endl;

		if (eval) {
			pred_labels.push_back(x->result_);
			true_labels.push_back(x->answer_);
		}
	}

	if (eval)
		calculateAcc(true_labels, pred_labels);

	out.close();

	delete x;
	x = NULL;

}

void PairwiseModel::LoadEmbeddings(string infilename) {
	word_lookup_table.clear();
	vector<string> lines;
	string line;
	ifstream indexfile(infilename.c_str(), ios::in);
	while (getline(indexfile, line)) {
		lines.push_back(line);
	}
	indexfile.close();

	int wsize = lines.size();

	word_embeddings = new float*[wsize];
	for (int wid = 0; wid < wsize; ++wid) {
		string str = lines[wid];
		int index = str.find('\t');
		string word = str.substr(0, index);
		word_lookup_table[word] = wid;

		string values = str.substr(index+1);
		vector<string> fields;
		Utils::Split(values, ' ', fields);
		embed_dim = fields.size();
		word_embeddings[wid] = new float[fields.size()];
		for (int i = 0; i < fields.size(); ++i) {
			word_embeddings[wid][i] = atof(fields[i].c_str());
		}
	}
}

void PairwiseModel::ClearEmbeddings() {
	if (word_embeddings != NULL) {
		delete word_embeddings;
		word_embeddings = NULL;
	}
}

void PairwiseModel::OutputLRData(EncoderFeatureIndex* feature_index, vector<CRFPP::TaggerImpl*> data, string outputfilename) {
	ofstream outfile(outputfilename.c_str(), ios::out);
	for (int i = 0; i < data.size(); ++i) {
		map<int, float> fvec;
		data[i]->LR_fvec(fvec);
		outfile<<data[i]->answer_<<" ";
		for (map<int, float>::iterator it = fvec.begin(); it != fvec.end(); ++it) {
			outfile<<feature_index->fid2fstr[it->first]<<":"<<it->second<<" ";
		}
		outfile<<endl;
	}
	outfile.close();
}

void PairwiseModel::OutputMaxEntData(vector<CRFPP::TaggerImpl*> data, string outputfilename) {
	ofstream outfile(outputfilename.c_str(), ios::out);
	int old_id = -1;
	for (int i = 0; i < data.size(); ++i) {
		vector<map<int, float> > fvecs;
		data[i]->MaxEnt_fvec(fvecs);
		if (data[i]->x == old_id) {
			// only output positive pair
			outfile<<"1 ";
			for (map<int, float>::iterator it = fvecs[0].begin(); it != fvecs[0].end(); ++it) {
				outfile<<it->first+1<<":"<<it->second<<" ";
			}
			outfile<<endl;
		} else {
			for (int j = 0; j < fvecs.size(); ++j) {
				if (j == 0) outfile<<"1 ";
				else outfile<<"0 ";
				for (map<int, float>::iterator it = fvecs[j].begin(); it != fvecs[j].end(); ++it) {
					outfile<<it->first+1<<":"<<it->second<<" ";
				}
				outfile<<endl;
			}
		}
		old_id = data[i]->x;
	}
	outfile.close();
}

void PairwiseModel::BuildCDPairwiseEncoder(CorefCorpus *data, bool train) {
	//encoder.model_type = MAXENT;
	encoder.model_type = LR;

	if (train) {
		labelx.clear();
	} else {
		devx.clear();
	}

	int pos_n = 0;
	int neg_n = 0;
	int total_n = 0;
	int head_acc = 0;

	// positive + negative pairs from within-doc chain
	for (map<string, Document*>::iterator it = data->documents.begin(); it != data->documents.end(); ++it) {
		Document *doc = it->second;
		for (int i = 0; i < doc->SentNum(); ++i) {
			Sentence *s1 = doc->GetSentence(i);
			if (s1->gold_mentions.size() == 0) continue;

			for (int j = 0; j < s1->predict_mentions.size(); ++j) {
				Mention *m1 = s1->predict_mentions[j];

				// same sentence
				for (int j1 = j-1; j1 >= 0; j1--) {
					Mention *m2 = s1->predict_mentions[j1];

					int ans = 0;
					if (m1->gold_entity_id != -1 && m1->gold_entity_id == m2->gold_entity_id) ans = 1;
					if (ans == 1) pos_n++;
					else neg_n++;
					total_n++;
					if (m1->HeadMatch(m2)) {
						if (ans == 1) head_acc++;
					} else {
						if (ans == 0) head_acc++;
					}

					map<string, float> fvec;
					GenCDEventPairFeatures(m1, m2, fvec);
					GenEventPairEmbeddingFeatures(m1, s1, m2, s1, fvec);

					CRFPP::TaggerImpl* x = new CRFPP::TaggerImpl();
					x->MentionPairToTagger(fvec, ans, encoder.feature_index,  train);
					if (train) {
						labelx.push_back(x);
					} else {
						devx.push_back(x);
					}
				}

				// previous sentences
				for (int i1=i-1; i1>=max(i-3,0); i1--) {
					Sentence *s2 = doc->GetSentence(i1);

					// ???
					//if (s2->gold_mentions.size() == 0) continue;

					for (int j1 = 0; j1 < s2->predict_mentions.size(); ++j1) {
						Mention *m2 = s2->predict_mentions[j1];

						int ans = 0;
						if (m1->gold_entity_id != -1 && m1->gold_entity_id == m2->gold_entity_id) ans = 1;
						if (ans == 1) pos_n++;
						else neg_n++;
						total_n++;
						if (m1->HeadMatch(m2)) {
							if (ans == 1) head_acc++;
						} else {
							if (ans == 0) head_acc++;
						}

						map<string, float> fvec;
						GenCDEventPairFeatures(m1, m2, fvec);
						GenEventPairEmbeddingFeatures(m1, s1, m2, s2, fvec);

						CRFPP::TaggerImpl* x = new CRFPP::TaggerImpl();
						x->MentionPairToTagger(fvec, ans, encoder.feature_index,  train);
						if (train) {
							labelx.push_back(x);
						} else {
							devx.push_back(x);
						}
					}
				}
			}
		}
	}

	// only positive pairs from cross-doc chain
	for (map<string, Document*>::iterator it = data->topic_document.begin(); it != data->topic_document.end(); ++it) {
		Document *doc = it->second;
		for (map<int, Entity*>::iterator it1 = doc->gold_entities.begin(); it1 != doc->gold_entities.end(); ++it1) {
			Entity *en = it1->second;
			for (int i = 0; i < en->coref_mentions.size(); ++i) {
				Mention *m1 = en->coref_mentions[i];
				Sentence *s1 = data->documents[m1->doc_id]->GetSentence(m1->sent_id);

				for (int j = i-1; j >=0; j--) {
					Mention *m2 = en->coref_mentions[j];
					if (m2->doc_id == m1->doc_id) continue;

					Sentence *s2 = data->documents[m2->doc_id]->GetSentence(m2->sent_id);

					int ans = 1;
					pos_n++;
					total_n++;
					if (m1->HeadMatch(m2)) {
						if (ans == 1) head_acc++;
					}

					map<string, float> fvec;
					GenCDEventPairFeatures(m1, m2, fvec);
					GenEventPairEmbeddingFeatures(m1, s1, m2, s2, fvec);

					CRFPP::TaggerImpl* x = new CRFPP::TaggerImpl();
					x->MentionPairToTagger(fvec, ans, encoder.feature_index,  train);
					if (train) {
						labelx.push_back(x);
					} else {
						devx.push_back(x);
					}
				}
			}
		}
	}

	// positive and negative pairs from cross-doc chain
/*	for (map<string, Document*>::iterator it = data->topic_document.begin(); it != data->topic_document.end(); ++it) {
		Document *doc = it->second;
		for (map<int, Entity*>::iterator it1 = doc->gold_entities.begin(); it1 != doc->gold_entities.end(); ++it1) {
			Entity *en = it1->second;
			for (int i = 0; i < en->coref_mentions.size(); ++i) {
				Mention *m1 = en->coref_mentions[i];
				Sentence *s1 = data->documents[m1->doc_id]->GetSentence(m1->sent_id);

				// cross-doc sentences
				map<string, Sentence*> cand_sents;
				for (int j = i-1; j >=0; j--) {
					Mention *m2 = en->coref_mentions[j];
					if (m2->doc_id == m1->doc_id) continue;
					string key = m2->doc_id + "_" + Utils::int2string(m2->sent_id);
					cand_sents[key] = data->documents[m2->doc_id]->GetSentence(m2->sent_id);
				}

				for (map<string, Sentence*>::iterator sit = cand_sents.begin(); sit != cand_sents.end(); ++sit) {
					Sentence *s2 = sit->second;
					for (int j = 0; j < s2->predict_mentions.size(); ++j) {
						Mention *m2 = s2->predict_mentions[j];

						int ans = 0;
						if (m1->gold_entity_id != -1 && m1->gold_entity_id == m2->gold_entity_id) ans = 1;
						if (ans == 1) pos_n++;
						else neg_n++;
						total_n++;
						if (m1->HeadMatch(m2)) {
							if (ans == 1) head_acc++;
						} else {
							if (ans == 0) head_acc++;
						}

						map<string, float> fvec;
						GenCDEventPairFeatures(m1, m2, fvec);
						GenEventPairEmbeddingFeatures(m1, s1, m2, s2, fvec);

						CRFPP::TaggerImpl* x = new CRFPP::TaggerImpl();
						x->MentionPairToTagger(fvec, ans, encoder.feature_index,  train);
						if (train) {
							encoder.labelx.push_back(x);
						} else {
							encoder.devx.push_back(x);
						}
					}
				}
			}
		}
	}
*/
	cout<<"training: "<<labelx.size()<<" test: "<<devx.size()<<endl;
	cout<<"positive samples: "<<pos_n<<" negative samples: "<<neg_n<<endl;
	cout<<"head matching baseline: "<<(double)head_acc/total_n<<" "<<total_n<<" "<<head_acc<<endl;


/*	ofstream logfile;
	if (train) logfile.open("train.log", ios::out);
	else logfile.open("test.log", ios::out);

	vector<int> true_labels;
	vector<int> pred_labels;
	for (map<string, vector<Document*> >::iterator it = data->topic_to_documents.begin();
			it != data->topic_to_documents.end(); ++it) {
		// within-topic
		for (int d = 0; d < it->second.size(); ++d) {
			Document *cur_doc = it->second[d];

			vector<Entity*> cand_entities;
			for (int d1 = 0; d1 < d; ++d1) {
				for (map<int, Entity*>::iterator eit = it->second[d1]->predict_entities.begin();
						eit != it->second[d1]->predict_entities.end(); ++eit) {
					eit->second->doc_id = d1;
					cand_entities.push_back(eit->second);
				}
			}

			for (map<int, Entity*>::iterator eit1 = cur_doc->predict_entities.begin();
					eit1 != cur_doc->predict_entities.end(); ++eit1) {
				Entity *en = eit1->second;
				en->doc_id = d;

				// positive entity pairs
				for (int i = 0; i < cand_entities.size(); ++i) {
					Entity *ref_en = cand_entities[i];
					if (en->entity_id != ref_en->entity_id) continue;

					int label = 1;
					pos_n++;

					map<string, float> fvec;
					GenCDEventPairFeatures(en->coref_mentions[0], ref_en->coref_mentions[0], fvec);

					// representative mention only?
					GenEventPairEmbeddingFeatures(en->coref_mentions[0], cur_doc->GetSentence(en->coref_mentions[0]->sent_id),
							ref_en->coref_mentions[0], data->documents[ref_en->coref_mentions[0]->doc_id]->GetSentence(ref_en->coref_mentions[0]->sent_id), fvec);

//					Mention *ref_men = en->coref_mentions[0];
//					Mention *ref_ant_men = ref_en->coref_mentions[0];
//					if (!ref_men->SameSentence(ref_ant_men)) {
//						GetSentPairFeatures(cur_doc->GetSentence(ref_men->sent_id), data->documents[ref_ant_men->doc_id]->GetSentence(ref_ant_men->sent_id), fvec);
//					}

					true_labels.push_back(label);
					if (en->coref_mentions[0]->HeadMatch(ref_en->coref_mentions[0])) {
						pred_labels.push_back(1);
					} else {
						pred_labels.push_back(0);
					}

					CRFPP::TaggerImpl* x = new CRFPP::TaggerImpl();
					x->InitLRTagger(label, encoder.feature_index);
					vector<map<string, float> > fvecs;
					fvecs.push_back(fvec);
					encoder.feature_index->buildFeatures(x, fvecs, train);

					if (train) {
						encoder.labelx.push_back(x);
					} else {
						int xid = encoder.devx.size();
						encoder.devx.push_back(x);
					}
				}

				// negative entity pairs
				for (int i = 0; i < cand_entities.size(); ++i) {
					Entity *ref_en = cand_entities[i];
					if (en->entity_id == ref_en->entity_id) continue;

					// only the first sentence...
					//if (en->coref_mentions[0]->sent_id > 3) continue;
					if (en->doc_id - ref_en->doc_id >= 2) continue;

					int label = 0;
					neg_n++;

					map<string, float> fvec;
					GenCDEventPairFeatures(en->coref_mentions[0], ref_en->coref_mentions[0], fvec);

					// representative mention only?
					GenEventPairEmbeddingFeatures(en->coref_mentions[0], cur_doc->GetSentence(en->coref_mentions[0]->sent_id),
							ref_en->coref_mentions[0], data->documents[ref_en->coref_mentions[0]->doc_id]->GetSentence(ref_en->coref_mentions[0]->sent_id), fvec);

//					Mention *ref_men = en->coref_mentions[0];
//					Mention *ref_ant_men = ref_en->coref_mentions[0];
//					if (!ref_men->SameSentence(ref_ant_men)) {
//						GetSentPairFeatures(cur_doc->GetSentence(ref_men->sent_id), data->documents[ref_ant_men->doc_id]->GetSentence(ref_ant_men->sent_id), fvec);
//					}

					true_labels.push_back(label);
					// Head matching baseline
					if (en->coref_mentions[0]->HeadMatch(ref_en->coref_mentions[0])) {
						pred_labels.push_back(1);
					} else {
						pred_labels.push_back(0);
					}

					CRFPP::TaggerImpl* x = new CRFPP::TaggerImpl();
					x->InitLRTagger(label, encoder.feature_index);
					vector<map<string, float> > fvecs;
					fvecs.push_back(fvec);
					encoder.feature_index->buildFeatures(x, fvecs, train);

					if (train) {
						int xid = encoder.labelx.size();
						encoder.labelx.push_back(x);
						if (((int)encoder.labelx.size())%1000 == 0) {
							cout<<"Loaded "<<encoder.labelx.size()<<" instances......"<<endl;
						}
					} else {
						int xid = encoder.devx.size();
						encoder.devx.push_back(x);
						if (((int)encoder.devx.size())%1000 == 0) {
							cout<<"Loaded "<<encoder.devx.size()<<" instances......"<<endl;
						}
					}
				}
			}
		}
	}

	encoder.calculateAcc(true_labels, pred_labels);
	cout<<"training: "<<encoder.labelx.size()<<" test: "<<encoder.devx.size()<<endl;
	cout<<"positive samples: "<<pos_n<<" negative samples: "<<neg_n<<endl;

	logfile.close();
*/
}

void PairwiseModel::LoadLRInstance(string line, int &ans, map<string, float> &fvec) {
	vector<string> fields;
	Utils::Split(line, '\t', fields);
	if (fields.size() != 6) {
		cout<<"input format wrong!!!"<<endl;
		cout<<line<<endl;
		exit(0);
	}

	ans = atoi(fields[4].c_str());
	vector<string> feas;
	Utils::Split(fields[5], ' ', feas);

	for (int j = 0; j < feas.size(); ++j) {
		int k = feas[j].rfind(':');
		string fstr = feas[j].substr(0,k);
		float fv = atof(feas[j].substr(k+1).c_str());
		fvec[fstr] = fv;
	}
}

