/*
 * Document.cpp
 *
 *  Created on: Mar 19, 2013
 *      Author: bishan
 */

#include "Document.h"
#include "./Clustering/PairwiseClustering.h"
#include <assert.h>
#include <string>

Document::Document() {
	// TODO Auto-generated constructor stub
	doc_index = -1;
	predict_doc_ant = "";
	predict_topic = -1;
	doc_norm = 0.0;
}

Document::~Document() {
	// TODO Auto-generated destructor stub
}

void Document::GroupSentencesByDist() {
	int topK = 3;
	for (int i = 0; i < SentNum(); ++i) {
		Sentence *sent = GetSentence(i);
		for (int j = max(0, i-topK); j <= i; ++j) {
		//for (int j = 0; j <= i; ++j) {
			//if (sent->Overlap(GetSentence(j))) {
				string key = doc_id + "\t" + Utils::int2string(j);
				sent->cand_sents[key] = GetSentence(j);
			//}
		}
	}
}

void Document::GroupSentences(map<string, int> &sentence2idx, vector<vector<double> > &sentence_vecs) {
	int topK = 10;
	double threshold = 0.7;
	for (int i = 0; i < SentNum(); ++i) {
		Sentence *sent = GetSentence(i);
		string sent1 = sent->sent_key;

		vector<pair<int, double> > cands;
		// only consider previous sentences
		for (int j = 0; j < i; ++j) {
			Sentence *cand_sent = GetSentence(j);
			string sent2 = cand_sent->sent_key;
			double sim = 0.0;
			if (sentence2idx.find(sent1) == sentence2idx.end() ||
					sentence2idx.find(sent2) == sentence2idx.end()) {
				sim = -1;
			} else {
				sim = 0;
				vector<double> v1 = sentence_vecs[sentence2idx[sent1]];
				vector<double> v2 = sentence_vecs[sentence2idx[sent2]];
				for (int i = 0; i < v1.size(); ++i) {
					sim += v1[i] * v2[i];
				}
			}
			cands.push_back(make_pair(j, sim));
			//sent->cand_sents.push_back(doc->GetSentence(j));
		}
		sort(cands.begin(), cands.end(), Utils::decrease_second<int, double>);
		for (int j = 0; j < min((int)cands.size(), topK); ++j) {
			if (cands[j].second > threshold) {
				string key = doc_id + "\t" + Utils::int2string(cands[j].first);
				sent->cand_sents[key] = GetSentence(cands[j].first);
			}
		}
	}
}

void Document::ReadAnnotation(vector<string> coref_tags, int sent_id) {
	for (int i = 0; i < sentences[sent_id]->predict_mentions.size(); ++i) {
		int start = sentences[sent_id]->predict_mentions[i]->start_offset;
		int end = sentences[sent_id]->predict_mentions[i]->end_offset;

		string stag = coref_tags[start];
		string etag = coref_tags[end];
		if (stag == "-" || etag == "-") continue;
		vector<string> splits1;
		Utils::Split(stag, '|', splits1);
		vector<string> splits2;
		Utils::Split(etag, '|', splits2);

		int tag = -1;
		if (start == end) {
			for (int j = 0; j < splits1.size(); ++j) {
				int idx1 = splits1[j].find('(');
				if (idx1 < 0) continue;
				int idx2 = splits1[j].find(')');
				if (idx2 >= 0) {
					tag = atoi(splits1[j].substr(idx1+1, idx2-idx1-1).c_str());
					break;
				}
			}
		} else {
			vector<int> starts;
			vector<int> ends;
			for (int j = 0; j < splits1.size(); ++j) {
				int idx1 = splits1[j].find('(');
				if (idx1 < 0) continue;
				int idx2 = splits1[j].find(')');
				if (idx2 < 0) {
					starts.push_back(atoi(splits1[j].substr(idx1+1).c_str()));
				}
			}

			for (int j = 0; j < splits2.size(); ++j) {
				int idx1 = splits2[j].find(')');
				if (idx1 < 0) continue;
				int idx2 = splits2[j].find('(');
				if (idx2 < 0) {
					ends.push_back(atoi(splits2[j].substr(0,idx1).c_str()));
				}
			}

			for (int j = 0; j < starts.size(); ++j) {
				for (int k = 0; k < ends.size(); ++k) {
					if (starts[j] == ends[k]) {
						tag = starts[j];
						break;
					}
				}
				if (tag >= 0) break;
			}
		}

		sentences[sent_id]->predict_mentions[i]->gold_entity_id = tag;
	}
}

void Document::LoadEventCorefClusters(vector<string> coref_tags, Sentence *sent) {
	// The mention span may not be continuous.
	map<int, vector<pair<int, int> > > coref_clusters;
	map<string, vector<pair<int, int> > > additional_mentions;

	map<int, stack<int> > exposedChunkStartIndices;
	map<string, stack<int> > additionalStartIndices;
	for (int i = 0; i < coref_tags.size(); ++i) {
	  string bit = coref_tags[i];
	  if (bit != "-") {
		  vector<string> parts;
		  Utils::Split(bit, '|', parts);
		  for (int j = 0; j < parts.size(); ++j) {
			  string part = parts[j];
			  if (part.find("(") != string::npos && part.find(")") != string::npos) {
				  part = part.substr(1, part.size()-2);
				  if (part.substr(part.size()-1) == "*") {
					  if (additional_mentions.find(part) == additional_mentions.end()) {
						  vector<pair<int, int> > p;
						  additional_mentions[part] = p;
					  }
					  additional_mentions[part].push_back(make_pair(i, i));
				  } else {
					  int coref_id = atoi(part.c_str());
					  if (coref_clusters.find(coref_id) == coref_clusters.end()) {
						  vector<pair<int, int> > mentions;
						  coref_clusters[coref_id] = mentions;
					  }
					  coref_clusters[coref_id].push_back(make_pair(i, i));
				  }
			  } else if (part.find("(") != string::npos) {
				  part = part.substr(1);
				  if (part.substr(part.size()-1) == "*") {
					  if (additionalStartIndices.find(part) == additionalStartIndices.end()) {
						  stack<int> p;
						  additionalStartIndices[part] = p;
					  }
					  additionalStartIndices[part].push(i);
				  } else {
					  int coref_id = atoi(part.c_str());
					  if(exposedChunkStartIndices.find(coref_id) == exposedChunkStartIndices.end()) {
						  stack<int> p;
						  exposedChunkStartIndices[coref_id] = p;
					  }
					  exposedChunkStartIndices[coref_id].push(i);
				  }
			  } else if (part.find(")") != string::npos) {
				  part = part.substr(0, part.size()-1);
				  bool additional = false;
				  if (part.substr(part.size()-1) == "*") {
					  int start_offset = additionalStartIndices[part].top();
					  additionalStartIndices[part].pop();
					  if (additional_mentions.find(part) == additional_mentions.end()) {
						  vector<pair<int, int> > p;
						  additional_mentions[part] = p;
					  }
					  additional_mentions[part].push_back(make_pair(start_offset, i));
				  } else {
					  int coref_id = atoi(part.c_str());
					  assert(exposedChunkStartIndices.find(coref_id) != exposedChunkStartIndices.end());
					  assert(!exposedChunkStartIndices[coref_id].empty());
					  int start_offset = exposedChunkStartIndices[coref_id].top();
					  exposedChunkStartIndices[coref_id].pop();

					  if (coref_clusters.find(coref_id) == coref_clusters.end()) {
						  vector<pair<int, int> > mentions;
						  coref_clusters[coref_id] = mentions;
					  }
					  coref_clusters[coref_id].push_back(make_pair(start_offset, i));
				  }
			  }
		  }
	  }
	}

	map<int, vector<pair<int, int> > >::iterator it;
	for (it = coref_clusters.begin(); it != coref_clusters.end(); ++it) {
		// !!! distinguish from the entity clusters.
		int coref_id = it->first;

		vector<pair<int, int> > mentions = it->second;
		for (int i = 0; i < mentions.size(); ++i) {
			Mention *m = sent->FindPredMentionBySpan(mentions[i].first, mentions[i].second);
			if (m != NULL)
				m->pred_entity_id = coref_id;
		}
	}
}

void Document::BuildCorefClusters(vector<string> coref_tags, Sentence *sent, EntityType type) {
	// The mention span may not be continuous.
	map<int, vector<pair<int, int> > > coref_clusters;
	map<string, vector<pair<int, int> > > additional_mentions;

	map<int, stack<int> > exposedChunkStartIndices;
	map<string, stack<int> > additionalStartIndices;
	for (int i = 0; i < coref_tags.size(); ++i) {
	  string bit = coref_tags[i];
	  if (bit != "-") {
		  vector<string> parts;
		  Utils::Split(bit, '|', parts);
		  for (int j = 0; j < parts.size(); ++j) {
			  string part = parts[j];
			  if (part.find("(") != string::npos && part.find(")") != string::npos) {
				  part = part.substr(1, part.size()-2);
				  if (part.substr(part.size()-1) == "*") {
					  if (additional_mentions.find(part) == additional_mentions.end()) {
						  vector<pair<int, int> > p;
						  additional_mentions[part] = p;
					  }
					  additional_mentions[part].push_back(make_pair(i, i));
				  } else {
					  int coref_id = atoi(part.c_str());
					  if (coref_clusters.find(coref_id) == coref_clusters.end()) {
						  vector<pair<int, int> > mentions;
						  coref_clusters[coref_id] = mentions;
					  }
					  coref_clusters[coref_id].push_back(make_pair(i, i));
				  }
			  } else if (part.find("(") != string::npos) {
				  part = part.substr(1);
				  if (part.substr(part.size()-1) == "*") {
					  if (additionalStartIndices.find(part) == additionalStartIndices.end()) {
						  stack<int> p;
						  additionalStartIndices[part] = p;
					  }
					  additionalStartIndices[part].push(i);
				  } else {
					  int coref_id = atoi(part.c_str());
					  if(exposedChunkStartIndices.find(coref_id) == exposedChunkStartIndices.end()) {
						  stack<int> p;
						  exposedChunkStartIndices[coref_id] = p;
					  }
					  exposedChunkStartIndices[coref_id].push(i);
				  }
			  } else if (part.find(")") != string::npos) {
				  part = part.substr(0, part.size()-1);
				  bool additional = false;
				  if (part.substr(part.size()-1) == "*") {
					  int start_offset = additionalStartIndices[part].top();
					  additionalStartIndices[part].pop();
					  if (additional_mentions.find(part) == additional_mentions.end()) {
						  vector<pair<int, int> > p;
						  additional_mentions[part] = p;
					  }
					  additional_mentions[part].push_back(make_pair(start_offset, i));
				  } else {
					  int coref_id = atoi(part.c_str());
					  assert(exposedChunkStartIndices.find(coref_id) != exposedChunkStartIndices.end());
					  assert(!exposedChunkStartIndices[coref_id].empty());
					  int start_offset = exposedChunkStartIndices[coref_id].top();
					  exposedChunkStartIndices[coref_id].pop();

					  if (coref_clusters.find(coref_id) == coref_clusters.end()) {
						  vector<pair<int, int> > mentions;
						  coref_clusters[coref_id] = mentions;
					  }
					  coref_clusters[coref_id].push_back(make_pair(start_offset, i));
				  }
			  }
		  }
	  }
	}

	map<int, vector<pair<int, int> > >::iterator it;
	for (it = coref_clusters.begin(); it != coref_clusters.end(); ++it) {
		// !!! distinguish from the entity clusters.
		int coref_id = it->first;

		vector<pair<int, int> > mentions = it->second;

		// Sentence-level entity.
		Entity *entity = new Entity();
		entity->SetEntityID(coref_id);
		entity->entity_type = type;

		for (int i = 0; i < mentions.size(); ++i) {
			Mention *mention = new Mention();
			mention->SetSentenceID(sent->SentenceID());
			mention->SetDocID(sent->DocID());
			mention->SetStartOffset(mentions[i].first);
			mention->SetEndOffset(mentions[i].second);

			mention->head_index = mentions[i].first;
			mention->head_lemma = sent->tokens[mention->head_index]->lemma;

			mention->gold_entity_id = coref_id;
			mention->entity_type = type;

			string chain_key = Utils::int2string(coref_id)+"*";
			if (additional_mentions.find(chain_key) != additional_mentions.end()) {
				Mention *prev = mention;
				for (int j = 0; j < additional_mentions[chain_key].size(); ++j) {
					Mention *am = new Mention();
					am->SetSentenceID(sent->SentenceID());
					am->SetDocID(sent->DocID());
					am->SetStartOffset(additional_mentions[chain_key][j].first);
					am->SetEndOffset(additional_mentions[chain_key][j].second);
					am->gold_entity_id = coref_id;
					am->antecedent = prev;
					prev = am;
				}
			}

			// Only add mentions if it is not additional mentions!!!
			entity->AddMention(mention);

			if (type == EVENT) {
				mention->anno_type = "EVENT";
				mention->AddMention(sent->gold_mentions, true);
			} else if (type == PARTICIPANT) {
				mention->anno_type = "PARTICIPANT";
				mention->AddMention(sent->gold_participant_mentions, true);
			} else if (type == EN_TIME) {
				mention->anno_type = "TIME";
				mention->AddMention(sent->gold_time_mentions, true);
			} else if (type == EN_LOC) {
				mention->anno_type = "LOC";
				mention->AddMention(sent->gold_loc_mentions, true);
			}
		}

		if (type == EVENT)
			sent->gold_entities[coref_id] = entity;
		else if (type == PARTICIPANT)
			sent->gold_participant_entities[coref_id] = entity;
		else if (type == EN_TIME)
			sent->gold_time_entities[coref_id] = entity;
		else if (type == EN_LOC)
			sent->gold_loc_entities[coref_id] = entity;
	}
}

void Document::BuildDocumentMentions(bool gold) {
	if (gold) {
		gold_mentions.clear();
		// Build document-level mentions (in the linear order).
		for (int i = 0; i < sentences.size(); ++i) {
			for (int j = 0; j < sentences[i]->gold_mentions.size(); ++j) {
				gold_mentions.push_back(sentences[i]->gold_mentions[j]);
			}
		}

		gold_participant_mentions.clear();
		for (int i = 0; i < sentences.size(); ++i) {
			for (int j = 0; j < sentences[i]->gold_participant_mentions.size(); ++j) {
				gold_participant_mentions.push_back(sentences[i]->gold_participant_mentions[j]);
			}
		}

		gold_time_mentions.clear();
		for (int i = 0; i < sentences.size(); ++i) {
			for (int j = 0; j < sentences[i]->gold_time_mentions.size(); ++j) {
				gold_time_mentions.push_back(sentences[i]->gold_time_mentions[j]);
			}
		}

		gold_loc_mentions.clear();
		for (int i = 0; i < sentences.size(); ++i) {
			for (int j = 0; j < sentences[i]->gold_loc_mentions.size(); ++j) {
				gold_loc_mentions.push_back(sentences[i]->gold_loc_mentions[j]);
			}
		}
	} else {
		predict_mentions.clear();
		// Build document-level mentions (in the linear order).
		for (int i = 0; i < sentences.size(); ++i) {
			for (int j = 0; j < sentences[i]->predict_mentions.size(); ++j) {
				// check head lemma
				//if (sentences[i]->predict_mentions[j]->head_index < 0 || sentences[i]->predict_mentions[j]->head_lemma == "") {
				//	cout<<doc_id<<"\t"<<i<<"\t"<<sentences[i]->predict_mentions[j]->head_lemma<<"\t"
				//			<<sentences[i]->predict_mentions[j]->mention_str<<endl;
				//}
				predict_mentions.push_back(sentences[i]->predict_mentions[j]);
			}
		}

		predict_participant_mentions.clear();
		for (int i = 0; i < sentences.size(); ++i) {
			for (int j = 0; j < sentences[i]->predict_participant_mentions.size(); ++j) {
				predict_participant_mentions.push_back(sentences[i]->predict_participant_mentions[j]);
			}
		}

		predict_time_mentions.clear();
		for (int i = 0; i < sentences.size(); ++i) {
			for (int j = 0; j < sentences[i]->predict_time_mentions.size(); ++j) {
				predict_time_mentions.push_back(sentences[i]->predict_time_mentions[j]);
			}
		}

		predict_loc_mentions.clear();
		for (int i = 0; i < sentences.size(); ++i) {
			for (int j = 0; j < sentences[i]->predict_loc_mentions.size(); ++j) {
				predict_loc_mentions.push_back(sentences[i]->predict_loc_mentions[j]);
			}
		}
	}
}

// Read the entity_id field in each mention in the document.
void Document::BuildDocumentEntities(bool gold) {
	if (gold) {
		gold_entities.clear();
		gold_participant_entities.clear();
		gold_time_entities.clear();
		gold_loc_entities.clear();
		for (int i = 0; i < sentences.size(); ++i) {
			for (map<int, Entity *>::iterator it = sentences[i]->gold_entities.begin();
					it != sentences[i]->gold_entities.end(); ++it) {
				int coref_id = it->first;
				if (gold_entities.find(coref_id) == gold_entities.end()) {
					gold_entities[coref_id] = new Entity();
					gold_entities[coref_id]->CopyEntity(it->second);
				} else {
					gold_entities[coref_id]->MergeWithEntity(it->second);
				}
			}
			for (map<int, Entity *>::iterator it = sentences[i]->gold_participant_entities.begin();
					it != sentences[i]->gold_participant_entities.end(); ++it) {
				int coref_id = it->first;
				if (gold_participant_entities.find(coref_id) == gold_participant_entities.end()) {
					gold_participant_entities[coref_id] = new Entity();
					gold_participant_entities[coref_id]->CopyEntity(it->second);
				} else {
					gold_participant_entities[coref_id]->MergeWithEntity(it->second);
				}
			}
			for (map<int, Entity *>::iterator it = sentences[i]->gold_time_entities.begin();
					it != sentences[i]->gold_time_entities.end(); ++it) {
				int coref_id = it->first;
				if (gold_time_entities.find(coref_id) == gold_time_entities.end()) {
					gold_time_entities[coref_id] = new Entity();
					gold_time_entities[coref_id]->CopyEntity(it->second);
				} else {
					gold_time_entities[coref_id]->MergeWithEntity(it->second);
				}
			}
			for (map<int, Entity *>::iterator it = sentences[i]->gold_loc_entities.begin();
					it != sentences[i]->gold_loc_entities.end(); ++it) {
				int coref_id = it->first;
				if (gold_loc_entities.find(coref_id) == gold_loc_entities.end()) {
					gold_loc_entities[coref_id] = new Entity();
					gold_loc_entities[coref_id]->CopyEntity(it->second);
				} else {
					gold_loc_entities[coref_id]->MergeWithEntity(it->second);
				}
			}
		}
	} else {
		predict_entities.clear();
		predict_participant_entities.clear();
		predict_time_entities.clear();
		predict_loc_entities.clear();
		for (int i = 0; i < sentences.size(); ++i) {
			for (map<int, Entity *>::iterator it = sentences[i]->predict_entities.begin();
					it != sentences[i]->predict_entities.end(); ++it) {
				int coref_id = it->first;
				if (predict_entities.find(coref_id) == predict_entities.end()) {
					predict_entities[coref_id] = new Entity();
					predict_entities[coref_id]->CopyEntity(it->second);
				} else {
					predict_entities[coref_id]->MergeWithEntity(it->second);
				}
			}
			for (map<int, Entity *>::iterator it = sentences[i]->predict_participant_entities.begin();
					it != sentences[i]->predict_participant_entities.end(); ++it) {
				int coref_id = it->first;
				if (predict_participant_entities.find(coref_id) == predict_participant_entities.end()) {
					predict_participant_entities[coref_id] = new Entity();
					predict_participant_entities[coref_id]->CopyEntity(it->second);
				} else {
					predict_participant_entities[coref_id]->MergeWithEntity(it->second);
				}
			}
			for (map<int, Entity *>::iterator it = sentences[i]->predict_time_entities.begin();
					it != sentences[i]->predict_time_entities.end(); ++it) {
				int coref_id = it->first;
				if (predict_time_entities.find(coref_id) == predict_time_entities.end()) {
					predict_time_entities[coref_id] = new Entity();
					predict_time_entities[coref_id]->CopyEntity(it->second);
				} else {
					predict_time_entities[coref_id]->MergeWithEntity(it->second);
				}
			}
			for (map<int, Entity *>::iterator it = sentences[i]->predict_loc_entities.begin();
					it != sentences[i]->predict_loc_entities.end(); ++it) {
				int coref_id = it->first;
				if (predict_loc_entities.find(coref_id) == predict_loc_entities.end()) {
					predict_loc_entities[coref_id] = new Entity();
					predict_loc_entities[coref_id]->CopyEntity(it->second);
				} else {
					predict_loc_entities[coref_id]->MergeWithEntity(it->second);
				}
			}
		}
	}
}

void Document::FreeEntities(map<int, Entity*> &entities) {
	for (map<int, Entity*>::iterator it = entities.begin(); it != entities.end(); ++it) {
		delete it->second;
		it->second = NULL;
	}
}

// Read the entity_id field in each mention in the document.
void Sentence::BuildSentenceEntities(bool gold) {
	if (gold) {
		gold_entities.clear();
		// cout << "gold_mentions size: " << gold_mentions.size() << endl;
		for (int i = 0; i < gold_mentions.size(); ++i) {
			Mention *m = gold_mentions[i];
			if (!m->valid || m->gold_entity_id < 0) continue;

			if (gold_entities.find(m->gold_entity_id) == gold_entities.end()) {
				Entity *en = new Entity();
				en->SetEntityID(m->gold_entity_id);
				gold_entities[m->gold_entity_id] = en;
			}
			gold_entities[m->gold_entity_id]->AddMention(m);
		}
	} else {
		predict_entities.clear();
		for (int i = 0; i < predict_mentions.size(); ++i) {
			Mention *m = predict_mentions[i];
			// Invalid cluster. (e.g. pronouns)
			if (!m->valid || m->pred_entity_id < 0) continue;

			if (predict_entities.find(m->pred_entity_id) == predict_entities.end()) {
				Entity *en = new Entity();
				en->SetEntityID(m->pred_entity_id);
				predict_entities[m->pred_entity_id] = en;
			}
			predict_entities[m->pred_entity_id]->AddMention(m);
		}
	}
}

string Sentence::GetMentionStr(Mention *m) {
	return GetSpanStr(m->StartOffset(), m->EndOffset());
	//return GetSpanContext(m->StartOffset(), m->EndOffset());
}

string Sentence::GetMentionContext(Mention *m) {
	return GetSpanContext(m->StartOffset(), m->EndOffset());
}

string Sentence::GetMentionSRLArguments(Mention *m) {
	string str = "";
	if (m->srl_args.size() == 0) return str;
	str = "args: ";
	for (map<string, vector<Mention*> >::iterator it = m->srl_args.begin(); it != m->srl_args.end(); ++it) {
		str += it->first+":" + "; ";
	}
	return str;
}

void Document::GetOrderedAntecedents(
      Sentence *ant_sent,
      Sentence *m_sent,
      Mention *m,
      map<int, Entity *> &coref_cluster,
      vector <Mention*> &antecedent_mentions) {
	antecedent_mentions.clear();

    // ordering antecedents
/*    if (ant_sent->SentenceID() == m_sent->SentenceID()) {   // same sentence
    	    // Add all the mentions appeared in the sentence.
    	    for (int i = 0; i < m_sent->predict_mentions.size(); ++i) {
    	    		antecedent_mentions.push_back(m_sent->predict_mentions[i]);
    	    }

      if(coref_cluster[m->pred_entity_id].IsSinglePronounCluster()) {
         SortMentionsForPronoun(antecedent_mentions, m, true);
      }
      if(dict.relativePronouns.contains(m1.spanToString()))
    	  	  Collections.reverse(orderedAntecedents);
    } else {    // previous sentence
    	    // Add all the mentions appeared in the sentence.
		for (int i = 0; i < m_sent->predict_mentions.size(); ++i) {
			antecedent_mentions.push_back(m_sent->predict_mentions[i]);
		}
    }
*/
    //return orderedAntecedents;
  }

  /** Divides a sentence into clauses and sort the antecedents for pronoun matching  */
void Document::SortMentionsForPronoun(vector<Mention *> l, Mention *m, bool sameSentence) {
/*    List<Mention> sorted = new ArrayList<Mention>();
    Tree tree = m1.contextParseTree;
    Tree current = m1.mentionSubTree;
    if(sameSentence){
      while(true){
        current = current.ancestor(1, tree);
        if(current.label().value().startsWith("S")){
          for(Mention m : l){
            if(!sorted.contains(m) && current.dominates(m.mentionSubTree)) sorted.add(m);
          }
        }
        if(current.label().value().equals("ROOT") || current.ancestor(1, tree)==null) break;
      }
      if(l.size()!=sorted.size()) {
        SieveCoreferenceSystem.logger.finest("sorting failed!!! -> parser error?? \tmentionID: "+m1.mentionID+" " + m1.spanToString());
        sorted=l;
      } else if(!l.equals(sorted)){
        SieveCoreferenceSystem.logger.finest("sorting succeeded & changed !! \tmentionID: "+m1.mentionID+" " + m1.spanToString());
        for(int i=0; i<l.size(); i++){
          Mention ml = l.get(i);
          Mention msorted = sorted.get(i);
          SieveCoreferenceSystem.logger.finest("\t["+ml.spanToString()+"]\t["+msorted.spanToString()+"]");
        }
      }
    }
*/
    //return sorted;
}

bool Document::RulebasedCoreferent(Entity *e, Entity *ant,
		Mention *m, Mention *ant_m, map<string, double> &distance) {
	// Must-not-link constraints, return 0.
/*	if (!MentionTypeAgreement(m1, m2) ||
			NumberAgreement(m1, m2) == 0 ||
			GenderAgreement(m1, m2) == 0 ||
			AnimateAgreement(m1, m2) == 0 ||
			PersonAgreement(m1, m2) == 0)
		return 0;

	// Must-link constraints, return 1.
	if (DependencyAgreement(m1, m2, distance) == 1) return 1;

	// Uncertain, return 0.5.
	return 0.5;
*/
	return true;
}

/*CorefGraph* Document::BuildCorefGraph() {
	int size = predict_mentions.size();
	CorefGraph *graph = new CorefGraph(size);

	// Assume the mentions are ordered!!!
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < i; ++j) {
			AntecedentEdge *edge = new AntecedentEdge();
			edge->mention_index = i;
			edge->antecedent_index = j;
			edge->distance = Constraints::PairwiseDistance(predict_mentions[i], predict_mentions[j]);
			graph->nodes[i]->antecedent_edges.push_back(edge);
		}
	}

	return graph;
}*/

bool Sentence::similarity_check(Sentence *sent) {
	// check time overlap
	if (predict_time_mentions.size() > 0 && sent->predict_time_mentions.size() > 0) {
		bool overlap = false;
		for (int i = 0; i < predict_time_mentions.size(); ++i) {
			for (int j = 0; j < sent->predict_time_mentions.size(); ++j) {
				if (predict_time_mentions[i]->HeadMatch(sent->predict_time_mentions[j])) {
					overlap = true;
					break;
				}
			}
		}
		return overlap;
	}
	// no rules apply
	return true;
}

void Sentence::AddCoNLLTag(int start, int end, string coref_id, vector<string> &coref_tags) {
	if (start == end) {
		if (coref_tags[start] == "-") {
			coref_tags[start] = "(" + coref_id + ")";
		} else {
			coref_tags[start] += "|(" + coref_id + ")";
		}
	} else {
		if (coref_tags[start] == "-") {
			coref_tags[start] = "(" + coref_id;
		} else {
			coref_tags[start] += "|(" + coref_id;
		}
		if (coref_tags[end] == "-") {
			coref_tags[end] = coref_id + ")";
		} else {
			coref_tags[end] += "|" + coref_id + ")";
		}
	}
}

bool Sentence::Equal(Sentence *sent) {
	if (sent->TokenSize() != tokens.size()) return false;
	for (int i = 0; i < tokens.size(); ++i) {
		if (tokens[i]->word != sent->GetToken(i)->word)
			return false;
	}
	return true;
}

string Sentence::OutputArguments(Mention *m) {
	return "";
}

void Sentence::GetCoNLLPredictMentions(vector<string> & coref_tags) {
	coref_tags.resize(tokens.size(), "-");
	for (int i = 0; i < predict_mentions.size(); ++i) {
		string coref_id = Utils::int2string(i);
		AddCoNLLTag(predict_mentions[i]->StartOffset(),
				predict_mentions[i]->EndOffset(), coref_id, coref_tags);
	}
}

void Sentence::GetCoNLLTags(vector<string> &coref_tags) {
	coref_tags.resize(tokens.size(), "-");
	for (map<int, Entity*>::iterator it = gold_entities.begin();
			it != gold_entities.end(); ++it) {
		string coref_id = Utils::int2string(it->first);
		for (int i = 0; i < it->second->Size(); ++i) {
			Mention *m = it->second->GetCorefMention(i);
			if (m->antecedent != NULL) continue;
			int start = m->StartOffset();
			int end = m->EndOffset();
			AddCoNLLTag(start, end, coref_id, coref_tags);
		}
	}
}

string Sentence::GetParseTag(Mention *m) {
	assert(penntree != NULL);
	return penntree->GetParseTag(m->StartOffset(), m->EndOffset());
}

void Sentence::buildDepGraph(string dep_graph)
{
	vector<string> lines;
	Utils::Split(dep_graph, '\n', lines);

	if (G) {
		delete G;
		G = NULL;
	}
	G = new DependencyGraph();

	map< int, vector<string> > depmap;
	depmap.clear();
	for(int i=0; i<lines.size(); i++)
	{
	   int sindex = lines[i].find("(");
	   int eindex = lines[i].rfind(")");
	   string dep = lines[i].substr(0, sindex); //e.g. det, nsubj

	   string pairs = lines[i].substr(sindex+1, eindex-sindex-1);
	   vector<string> tmp;
	   sindex = pairs.find(", ");
	   tmp.push_back(pairs.substr(0, sindex));
	   tmp.push_back(pairs.substr(sindex+2));

	   sindex = tmp[0].size()-1;
	   while(tmp[0][sindex] != '-') sindex--;
	   int wid1 = atoi(tmp[0].substr(sindex+1).c_str())-1; //wordid-1, since id 0 refers to ROOT

	   sindex = tmp[1].size()-1;
	   while(tmp[1][sindex] != '-') sindex--;
	   int wid2 = atoi(tmp[1].substr(sindex+1).c_str())-1;

	   string v = dep+"/"+Utils::int2string(wid2); //det could contain "_", so use "/" as the delim here
	   Utils::addElem(depmap, wid1, v);

	   string depstr = dep+","+Utils::int2string(wid1)+","+Utils::int2string(wid2);
	}

	G->buildGraph(depmap, tokens.size());
}

void Sentence::buildPennTree(string parse_tree)
{
	if (penntree) {
		delete penntree;
		penntree = NULL;
	}
	penntree = new PennTree();
	penntree->ReadTree(parse_tree);
}

void Sentence::buildDiscourseTree(string discourse_tree)
{
	if (penntree) {
		delete penntree;
		penntree = NULL;
	}
	penntree = new PennTree();
	penntree->ReadDiscourseTree(discourse_tree);
}

void Sentence::SortMentionsInLinearOrder(bool gold) {
	if (gold) {
		sort(gold_mentions.begin(), gold_mentions.end(), Mention::increase_start);
		for (map<int, Entity *>::iterator it = gold_entities.begin(); it != gold_entities.end(); ++it) {
			it->second->SortMentionInLinearOder();
		}
	} else {
		sort(predict_mentions.begin(), predict_mentions.end(), Mention::increase_start);
		for (map<int, Entity *>::iterator it = predict_entities.begin(); it != predict_entities.end(); ++it) {
			it->second->SortMentionInLinearOder();
		}
	}

}

string Sentence::toString()
{
	string str = "";
	for(int i=0; i<tokens.size(); i++)
		str += tokens[i]->word+" ";
	return str.substr(0, str.size()-1);
}

string Sentence::GetOrigSpanStr(int start, int end) {
	if (start < 0 || end >= tokens.size() || start > end) return "";

	string str = "";
	for(int i=start; i<=end; i++) {
		str += tokens[i]->word + "_";
	}
	return str.substr(0, str.size()-1);
}

string Sentence::GetSpanWordStr(int start, int end) {
	if (start < 0 || end >= tokens.size() || start > end) return "";

	string str = "";
	for(int i=start; i<=end; i++) {
		str += tokens[i]->word + " ";
	}
	return str.substr(0, str.size()-1);
}

string Sentence::GetSpanLemmaStr(int start, int end) {
	if (start < 0 || end >= tokens.size() || start > end) return "";

	string str = "";
	for(int i=start; i<=end; i++) {
		str += tokens[i]->lemma+"_";
	}
	return str.substr(0, str.size()-1);
}

string Sentence::GetSpanStr(int start, int end)
{
	if (start < 0 || end >= tokens.size() || start > end) return "";

	string str = "";
	for(int i=start; i<=end; i++) {
		//str += tokens[i]->word+" ";
		str += tokens[i]->lemma+" ";
	}
	return str.substr(0, str.size()-1);
}

string Sentence::GetSpanLowerCase(int start, int end)
{
	if (start == end) {
		return tokens[start]->lowercase;
	}
	string str = tokens[start]->lowercase;
	for(int i=start+1; i<=end; i++)
		str += " " + tokens[i]->lowercase;
	return str;
}

string Sentence::GetSpanContext(int start, int end) {
	int context_start = max(0, start - 3);
	int context_end = min(end + 3, (int)tokens.size() - 1);
	string str = "";
	for (int i = context_start; i < start; ++i) {
		str += tokens[i]->word + " ";
	}
	str += "[" + GetSpanStr(start, end) + "] ";
	for (int i = end + 1; i <= context_end; ++i) {
		str += tokens[i]->word + " ";
	}
	return str.substr(0, str.size()-1);
}

void Sentence::SetHeadIndex(Mention *m, int hindex) {
	m->mention_length = m->end_offset - m->start_offset + 1;
	m->mention_str = GetMentionStr(m);

	// Setting head_index right is very important!!!
	m->head_index = hindex;
	if (m->head_index == -1) m->head_index = m->start_offset;

	if (m->anno_type == "TIME") {
		int index;
		if (Dictionary::containTemporal(m->mention_str, index))
			m->head_index = m->start_offset + index;
	} else if (m->anno_type == "LOC") {
		int index;
		if (Dictionary::containLocation(m->mention_str, index))
			m->head_index = m->start_offset + index;
	} else if (m->anno_type == "EVENT") {
		if (tokens[m->head_index]->pos[0] != 'V' && tokens[m->head_index]->pos[0] != 'N') {
			int index = m->start_offset;
			bool found = false;
			for (; index <= m->end_offset; ++index) {
				if (tokens[m->head_index]->pos[index] == 'V' || tokens[m->head_index]->pos[index] == 'N') {
					found = true;
					break;
				}
			}
			if (found) {
				m->head_index = index;
			}
		}
		set<string> stoplist;
		stoplist.insert("be");
		stoplist.insert("have");
		stoplist.insert("has");
		stoplist.insert("seem");
		if (stoplist.find(tokens[m->head_index]->lemma) != stoplist.end()) {
			for (int index = m->head_index+1; index <= m->end_offset; ++index) {
				if (tokens[m->head_index]->pos[index] == 'V' || tokens[m->head_index]->pos[index] == 'N') {
					m->head_index = index;
					break;
				}
			}
		}
	} else if (m->anno_type == "PARTICIPANT") {
		if (tokens[m->head_index]->pos[0] != 'N') {
			int index = m->start_offset;
			bool found = false;
			for (; index <= m->end_offset; ++index) {
				if (tokens[m->head_index]->pos[index] == 'N') {
					found = true;
					break;
				}
			}
			if (found) {
				m->head_index = index;
			}

			set<string> stoplist;
			stoplist.insert("the");
			stoplist.insert("this");
			stoplist.insert("that");
			if (stoplist.find(tokens[m->head_index]->lemma) != stoplist.end()) {
				for (int index = m->head_index+1; index <= m->end_offset; ++index) {
					if (tokens[m->head_index]->pos[index] == 'N') {
						m->head_index = index;
						break;
					}
				}
			}
		}
	}

	m->head_word = tokens[m->head_index]->word;

	// set head span
	m->head_span = m->head_word;

	m->head_pos = tokens[m->head_index]->pos;
	if (m->head_pos == "") {
		cout<<m->head_lemma<<" has empty pos "<<m->mention_str<<endl;
	}

	m->head_lemma = tokens[m->head_index]->lemma;
}

void Sentence::SetHeadForPredictMentions(map<int, int> &head_dict) {
	for (int i = 0; i < predict_mentions.size(); ++i) {
		Mention *men = predict_mentions[i];
		int key = men->start_offset * tokens.size() + men->end_offset;
		if (head_dict.find(key) != head_dict.end()) {
			SetHeadIndex(men, head_dict[key]);
		}
	}
	for (int i = 0; i < predict_participant_mentions.size(); ++i) {
		Mention *men = predict_participant_mentions[i];
		int key = men->start_offset * tokens.size() + men->end_offset;
		if (head_dict.find(key) != head_dict.end()) {
			SetHeadIndex(men, head_dict[key]);
		}
	}
	for (int i = 0; i < predict_time_mentions.size(); ++i) {
		Mention *men = predict_time_mentions[i];
		int key = men->start_offset * tokens.size() + men->end_offset;
		if (head_dict.find(key) != head_dict.end()) {
			SetHeadIndex(men, head_dict[key]);
		}
	}
	for (int i = 0; i < predict_loc_mentions.size(); ++i) {
		Mention *men = predict_loc_mentions[i];
		int key = men->start_offset * tokens.size() + men->end_offset;
		if (head_dict.find(key) != head_dict.end()) {
			SetHeadIndex(men, head_dict[key]);
		}
	}
}

void Sentence::ParseMention(vector<string> fields, Mention *m) {
	m->SetDocID(doc_id);
	int offset = 0;
	m->SetSentenceID(atoi(fields[offset].c_str()));
	m->SetStartOffset(atoi(fields[offset+1].c_str()));
	m->SetEndOffset(atoi(fields[offset+2].c_str())-1);
	m->mention_length = m->end_offset - m->start_offset + 1;
	m->mention_str = GetMentionStr(m);

	// Setting head_index right is very important!!!

	int hindex = atoi(fields[offset+3].c_str());
	m->head_index = hindex;
	if (hindex < 0) m->head_index = m->start_offset;

	if (m->anno_type == "TIME") {
		int index;
		if (Dictionary::containTemporal(m->mention_str, index))
			m->head_index = m->start_offset + index;
	} else if (m->anno_type == "LOC") {
		int index;
		if (Dictionary::containLocation(m->mention_str, index))
			m->head_index = m->start_offset + index;
	} else if (m->anno_type == "EVENT") {
		if (tokens[m->head_index]->pos[0] != 'V') {
			int index = m->start_offset;
			bool found = false;
			for (; index <= m->end_offset; ++index) {
				if (tokens[m->head_index]->pos[index] == 'V') {
					found = true;
					break;
				}
			}
			if (found) {
				m->head_index = index;
			}
		}
	} else if (m->anno_type == "PARTICIPANT") {
		if (tokens[m->head_index]->pos[0] != 'N') {
			int index = m->start_offset;
			bool found = false;
			for (; index <= m->end_offset; ++index) {
				if (tokens[m->head_index]->pos[index] == 'N') {
					found = true;
					break;
				}
			}
			if (found) {
				m->head_index = index;
			}
		}
	}

	m->head_word = tokens[m->head_index]->word;

	// set head span
	m->head_span = m->head_word;
//	for (int i = 0; i < phrases.size(); ++i) {
//		if (phrases[i].isContain(m->head_index) && phrases[i].inRange(m->start_offset, m->end_offset)) {
//			m->head_span = phrases[i].label;
//			break;
//		}
//	}

	//m->head_pos = fields[offset+5];
	m->head_pos = tokens[m->head_index]->pos;
	if (m->head_pos == "") {
		cout<<m->head_lemma<<" has empty pos "<<m->mention_str<<endl;
	}
	//m->head_lemma = Dictionary::getLemma(m->head_word, m->head_pos);
	m->head_lemma = tokens[m->head_index]->lemma;

	// Fix head index.
	/*if (m->head_pos[0] != 'V' && m->head_pos[0] != 'N') {
		for (int i = m->head_index+1; i <= m->EndOffset(); ++i) {
			string pos = doc->GetSentence(m->SentenceID())->GetToken(i)->pos;
			if (pos[0] == 'V' || pos[0] == 'N') {
				//m->head_word += "_" + Dictionary::getLemma(doc->GetSentence(m->SentenceID())->GetToken(i)->word,
				//				doc->GetSentence(m->SentenceID())->GetToken(i)->pos);
				m->head_word += "_" + doc->GetSentence(m->SentenceID())->GetToken(i)->word;
				//cout<<"change ["<<fields[4]<<"] to ["<<m->head_word<<"] for ["<<m->mention_str<<"]"<<endl;
				break;
			}
		}
	}*/

	string mention_type = fields[offset+6];
	if (m->head_pos.substr(0,2) == "VB") {
		mention_type = "VERBAL";
	}
	m->SetMentionType(mention_type);

	m->number = fields[offset+7];
	m->gender = fields[offset+8];
	m->animate = fields[offset+9];
	m->person = fields[offset+10];
	m->ner = fields[offset+11];
	m->is_subj = fields[offset+12];
	m->is_dirobj = fields[offset+13];
	m->is_indirobj = fields[offset+14];
	m->is_prepobj = fields[offset+15];
	m->dep_verb = fields[offset+16];
}

void Sentence::ParseMentionFeatures(vector<string> fields, Mention *m) {
	// mention type: event or not
	string is_event = fields[3];

	// time: NONE or sent,start,end
/*	string time_str = fields[4];
	if (time_str != "NONE") {
		vector<string> splits;
		Utils::Split(time_str, ',', splits);
		int sent = atoi(splits[0].c_str());
		int s = atoi(splits[1].c_str());
		int t = atoi(splits[2].c_str()) - 1;
		m->time = FindPredMentionBySpan(s, t);
		if (m->time != NULL) {
			m->time->entity_type = ARG;
		}
	}

	// location: NONE or sent,start,end
	string loc_str = fields[5];
	if (loc_str != "NONE") {
		vector<string> splits;
		Utils::Split(loc_str, ',', splits);
		int sent = atoi(splits[0].c_str());
		int s = atoi(splits[1].c_str());
		int t = atoi(splits[2].c_str()) - 1;
		m->location = FindPredMentionBySpan(s, t);
		if (m->location != NULL) {
			m->location->entity_type = ARG;
		}
	}

	// arguments: NONE or m=type,start,end,relation (within the sentence)
	string arg_str = fields[6];
	if (arg_str != "NONE") {
		vector<string> splits;
		Utils::Split(arg_str, ';', splits);
		for (int i = 0; i < splits.size(); ++i) {
			int index = splits[i].find('[');
			string relstr = splits[i].substr(index+1, splits[i].size()-1-index-1);
			Argument *arg = new Argument();
			vector<string> splits1;
			Utils::Split(relstr, ',', splits1);
			for (int j = 0; j < splits1.size(); ++j) {
				arg->rel.push_back(splits1[j]);
			}

			string typeinfo = splits[i].substr(0, index);
			splits1.clear();
			Utils::Split(typeinfo, ',', splits1);
			int len = splits1.size();
			string type = splits1[0];
			for (int k = 1; k < len-2; ++k) {
				type += "," + splits1[k];
			}

			arg->sent_id = atoi(splits1[len-2].c_str()) - 1;
			if (arg->sent_id != sent_id) {
				//cout<<m->doc_id<<" "<<m->mention_str<<endl;
				continue;
			}
			arg->word_id = atoi(splits1[len-1].c_str()) - 1;
			if (arg->word_id < 0)
				continue;

			arg->word_str = tokens[arg->word_id]->lemma;

			m->arguments[type] = arg;
		}
	}

	// srl arguments: NONE or type,sent,start,end
	string srl_str = fields[7];
	if (srl_str != "NONE") {
		vector<string> splits;
		Utils::Split(srl_str, ' ', splits);
		for (int i = 0; i < splits.size(); ++i) {
			vector<string> splits1;
			Utils::Split(splits[i], ',', splits1);
			string type = splits1[0];
			int sent = atoi(splits1[1].c_str());
			int s = atoi(splits1[2].c_str());
			int t = atoi(splits1[3].c_str()) - 1;
			Mention *men = FindPredMentionBySpan(s, t);
			if (men != NULL) {
				m->srl_args[type] = men;
				//if ((type == "A0" || type == "A1") && men->mention_type != VERBAL)
				//	men->entity_type = ARG;
			}
		}
	}

	// predicates: NONE or role,sent,start,end
	string pred_str = fields[8];
	if (pred_str != "NONE") {
		vector<string> splits;
		Utils::Split(pred_str, ' ', splits);
		for (int i = 0; i < splits.size(); ++i) {
			vector<string> splits1;
			Utils::Split(splits[i], ',', splits1);
			string type = splits1[0];
			int sent = atoi(splits1[1].c_str());
			int s = atoi(splits1[2].c_str());
			int t = atoi(splits1[3].c_str()) - 1;
			Mention *men = FindPredMentionBySpan(s, t);
			if (men != NULL) {
				if (m->srl_predicates.find(type) == m->srl_predicates.end()) {
					vector<Mention*> mens;
					m->srl_predicates[type] = mens;
				}
				m->srl_predicates[type].push_back(men);
				//if ((type == "A0" || type == "A1") && m->mention_type != VERBAL)
				//	m->entity_type = ARG;
			}
		}
	}*/
}

Mention* Sentence::FetchGoldMention(int start_offset, int end_offset) {
	for (int i = 0; i < gold_mentions.size(); ++i) {
		Mention *m = gold_mentions[i];
		if (m->StartOffset() == start_offset && m->EndOffset() == end_offset) {
			return m;
		}
	}
	for (int i = 0; i < gold_participant_mentions.size(); ++i) {
		Mention *m = gold_participant_mentions[i];
		if (m->StartOffset() == start_offset && m->EndOffset() == end_offset) {
			return m;
		}
	}
	for (int i = 0; i < gold_time_mentions.size(); ++i) {
		Mention *m = gold_time_mentions[i];
		if (m->StartOffset() == start_offset && m->EndOffset() == end_offset) {
			return m;
		}
	}
	for (int i = 0; i < gold_loc_mentions.size(); ++i) {
		Mention *m = gold_loc_mentions[i];
		if (m->StartOffset() == start_offset && m->EndOffset() == end_offset) {
			return m;
		}
	}
	return NULL;
}

Mention* Sentence::FetchPredictMention(int start_offset, int end_offset) {
	for (int i = 0; i < predict_mentions.size(); ++i) {
		Mention *m = predict_mentions[i];
		if (m->StartOffset() == start_offset && m->EndOffset() == end_offset) {
			return m;
		}
	}

	for (int i = 0; i < predict_participant_mentions.size(); ++i) {
		Mention *m = predict_participant_mentions[i];
		if (m->StartOffset() == start_offset && m->EndOffset() == end_offset) {
			return m;
		}
	}

	for (int i = 0; i < predict_time_mentions.size(); ++i) {
		Mention *m = predict_time_mentions[i];
		if (m->StartOffset() == start_offset && m->EndOffset() == end_offset) {
			return m;
		}
	}

	for (int i = 0; i < predict_loc_mentions.size(); ++i) {
		Mention *m = predict_loc_mentions[i];
		if (m->StartOffset() == start_offset && m->EndOffset() == end_offset) {
			return m;
		}
	}

	return NULL;
}

// To do, add normalization forms of verbs.
void Sentence::ExtractVerbalMentions(vector<Mention*> &mentions) {
	mentions.clear();
	int i = 0;
	while (i < tokens.size()) {
		if (tokens[i]->pos.substr(0,2) == "VB") {
			int j = i + 1;
			while (j < tokens.size() && tokens[j]->pos.substr(0,2) == "VB") j++;

			if (i == j-1 && Dictionary::isCopulativeVerb(tokens[i]->word)) {
				i = j;
				continue;
			}
			Mention *m = new Mention();
			m->SetStartOffset(i);
			m->SetEndOffset(j-1);
			mentions.push_back(m);
			i = j;
		} else {
			i++;
		}
	}
}

// To do, add normalization forms of verbs.
void Sentence::ExtractGoldVerbalMentions(vector<Mention*> &mentions) {
	mentions.clear();
	for (int i = 0; i < gold_mentions.size(); ++i) {
		if (gold_mentions[i]->entity_type == EVENT) {
			mentions.push_back(gold_mentions[i]);
		}
	}
}

/*void Sentence::CollectDepArguments(int index, string rel, vector<Argument> &args) {
	vector<int> words1;
	G->getDepArgs(index, rel, words1);
	for (int i = 0; i < words1.size(); ++i) {
		Argument arg;
		int word = words1[i];
		arg.rel = rel;
		if (word < 0) {
			arg.rel = "-1_"+rel;
			word = -word;
		}
		arg.pos = word;
		arg.word = tokens[word]->word;
		arg.postag = tokens[word]->pos;
		arg.lemma = Dictionary::getLemma(arg.word, arg.postag);
		args.push_back(arg);
	}
}

void Sentence::GetDepArguments(int index, vector<Argument> &args) {
	string rel = "nsubj";
	CollectDepArguments(index, rel, args);

	rel = "dobj";
	CollectDepArguments(index, rel, args);

	rel = "iobj";
	CollectDepArguments(index, rel, args);

	rel = "poss";
	CollectDepArguments(index, rel, args);
}
*/

int Sentence::FindVerbalHead(Span argSpan) {
	int headindex = argSpan.start;
	for (int i = argSpan.start; i <= argSpan.end; ++i) {
		if (tokens[i]->pos[0] == 'V') {
			headindex = i;
			break;
		}
	}
	return headindex;
}

int Sentence::FindHeadIndexOfArg(Span argSpan) {
    vector<ParseTreeNode*> leaves = penntree->leaves;
    string pos = tokens[argSpan.start]->pos;

    ParseTreeNode *startNode =
    		((pos == "IN" || pos == "TO" || pos == "RP") && leaves.size() > argSpan.start+1) ? leaves[argSpan.start+1] : leaves[argSpan.start];
    ParseTreeNode *endNode = leaves[argSpan.end-1];
    ParseTreeNode *node = startNode;
    int headIdx = argSpan.end;   // for single token mention

    while(node!=NULL) {
      if(node->dominates(endNode)) {
    	      break; //??
      } else {
        ParseTreeNode *parent = node->getParent();
        if (parent == NULL) headIdx = -1;
        else headIdx = parent->headindex + 1;
        if(headIdx > argSpan.start && headIdx <= argSpan.end) {
          node = parent;
        } else {
          headIdx = node->headindex + 1;
          break;
        }
      }
    }
    return headIdx - 1;
}

Mention *Sentence::FindPredMentionBySpan(int start, int end) {
	for (int i = 0; i < predict_mentions.size(); ++i) {
		if (predict_mentions[i]->start_offset == start && predict_mentions[i]->end_offset == end) {
			return predict_mentions[i];
		}
	}
	return NULL;
}

Mention *Sentence::FindMentionByHead(int headindex) {
	for (int i = 0; i < predict_mentions.size(); ++i) {
		if (predict_mentions[i]->head_index == headindex) {
			return predict_mentions[i];
		}
	}
	return NULL;
}

void Sentence::BuildSrlArguments(map<int, map<pair<int, int>, string> > &args) {
	for(map<int, map<pair<int, int>, string > >::iterator arg_it = args.begin(); arg_it != args.end(); ++arg_it) {
		Mention *mention = FetchPredictMention(arg_it->first, arg_it->first);
		if (mention == NULL || mention->anno_type != "EVENT") {
			// create a new verbal mention
			mention = new Mention();
			mention->SetDocID(doc_id);
			mention->SetSentenceID(sent_id);
			mention->SetStartOffset(arg_it->first);
			mention->SetEndOffset(arg_it->first);

			mention->head_index = arg_it->first;
			mention->head_lemma = tokens[mention->head_index]->lemma;
			mention->mention_str = tokens[mention->head_index]->lemma;

			mention->anno_type = "EVENT";
			AddEventMention(mention);
			continue;
		}

		// Set up srl arguments for mention m.
		for (map<pair<int, int>, string>::iterator it = arg_it->second.begin(); it != arg_it->second.end(); ++it) {
			Span argSpan(it->first.first, it->first.second);
			string role = it->second;
			// only consider AM-TMP, AM-LOC, A0-A5
			if (!(role == "AM-TMP" || role == "AM-LOC" || (role[0] == 'A' && role.size() == 2))) {
				continue;
			}

			if (role == "AM-TMP") {
				role = "TIME";
			} else if (role == "AM-LOC") {
				role = "LOC";
			}

			Mention *argm = FetchPredictMention(argSpan.start, argSpan.end);
			bool compatible = true;
			if (argm != NULL) {
				string argrole = argm->anno_type;
				if (role == "TIME" || role =="LOC") {
					if (argrole != role) compatible = false;
				} else {
					if (argrole == "TIME" || role == "LOC") compatible = false;
				}
			}

			if (argm == NULL || !compatible) {
				argm = new Mention();
				argm->SetDocID(doc_id);
				argm->SetSentenceID(sent_id);
				argm->SetStartOffset(argSpan.start);
				argm->SetEndOffset(argSpan.end);

				argm->head_index = argSpan.start;
				argm->head_lemma = tokens[argm->head_index]->lemma;
				argm->mention_str = GetMentionStr(argm);

				argm->anno_type = role;

				if (argm->anno_type == "TIME") {
					AddTimeMention(argm);
				} else if (argm->anno_type == "LOC") {
					AddLocMention(argm);
				} else {
					AddParticipantMention(argm);
				}
				continue;
			}

			if (mention->srl_args.find(role) == mention->srl_args.end()) {
				vector<Mention *> mlist;
				mention->srl_args[role] = mlist;
			}
			mention->srl_args[role].push_back(argm);

			if (argm->srl_predicates.find(role) == argm->srl_predicates.end()) {
				vector<Mention *> mlist;
				argm->srl_predicates[role] = mlist;
			}
			argm->srl_predicates[role].push_back(mention);
		}
	}
}

void Document::BuildSimpleDocumentVec(map<string, int> &features) {
	//for (int i = 0; i < SentNum(); ++i) {
	for (int i = 0; i < min(3, SentNum()); ++i) {
		Sentence *sent = sentences[i];
//		for (int j = 0; j < sent->predict_mentions.size(); ++j) {
//			Mention *m = sent->predict_mentions[j];
//			for (int k = m->start_offset; k <= m->end_offset; ++k) {
//				string word = "Event_"+sent->tokens[k]->lemma;
//
//				if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;
//
//				if (features.find(word) == features.end()) {
//					features[word] = features.size();
//				}
//			}
//		}

//		for (int j = 0; j < sent->predict_participant_mentions.size(); ++j) {
//			Mention *m = sent->predict_participant_mentions[j];
//			for (int k = m->start_offset; k <= m->end_offset; ++k) {
//				string word = sent->tokens[k]->lemma;
//
//				if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;
//
//				if (features.find(word) == features.end()) {
//					features[word] = features.size();
//				}
//				int fid = features[word];
//				if (doc_fvec.find(fid) == doc_fvec.end()) {
//					doc_fvec[fid] = 1;
//				} else {
//					doc_fvec[fid] += 1;
//				}
//			}
//		}

		for (int j = 0; j < sent->predict_time_mentions.size(); ++j) {
			Mention *m = sent->predict_time_mentions[j];
			for (int k = m->start_offset; k <= m->end_offset; ++k) {
				string word = "Time_"+sent->tokens[k]->lemma;
				if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;
				if (features.find(word) == features.end()) {
					features[word] = features.size();
				}
			}
		}

		for (int j = 0; j < sent->predict_loc_mentions.size(); ++j) {
			Mention *m = sent->predict_loc_mentions[j];
			for (int k = m->start_offset; k <= m->end_offset; ++k) {
				string word = "Loc_"+sent->tokens[k]->lemma;

				if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;
				if (features.find(word) == features.end()) {
					features[word] = features.size();
				}
			}
		}
	}
}


void Document::BuildDocumentVec(map<string, int> &features) {
	doc_fvec.clear();
	for (int i = 0; i < SentNum(); ++i) {
		Sentence *sent = sentences[i];

/*		for (int j = 0; j < sent->TokenSize(); ++j) {
			string word = sent->tokens[j]->lemma;
			if (features.find(word) == features.end()) {
				features[word] = features.size();
			}
			int fid = features[word];
			if (doc_fvec.find(fid) == doc_fvec.end()) {
				doc_fvec[fid] = 1;
			} else {
				doc_fvec[fid] += 1;
			}
		}
*/
		for (int j = 0; j < sent->predict_mentions.size(); ++j) {
			Mention *m = sent->predict_mentions[j];
			for (int k = m->start_offset; k <= m->end_offset; ++k) {
				string word = sent->tokens[k]->lemma;

				if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

				if (features.find(word) == features.end()) {
					features[word] = features.size();
				}
				int fid = features[word];
				if (doc_fvec.find(fid) == doc_fvec.end()) {
					doc_fvec[fid] = 1;
				} else {
					doc_fvec[fid] += 1;
				}
			}
		}

		for (int j = 0; j < sent->predict_participant_mentions.size(); ++j) {
			Mention *m = sent->predict_participant_mentions[j];
			for (int k = m->start_offset; k <= m->end_offset; ++k) {
				string word = sent->tokens[k]->lemma;

				if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

				if (features.find(word) == features.end()) {
					features[word] = features.size();
				}
				int fid = features[word];
				if (doc_fvec.find(fid) == doc_fvec.end()) {
					doc_fvec[fid] = 1;
				} else {
					doc_fvec[fid] += 1;
				}
			}
		}

		for (int j = 0; j < sent->predict_time_mentions.size(); ++j) {
			Mention *m = sent->predict_time_mentions[j];
			for (int k = m->start_offset; k <= m->end_offset; ++k) {
				string word = sent->tokens[k]->lemma;

				if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

				if (features.find(word) == features.end()) {
					features[word] = features.size();
				}
				int fid = features[word];
				if (doc_fvec.find(fid) == doc_fvec.end()) {
					doc_fvec[fid] = 1;
				} else {
					doc_fvec[fid] += 1;
				}
			}
		}

		for (int j = 0; j < sent->predict_loc_mentions.size(); ++j) {
			Mention *m = sent->predict_loc_mentions[j];
			for (int k = m->start_offset; k <= m->end_offset; ++k) {
				string word = sent->tokens[k]->lemma;

				if (Dictionary::stopWords.find(word) != Dictionary::stopWords.end()) continue;

				if (features.find(word) == features.end()) {
					features[word] = features.size();
				}
				int fid = features[word];
				if (doc_fvec.find(fid) == doc_fvec.end()) {
					doc_fvec[fid] = 1;
				} else {
					doc_fvec[fid] += 1;
				}
			}
		}
	}

	doc_norm = 0.0;
	for (map<int, double>::iterator it = doc_fvec.begin(); it != doc_fvec.end(); ++it) {
		doc_norm += it->second * it->second;
	}
	doc_norm = sqrt(doc_norm);
}

void Document::BuildTFIDFMentionVec(Mention *m, map<int, float> &idf_map) {
/*	int fnum = 0;
	for (int *f = m->word_vec; *f != -1; ++f) {
		fnum++;
	}
	m->tfidf_vec = new ITEM[fnum + 1];

	fnum = 0;
	for (int *f = m->word_vec; *f != -1; ++f) {
		int w = *f;
		m->tfidf_vec[fnum].wid = w;
		m->tfidf_vec[fnum].weight = (float)idf_map[w];
		fnum++;
	}
	m->tfidf_vec[fnum].wid = -1;
	m->tfidf_vec[fnum].weight = 0;*/
}

bool Sentence::Overlap(Sentence *sent) {
	set<string> words;
	for (int i = 0; i < predict_mentions.size(); ++i) {
		words.insert(predict_mentions[i]->head_lemma);
	}
	for (int i = 0; i < sent->predict_mentions.size(); ++i) {
		string head = sent->predict_mentions[i]->head_lemma;
		if (words.find(head) != words.end()) return true;
	}
	return false;
}

void Sentence::AddEventMention(Mention *m) {
	if (m->start_offset == m->end_offset && Dictionary::DictionaryLookup(Dictionary::EventStopList, tokens[m->start_offset]->lemma)) return;
	m->AddMention(predict_mentions, false);
}

void Sentence::AddParticipantMention(Mention *m) {
	//if (m->start_offset == m->end_offset && Dictionary::DictionaryLookup(Dictionary::ParticipantStopList, tokens[m->start_offset]->lemma)) return;

	if (tokens[m->start_offset]->pos[0] == 'V') return;

//	int overlap = 0;
//	for (int i = 0; i < predict_participant_mentions.size(); ++i) {
//		if (m->Contain(predict_participant_mentions[i])) {
//			overlap++;
//		}
//	}
//	if (overlap >= 1) return;

	m->AddMention(predict_participant_mentions, false);
}

void Sentence::AddTimeMention(Mention *m) {
	m->AddMention(predict_time_mentions, false);
}

void Sentence::AddLocMention(Mention *m) {
	m->AddMention(predict_loc_mentions, false);
}
