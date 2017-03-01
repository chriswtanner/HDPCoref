#include "CoNLLDocumentReader.h"
#include "CorefEval.h"
#include "CorefCorpus.h"
#include "Document.h"
#include "DataStatistics.h"
#include "Logger.h"

#include "./DDCRF/ddcrp_utils.h"
#include "./DDCRF/hdp_state.h"
#include "./DDCRF/hdp.h"
#include "PairwiseModel.h"

#include "SystemConfig.h"

#include <cstdio>
#include <fstream>
#include <set>

#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

using namespace DDCRP;
using namespace std;

gsl_rng * RANDOM_NUMBER;

void LoadTopicIDs(string filename, vector<string> &topics, int K) {
	topics.clear();
	ifstream infile(filename.c_str(), ios::in);
	string line;
	while(getline(infile, line)) {
		topics.push_back(line);
		if (K > 0 && topics.size() == K) break;
	}
	infile.close();
}

CorefCorpus* SelectDocuments(vector<string> topics, CorefCorpus *data, bool train) {
	set<string> topic_dict;
	for (int i = 0; i < topics.size(); ++i) {
		topic_dict.insert(topics[i]);
		//if (i == 1) break;
	}
	CorefCorpus *new_data = new CorefCorpus();
	for (map<string, Document*>::iterator it = data->documents.begin();
			it != data->documents.end(); ++it) {
		string tid = it->second->topic;
		if (topic_dict.find(tid) == topic_dict.end()) continue;
		new_data->documents[it->first] = it->second;
	}
	for (map<string, vector<Document*> >::iterator it = data->topic_to_documents.begin();
			it != data->topic_to_documents.end(); ++it) {
		int index = it->first.find('_');
		string topic = it->first.substr(0, index);
		if (topic_dict.find(topic) == topic_dict.end()) continue;
		new_data->topic_to_documents[it->first] = it->second;
	}
	for (map<string, Document* >::iterator it = data->topic_document.begin();
			it != data->topic_document.end(); ++it) {
		int index = it->first.find('_');
		string topic = it->first.substr(0, index);
		if (topic_dict.find(topic) == topic_dict.end()) continue;
		new_data->topic_document[it->first] = it->second;
	}
	return new_data;
}

void buildDistanceFunc(PairwiseModel &model, CorefCorpus* data, hdp_state *m_state, hdp_hyperparameter * hdp_hyperparam) {
	model.singleton_local_prob = hdp_hyperparam->m_alpha_a;

	CRFPP::TaggerImpl* x = new CRFPP::TaggerImpl();
	for (int i = 0; i < m_state->m_num_docs; ++i) {
		doc_state *m_doc_state = m_state->m_doc_states[i];
		//cout<<"process doc "<<i<<endl;
		for (int j = 0; j < m_doc_state->m_doc_length; ++j) {
			Elem *m_men = m_doc_state->m_words[j];

			Mention *m = data->mention_dict[m_men->mention_id];
			Sentence *sent = data->documents[m->doc_id]->GetSentence(m->sent_id);

			m_men->singleton_local_prob = model.singleton_local_prob;
			m_men->gold_entity_id = m->gold_entity_id;

			m_men->ant_local_cands.clear();
			// within document ant sentences
			for (int j1 = 0; j1 < j; ++j1) {
				Elem *ant_men = m_doc_state->m_words[j1];
				Mention *ant_m = data->mention_dict[ant_men->mention_id];
				Sentence *ant_sent = data->documents[ant_m->doc_id]->GetSentence(ant_m->sent_id);

				map<string, float> fvec;
				model.GenCDEventPairFeatures(m, ant_m, fvec);
				model.GenEventPairEmbeddingFeatures(m, sent, ant_m, ant_sent, fvec);

				x->clear();
				x->x = ant_m->mention_id;

				double dist = 0.0;
				int y = -1;
				if (m->gold_entity_id != -1 && ant_m->gold_entity_id != -1) {
					if (m->gold_entity_id == ant_m->gold_entity_id) y = 1;
					else y = 0;
				}

				x->MentionPairToTagger(fvec, y, model.decoder_feature_index,  false);
				x->LRInference();

				x->prob_ = 0.0;
				if (ant_men->word_str == m_men->word_str) {
					x->prob_ = 1.0;
				}
				if (x->node_[0]->prob > 0.5) {
					x->prob_ = x->node_[0]->prob;
				}
				dist = x->prob_;

				m_men->local_dist.push_back(dist);
				m_men->ant_local_cands.push_back(ant_men);
			}
		}
	}

	delete x;
	x = NULL;
}

void DDCRF_run(hdp_hyperparameter * hdp_hyperparam, CorefCorpus *data) {
	int nd = 0, nw = 0;

	DDCRP::ddcrf * hdp_instance = new ddcrf();

	vector<Document*> coref_docs;

	for (map<string, Document*>::iterator it = data->documents.begin(); it != data->documents.end(); ++it) {
		coref_docs.push_back(it->second);
	}

	hdp_instance->setup_state(hdp_hyperparam->eta, hdp_hyperparam->fea_eta, hdp_hyperparam->init_topics, hdp_hyperparam);
	hdp_instance->allocate_docs(data, coref_docs);

	printf("number of docs  : %d\n", hdp_instance->m_state->m_num_docs);
	printf("number of terms : %d\n", hdp_instance->m_state->m_size_vocab);
	printf("number of total words : %d\n", hdp_instance->m_state->m_total_num_words);

	PairwiseModel model;
	model.LoadEmbeddings(hdp_hyperparam->embeddingfile);
	model.LoadModel(hdp_hyperparam->local_model_file);
	buildDistanceFunc(model, data, hdp_instance->m_state, hdp_hyperparam);

	cout<<"start running for corpus"<<endl;
	char buffer[128];
	snprintf(buffer, sizeof(buffer), "%s", hdp_hyperparam->output_path.c_str());

	hdp_instance->run(buffer, hdp_hyperparam->burnin);
	cout << "finished running hddcrp" << endl;
	//hdp_instance->m_state->hard_em();

	// Output results
	for (int d = 0; d < hdp_instance->m_state->m_num_docs; ++d) {
		doc_state *m_doc_state = hdp_instance->m_state->m_doc_states[d];

		for (int n = 0; n < m_doc_state->m_doc_length; n++) {
			int mid = m_doc_state->m_words[n]->mention_id;
			//int t = m_doc_state->m_words[n]->best_c_value;
			//data->mention_dict[mid]->pred_entity_id = m_doc_state->m_tables[t]->best_c_value;
			int t = m_doc_state->m_words[n]->c_value;
			data->mention_dict[mid]->pred_entity_id = m_doc_state->m_tables[t]->c_value;
		}
	}

	// free resources
	delete hdp_instance;

	delete hdp_hyperparam;
}

int main(int argc, char** argv) {
	Config props(argv[1]);

 	// allocate the random number structure
	time_t t;
	time(&t);
	long seed = (long) t;
	//cout<<(long)seed<<endl;

 	// for debugging
 	//long seed = 1460348839;

	RANDOM_NUMBER = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(RANDOM_NUMBER, (long) seed); // init the seed

	SystemOptions option;

	CorefCorpus data;

	// Load gold event clusters
	vector<string> filenames;
	filenames.push_back(props.GetProperty("conll_file"));
	data.LoadCoNLLData(filenames);

	cout<<"Number of topics :"<<data.topic_to_documents.size()<<endl;
	cout<<"Number of documents:"<<data.documents.size()<<endl;
	data.BuildCorpusEntities(true);
	// event gold mention info
	data.SetGoldMentionInfo(props.GetProperty("event_gold_mention_table"));
	data.BuildCorpusMentions(true);
	int ngold_mentions = data.max_mention_id;
	cout<<"Number of gold mentions: "<<data.max_mention_id<<endl;
	// -----------------

	// Load predicted event mentions
	/*
	data.LoadPredictMentions(props.GetProperty("event_pred_mentions"));
	data.ReadSwirlOutput(props.GetProperty("swirl_path"));
	data.SetPredictedMentionInfo(props.GetProperty("event_pred_mention_table"));
	data.AddVerbalMentions();
	*/
	data.BuildCorpusPredictEntitiesUsingGold(); // NOTE< I ADDED THIS!

	if (props.GetBoolProperty("use_gold_boundaries")) {
		data.AddAllGoldMentions();
	}

	// this boolean simply adds our Mentions to our
	// predicted or golden data structure; it should always be 'false' here
	data.BuildCorpusMentions(false); 

	cout<<"Number of predicted mentions: "<<data.max_mention_id-ngold_mentions<<endl;

//	vector<string> train_topics;
//	LoadTopicIDs(props.GetProperty("train_topics"), train_topics, -1);
//	CorefCorpus *train_corpus = SelectDocuments(train_topics, &data, true);
//
//	vector<string> dev_topics;
//	LoadTopicIDs(props.GetProperty("dev_topics"), dev_topics, -1);
//	CorefCorpus *dev_corpus = SelectDocuments(dev_topics, &data, false);

	vector<string> test_topics;
	LoadTopicIDs(props.GetProperty("test_topics"), test_topics, props.GetIntProperty("topic_K"));
	CorefCorpus *test_corpus = SelectDocuments(test_topics, &data, false);


	// ============= Mention Extraction (headword match) ==============
	option.strict_match = false;
	data.MatchMentions(option.strict_match);

	data.BuildVocabulary();
	data.BuildMentionFeatures();

	// inference
	// load event features
	test_corpus->BuildEventMentionFeatures();

	string result_path = props.GetProperty("result_path");

	test_corpus->SetupMentionDict();

	//test_corpus->DocumentClustering();

	string model_type = props.GetProperty("model");
	double gamma_a = props.GetDoubleProperty("gamma"), gamma_b = 1.0,
			alpha_a = props.GetDoubleProperty("alpha"), alpha_b = 1.0,
			eta = props.GetDoubleProperty("eta"), fea_eta = props.GetDoubleProperty("fea_eta");
	int max_iter = props.GetIntProperty("max_iter"), save_lag = 100, init_topics = 0;
	int distfun_iter = props.GetIntProperty("distfun_iter");
	bool init_cluster = props.GetBoolProperty("init_cluster");

	int burn_in = props.GetIntProperty("burn_in");
	bool structured_info = props.GetBoolProperty("structured_info");

	hdp_hyperparameter * hdp_hyperparam = new hdp_hyperparameter();
	hdp_hyperparam->setup_parameters(model_type, props.GetProperty("pairwise_model"),
									props.GetProperty("embeddingfile"),
									gamma_a, gamma_b,
									alpha_a, alpha_b,
									eta, fea_eta, init_topics,
									max_iter, save_lag,
									props.GetProperty("distfun"), props.GetBoolProperty("init_distfun"), distfun_iter,
									init_cluster,
									structured_info,
									burn_in);

	hdp_hyperparam->output_path = props.GetProperty("ddcrf_output_path");

	DDCRF_run(hdp_hyperparam, test_corpus);

	CorefCorpus *eval_corpus = test_corpus;
	// ========== Evaluation ==============
	if (props.GetBoolProperty("evaluation")) {
		// collect results
		option.filter_not_annotated = true;
		option.filter_pronoun = props.GetBoolProperty("filter_pronoun");
		option.filter_twinless = props.GetBoolProperty("filter_twinless");
		option.filter_singleton = props.GetBoolProperty("filter_singleton");

		CorefEval eval;
		eval_corpus->FilterMentions(option);

		eval.corpus = eval_corpus;
		eval.corpus->BuildCorpusEntities(true);
		eval.corpus->BuildCorpusEntities(false);

		eval.corpus->OutputStatistics();

		eval.outputTopicScore(result_path+"/topic.coref.score.txt");

		eval.doCDScore();
		eval.printAccumulateScore();

		eval.doWDScore();
		eval.printAccumulateScore();

		eval_corpus->OutputWDSemEvalFiles(result_path+"/gold.WD.semeval.txt", true);
		eval_corpus->OutputWDSemEvalFiles(result_path+"/predict.WD.semeval.txt", false);

		eval_corpus->OutputCDSemEvalFiles(result_path+"/gold.CD.semeval.txt", true);
		eval_corpus->OutputCDSemEvalFiles(result_path+"/predict.CD.semeval.txt", false);
	}

	gsl_rng_free(RANDOM_NUMBER);


	return 0;
}
