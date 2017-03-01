/*
 * Corpus.h
 *
 *  Created on: Sep 18, 2013
 *      Author: bishan
 */

#ifndef CORPUS_H_
#define CORPUS_H_

#include "CoNLLDocumentReader.h"
#include "Document.h"
#include "Logger.h"

#include "SievesCoref.h"

using namespace std;

class SystemOptions {
public:
	SystemOptions() : filter_singleton(false),
	filter_pronoun(false), strict_match(false),
	filter_not_annotated(true), filter_twinless(false) {};
public:
	bool filter_singleton;
	bool filter_pronoun;
	bool filter_not_annotated;
	bool strict_match;
	bool filter_twinless;
};

class CorefCorpus {
public:
	CorefCorpus() : max_mention_id(0) {}
	virtual ~CorefCorpus();

	void BuildEventMentionFeatures();

	void   BuildClusters(Document *doc, double threshold);
	double MentionSimilarity(Mention *m1, Mention *m2);
	double MentionGlobalSimilarity(Mention *m1, Mention *m2);
	double EntitySimilarity(Entity *e1, Entity *e2);
	bool ParseSwirlSrlArguments(string sent_str, Sentence *sent);

	void BuildVocabulary();
	void BuildMentionFeatures();

	void AddVerbalMentions();

	void OutputStats();

	void OutputCorefResults(string filename);
	void OutputSRLResults(string filename);

	void LoadWDSemEvalFiles(string filename);

	void OutputEvents(string filename);
	void OutputEntities(string filename);

	void OutputSentences(string outfilename);

	void FilterPredictMentions();

	void BuildGoldSentClusters();
	void BuildCorpusPredictEntitiesUsingGold();

	void NearestSentences(int topK);

	void OutputAntecedent(string filename);
	void LoadAntecedentInfo(string filename);

	void OutputPredictMentions(string filename);

	void ReadDocumentAnnotation(string filename, EntityType type);

	void OutputEventVocab(string filename);

	void LoadSentVec(string filename, string indexfilename);

	void OutputVocab(string filename);

	void LoadEvalData(string filenames);

	void LoadCoNLLData(vector<string> filenames);
	void OutputCoNLLData(string output_path, bool gold);

	void LoadPhrases(string indexfilename, string phrasefilename);
	void LoadParsingInfo(string sentidfile, string depparse);

	void LoadPredictMentions(string filename);

	// Set head index, mention id, and mention attributes.
	void SetGoldMentionInfo(string filename);
	void SetPredictedMentionInfo(string filename);
	void AddExtraPredictedMentions(string filename, string entity_type);
	void AddAllGoldMentions();

	// Corpus-level entities
	void BuildCorpusMentions(bool gold);
	void BuildCorpusEntities(bool gold);
	void RebuildDocumentEntities(bool gold);

	void MatchMentions(bool exact);
	void StrictMatchMentions();
	void RelaxedMatchMentions();
	void EvaluateMentionExtraction();

	void MatchParticipantMentions(bool exact);
	void StrictMatchParticipantMentions();
	void RelaxedMatchParticipantMentions();
	void EvaluateParticipantMentionExtraction();

	void MatchTimeMentions(bool exact);
	void StrictMatchTimeMentions();
	void RelaxedMatchTimeMentions();
	void EvaluateTimeMentionExtraction();

	void MatchLocMentions(bool exact);
	void StrictMatchLocMentions();
	void RelaxedMatchLocMentions();
	void EvaluateLocMentionExtraction();

	// LDA-related.
	void OutputLDAdata(string indexfilename, string datafilename, string wordmapfile);
	void GenLDAdata(string indexfilename, string datafilename, string wordmapfile);
	void LoadLDAResults(string indexfilename, string resultfilename, string wordmapfile);

	void GenHDPData(string datafilename);
	void LoadHDPResults(string filename);

	// Reset the coreference fields.
	void LoadCoNLLResults(string filename);

	void OutputCorpusEntities(string filename);
	void OutputClusterResults(string filename);
	void OutputVerbalMentions(string filename);

	void FilterMentions(SystemOptions option);
	void FilterSingletonMentions(bool gold);
	void FilterPronouns(bool gold);
	void FilterNotAnnotated();
	void FilterTwinless();

	string SentenceToSemevalStr(Sentence *sent, string topic_id, string part_id, bool gold);

	void OutputWDSemEvalFiles(string filename, bool gold);
	void OutputCDSemEvalFiles(string filename, bool gold);

	void PairwiseBaseline(string modelfile, string embeddingfile, double local_threshold, double global_threshold);

	void ReadClusterAssignments(vector<int> &cluster_assignments);

	// w/ SRL baseline.
	void MakeSwirlInput(string path);
	void RunSwirl(string path, string swirl_home);
	bool ParseSrlArguments(string sent_str, Sentence *sent);
	void ReadSwirlOutput(string path);

	void LoadDDCRPResults(string filename);

	void OutputDDCRPDistance(string path);
	void OutputDDCRPCorpus(string path);
	void OutputInitialClusters(string path);

	void OutputStatistics() {
		int mention_num = 0, entity_num = 0, predict_mention_num = 0,
				predict_entity_num = 0;
		for (map<string, Document*>::iterator it = topic_document.begin();
				it != topic_document.end(); ++it) {
			entity_num += it->second->gold_entities.size();
			predict_entity_num += it->second->predict_entities.size();
			for (map<int, Entity*>::iterator it1 =
					it->second->gold_entities.begin();
					it1 != it->second->gold_entities.end(); ++it1) {
				mention_num += it1->second->Size();
			}
			for (map<int, Entity*>::iterator it1 =
					it->second->predict_entities.begin();
					it1 != it->second->predict_entities.end(); ++it1) {
				predict_mention_num += it1->second->Size();
			}
		}
		cout << "mention num: " << mention_num << " cluster num: " << entity_num
				<< endl;
		cout << "pred mention num: " << predict_mention_num
				<< " pred cluster num: " << predict_entity_num << endl;
	}

	void OutputSwirlInfo(Logger &logger);

	void LoadMentionFeatures(string filename, bool gold);
	void ParseMentionFeatures(Document *doc, vector<string> fields, Mention *m);

	void loadMentionDistance(string filename);
	void loadMentionID(string filename);

	void DistanceBaseline();

	void SetupMentionDistance();
	void SetupMentionDict();

	void OutputMultinomial(string filename, bool gold);

	void LoadEntityClusters(vector<string> filenames, EntityType type);

	void DocumentClustering();

	// For Stanford baseline
	void OutputCDPredictMentions(string topic_doc_file, string outputfile);

public:
	map<string, Document *> documents;

	// hierarchical topic-documents
	map<string, vector<Document *> > topic_to_documents;

	// merge topic documents into one document
	map<string, Document*> topic_document;

	int max_mention_id; //unique in the corpus

	map<int, Mention*> mention_dict;

	// sentence vec
	map<string, int> sentence2idx; // title + sent_id
	vector<vector<double> > sentence_vecs;
	map<string, map<string, double> > sentence_pairwise_distance;

	// head features
	map<string, int*> synonym_features;
	map<string, int*> hypernym_features;
	map<string, int*> verbnet_features;
	map<string, int*> framenet_features;

	map<string, int> vocabulary;
	map<int, string> id2word;
	map<int, float> idf_map;

	// pairwise distance
	//map<int, map<int, double> > pairwise_distance;
	map<int, map<int, double> > global_pairwise_distance;

	map<string, vector<Document *> > predict_document_clusters;
};

#endif /* CORPUS_H_ */
