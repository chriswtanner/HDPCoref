/*
 * PairwiseModel.h
 *
 *  Created on: Nov 9, 2014
 *      Author: bishan
 */

#ifndef PAIRWISEMODEL_H_
#define PAIRWISEMODEL_H_
#include "./LogLinear/encoder.h"
#include "./LogLinear/tagger.h"
#include "Constraints.h"
#include "CorefCorpus.h"

using namespace CRFPP;

class PairwiseModel {
public:
	CRFPP::Encoder encoder;

	float               **word_embeddings;
	int                 embed_dim;
	map<string, int>    word_lookup_table;

	std::vector<TaggerImpl* > labelx;

	// for predictor
	CRFPP::DecoderFeatureIndex *decoder_feature_index;
	std::vector<TaggerImpl* > devx;

	double singleton_local_prob;
	double singleton_global_prob;

public:
	PairwiseModel();
	virtual ~PairwiseModel();

	void BuildCDPairwiseEncoder(CorefCorpus *data, bool train);

	void LoadLRInstance(string line, int &ans, map<string, float> &fvec);

	void LoadModel(string modelfile);
	void Predict(string lrfile, string outputfile, bool eval);

	void calculateAcc(vector<int> &true_labels, vector<int> &pred_labels);

	double VecNorm(int *fvec);
	void GetSentPairFeatures(Sentence *s1, Sentence *s2, map<string, float>&fvec);

	double GetBOWSim(ITEM *fvec1, ITEM *fvec2, double norm1, double norm2);

	double GetFeatureMapSim(map<int, int> &fvector1, map<int, int> &fvector2);
	double GetFeatureVecSim(int* features1, int* features2);

	int  MentionGroupAlign(vector<Mention*> &g1, vector<Mention*> &g2);

	void LeftFeatures(Mention *m1, Sentence *s1, Mention *m2, Sentence *s2, map<string, float> &fvec);
	void RightFeatures(Mention *m1, Sentence *s1, Mention *m2, Sentence *s2, map<string, float> &fvec);

	void LoadEmbeddings(string embedfile);
	void ClearEmbeddings();

	double L2distance(vector<double> &v1, vector<double> &v2);

	void GenEntityPairEmbeddingFeatures(Sentence *s1, Mention *m1, Sentence *s2, Mention *m2, map<string, float> &fvec);
	void GenEventPairEmbeddingFeatures(Mention *m1, Sentence *s1, Mention *m2, Sentence *s2, map<string, float> &fvec);

	bool GetHeadEmbedding(Sentence *s, Mention *m, vector<double> &embed);
	bool GetPhraseEmbedding(Sentence *s, int start, int end, vector<double> &embed);

	void GenEmbeddingFeatures(string word, map<string, float> &fvec);

	string MentionTypePair(Mention *m1, Mention *m2);

	void GenSingletonFeatures(Sentence *sent, Mention *m, map<string, float> &fvec);

	void GenCDEventPairFeatures(Entity *e1, Entity *e2, map<string, float> &fvec);

	void GenWDEventPairFeatures(Mention *m1, Mention *m2, map<string, float> &fvec);
	void GenCDEventPairFeatures(Mention *m1, Mention *m2, map<string, float> &fvec);

	void GenPairwiseEntityFeatures(Mention *m1, Mention *m2, map<string, float> &fvec);

	void MergeFeatureVec(int* fvector1, int* fvector2, string key, map<string, float> &fvec);

	void SRLArgumentFeatures(Mention *m1, Mention *m2, map<string, float> &fvec);
	void DepArgumentFeatures(Mention *m1, Mention *m2, map<string, float> &fvec);
	void SRLPredicateFeatures(Mention *m1, Mention *m2, map<string, float> &fvec);

	void TrainModel(string lrfile, string modelfile);

	void OutputLRData(EncoderFeatureIndex* feature_index, vector<CRFPP::TaggerImpl*> data, string outputfilename);
	void OutputMaxEntData(vector<CRFPP::TaggerImpl*> data, string outputfilename);
};

#endif /* PAIRWISEMODEL_H_ */
