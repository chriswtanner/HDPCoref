/*
 * CoNLLDocumentReader.cpp
 *
 *  Created on: Sep 11, 2013
 *      Author: bishan
 */

#include "CoNLLDocumentReader.h"
#include <fstream>
#include <assert.h>
#include <stack>

CoNLLDocumentReader::CoNLLDocumentReader(string filename) {
	// TODO Auto-generated constructor stub
	infile.open(filename.c_str(), ios::in);
}

CoNLLDocumentReader::~CoNLLDocumentReader() {
	// TODO Auto-generated destructor stub
}

Document* CoNLLDocumentReader::ReadDocument() {
	string line;
	while (getline(infile, line) && line.find("#begin document") == string::npos);
	if (line == "") return NULL;

	Document *doc = new Document();
	int start_index = line.find('(');
	int end_index = line.find(')');
	string doc_id = line.substr(start_index+1, end_index-start_index-1);

	doc->SetDocID(doc_id);
	vector<string> fields;
	Utils::Split(doc_id, '_', fields);
	string topicid = fields[0];
	int s = doc_id.find('e');
	int t = doc_id.find('.');
	string endfix = doc_id.substr(s,t-s);
	doc->SetTopicID(fields[0] + "_" + endfix);
	doc->topic = fields[0];

	// Read sentences.
	int sent_id = 0;
	string sent_str = "";
	while (getline(infile, line) && line.find("#end document") == string::npos) {
		if (line == "") {
			if(ReadSentence(sent_str, sent_id, doc)) {
				sent_id ++;
			}
			sent_str = "";
		} else {
			sent_str += line + "\n";
		}
	}

	return doc;
}

Document* CoNLLDocumentReader::ReadEvalDocument() {
	string line;
	while (getline(infile, line) && line.find("#begin document") == string::npos);
	if (line == "") return NULL;

	Document *doc = new Document();
	int start_index = line.find('(');
	int end_index = line.find(')');
	string doc_id = line.substr(start_index+1, end_index-start_index-1);

	doc->SetDocID(doc_id);
	vector<string> fields;
	Utils::Split(doc_id, '_', fields);
	string topicid = fields[0];
	int s = doc_id.find('e');
	int t = doc_id.find('.');
	string endfix = doc_id.substr(s,t-s);
	doc->SetTopicID(fields[0] + "_" + endfix);
	doc->topic = fields[0];

	// Read sentence mentions
	vector<string> lines;
	int prevsentid = -1;
	while (getline(infile, line) && line.find("#end document") == string::npos) {
		vector<string> fields;
		Utils::Split(line, '\t', fields);
		int cursentid = atoi(fields[0].c_str());
		// cout << cursentid << endl;
		if (prevsentid != -1 && cursentid != prevsentid) {
			ReadSentenceMentions(lines, prevsentid, doc);
			lines.clear();
		}
		lines.push_back(line);
		prevsentid = cursentid;
	}
	ReadSentenceMentions(lines, prevsentid, doc);

	// Build document level entities and mentions by merging entities from individual sentences.
	// (mentions are in the order they appear in text).
	doc->BuildDocumentMentions(true);
	doc->BuildDocumentMentions(false);
	doc->BuildDocumentEntities(true);
	doc->BuildDocumentEntities(false);

	return doc;
}

bool CoNLLDocumentReader::ReadSentenceMentions(vector<string> lines, int sent_id, Document *doc) {
	Sentence *sent = new Sentence();

	sent->SetDocID(doc->DocID());
	sent->SetSentenceID(sent_id);
	sent->sent_key = doc->DocID() + "\t" + Utils::int2string(sent_id);

	for (int i = 0; i < lines.size(); ++i) {
		vector<string> fields;
		Utils::Split(lines[i], '\t', fields);

		int length = fields.size();

		Mention *mention = new Mention();
		mention->SetSentenceID(sent->SentenceID());
		mention->SetDocID(sent->DocID());

		mention->SetStartOffset(atoi(fields[1].c_str()));
		mention->SetEndOffset(atoi(fields[2].c_str()));
		mention->SetMentionID(atoi(fields[3].c_str()));

		int gold_clusterID = atoi(fields[4].c_str());
		int pred_clusterID = atoi(fields[5].c_str());

		if (gold_clusterID == -1 || pred_clusterID == -1) continue;

		mention->gold_entity_id = gold_clusterID;
		mention->pred_entity_id = pred_clusterID;

		mention->twinless = false;

		if (sent->gold_entities.find(gold_clusterID) == sent->gold_entities.end()) {
			Entity *entity = new Entity();
			entity->SetEntityID(gold_clusterID);
			sent->gold_entities[gold_clusterID] = entity;
		}
		sent->gold_entities[gold_clusterID]->AddMention(mention);

		if (sent->predict_entities.find(pred_clusterID) == sent->predict_entities.end()) {
			Entity * pred_entity = new Entity();
			pred_entity->SetEntityID(pred_clusterID);
			sent->predict_entities[pred_clusterID] = pred_entity;
		}
		sent->predict_entities[pred_clusterID]->AddMention(mention);

		mention->AddMention(sent->gold_mentions, true);
		mention->AddMention(sent->predict_mentions, true);
	}

	sent->SortMentionsInLinearOrder(true);
	sent->SortMentionsInLinearOrder(false);

	doc->AddSentence(sent);
	return true;
}

bool CoNLLDocumentReader::ReadSentence(string sent_str, int sent_id, Document *doc) {
	Sentence *sent = new Sentence();
	sent->conll_str = sent_str;
	vector<string> lines;
	Utils::Split(sent_str, '\n', lines);

	sent->SetDocID(doc->DocID());
	sent->SetSentenceID(sent_id);
	sent->sent_key = doc->DocID() + "\t" + Utils::int2string(sent_id);

	vector<string> coref_tags;
	vector<string> entity_coref_tags;
	string parse_tree = "";
	for (int i = 0; i < lines.size(); ++i) {
		vector<string> fields;
		Utils::Split(lines[i], '\t', fields);
		if (fields.size() < FIELDS_MIN) {
			cout<<"sentence format wrong!!!"<<endl;
			cout<<lines[i]<<endl;
			return false;
		}
		Token *token = new Token();
		token->word_id = i;
		token->word = fields[WORD];
		if (token->word == "\\/") {
			token->word = "|";
		}
		// clean the word
		if (token->word.size() > 0 && (token->word[0] == '\"' || token->word[0] == '\''))
			token->word = token->word.substr(1);

		token->lowercase = Utils::toLower(token->word);
		token->pos = fields[POS_TAG];

		// lemma is always lower case!!!???
		string w = fields[LEMMA];
		// clean the word
		if (w.size() > 0 && (w[0] == '\"' || w[0] == '\''))
			w = w.substr(1);

		token->lemma = Utils::toLower(w);

		Dictionary::getLemma(token->word, token->pos, token->second_lemma, token->lemma_pos);

		sent->AddToken(token);

		// Read parse tree info.
		string parse_tag = fields[PARSE_TAG];
		int index = parse_tag.find('*');
		parse_tree += parse_tag.substr(0, index) +
				"(" + token->pos + "[" + token->word + "/" + token->pos + "]" + " " + token->word + ")"
				+ parse_tag.substr(index+1);

		// Read coref mentions.
		coref_tags.push_back(fields[fields.size()-1]);
	}

	// Build parse tree.
	sent->buildPennTree(parse_tree);

	// Recover coref clusters.
	doc->BuildCorefClusters(coref_tags, sent, EVENT);
	if (sent->EntitySize(true) == 0) {
		sent->valid = false;
	}

	// start_offset in increased order
	sent->SortMentionsInLinearOrder(true);
	doc->AddSentence(sent);
	return true;
}
