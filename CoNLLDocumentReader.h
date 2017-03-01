/*
 * CoNLLDocumentReader.h
 *
 *  Created on: Sep 11, 2013
 *      Author: bishan
 */

#ifndef CONLLDOCUMENTREADER_H_
#define CONLLDOCUMENTREADER_H_

#include <string>
#include <vector>
#include <map>

#include "Document.h"

#include "./Parsing/Utils.h"

using namespace std;

enum FieldType { DOC_ID, PART_NO, WORD_NO, WORD, POS_TAG, PARSE_TAG, LEMMA,
	FRAMESET_ID, WORD_SENSE, SPEAKER_AUTHOR, NER_TAG, COREF_TAG};

#define FIELDS_MIN 12

class CoNLLDocumentReader {
public:
	CoNLLDocumentReader(string filename);
	virtual ~CoNLLDocumentReader();

	Document* ReadDocument();
	Document* ReadEvalDocument();

private:
	ifstream infile;
	bool ReadSentence(string line, int sent_id, Document *doc);
	bool ReadSentenceMentions(vector<string> mention_str, int sent_id, Document *doc);
};

#endif /* CONLLDOCUMENTREADER_H_ */
