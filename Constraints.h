/*
 * Constraints.h
 *
 *  Created on: Oct 1, 2013
 *      Author: bishan
 */

#ifndef CONSTRAINTS_H_
#define CONSTRAINTS_H_

#include "Entity.h"
#include "Logger.h"
#include "./Parsing/Dictionary.h"

class Constraints {
public:
	Constraints();
	virtual ~Constraints();

	static bool MentionTypeAgreement(Mention *m1, Mention *m2);

	static bool EntityTypesAgree(Mention *m1, Mention *m2);
	static int NumberAgreement(Mention *m1, Mention *m2);
	static int GenderAgreement(Mention *m1, Mention *m2);
	static int AnimateAgreement(Mention *m1, Mention *m2);
	static int PersonAgreement(Mention *m1, Mention *m2);

	static bool DependencyRoleAgreement(Mention *m1, Mention *m2);

	static bool EntityDisagreement(Mention *m1, Mention *m2);
	static bool EntityAgreement(Mention *m1, Mention *m2);

	static int DependencyAgreement(Mention *m1, Mention *m2);

	static double PairwiseDistance(Mention *m1, Mention *m2);

};

#endif /* CONSTRAINTS_H_ */
