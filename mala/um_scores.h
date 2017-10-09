#include <cstdint>

void c_um_scores(
	int numNodes,
	const double* emst,
	const float* dist,
	const float* distSquared,
	const int64_t* gtSeg,
	float alpha,
	float* numPairsNeg,
	float* numPairsPos,
	float* scoresA,
	float* scoresB,
	float* scoresC);
