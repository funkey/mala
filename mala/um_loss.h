#include <cstdint>

double c_um_loss_gradient(
	int numNodes,
	const double* mst,
	const int64_t* gtSeg,
	double alpha,
	double* gradients,
	int64_t* numPairsPos,
	int64_t* numPairsNeg);
