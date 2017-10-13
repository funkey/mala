#include <cstdint>

double c_um_loss_gradient(
	size_t numNodes,
	const double* mst,
	const int64_t* gtSeg,
	double alpha,
	double* gradients,
	double* ratioPos,
	double* ratioNeg,
	double& totalNumPairsPos,
	double& totalNumPairsNeg);
