#include <mlpack/methods/emst/dtb.hpp>

void mlpack_emst(
	int numPoints,
	int numDims,
	const double* points,
	double* mst) {

	arma::mat pointsMat(points, numDims, numPoints);
	arma::mat outputMat(mst, 3, numPoints - 1, /*copy_aux_mem*/false, /*strict*/true);

	mlpack::emst::DualTreeBoruvka<> dtb(pointsMat);
	dtb.ComputeMST(outputMat);
}
