#include <vector>
#include <map>
#include <boost/pending/disjoint_sets.hpp>
#include "um_loss.h"

double c_um_loss_gradient(
	size_t numNodes,
	const double* mst,
	const int64_t* gtSeg,
	double alpha,
	double* gradients,
	double* numPairsPos,
	double* numPairsNeg) {

	// labels and counts that each cluster overlaps with in gtSeg
	std::vector<std::map<int64_t, size_t>> overlaps(numNodes);

	// disjoint sets datastructure to keep track of cluster merging
	std::vector<size_t> rank(numNodes);
	std::vector<int64_t> parent(numNodes);
	boost::disjoint_sets<size_t*, int64_t*> clusters(&rank[0], &parent[0]);

	for (size_t i = 0; i < numNodes; i++) {

		// initially, every node is in its own cluster...
		clusters.make_set(i);

		// ...and overlaps only with one label (gtSeg[i])
		if (gtSeg[i] != 0)
			overlaps[i][gtSeg[i]] = 1;
	}

	// 1. Compute number of positive an negative pairs per edge.

	double totalNumPairsPos = 0.0;
	double totalNumPairsNeg = 0.0;

	for (size_t i = 0; i < numNodes - 1; i++) {

		int64_t u = mst[i*3];
		int64_t v = mst[i*3 + 1];
		int64_t clusterU = clusters.find_set(u);
		int64_t clusterV = clusters.find_set(v);

		assert(clusterU != clusterV);

		// link and make sure clusterU is the new root
		clusters.link(clusterU, clusterV);
		if (clusters.find_set(clusterU) == clusterV)
			std::swap(clusterU, clusterV);

		// find number of positive and negative pairs merged by (u, v)
		numPairsPos[i] = 0;
		numPairsNeg[i] = 0;
		for (const auto& overlapsU : overlaps[clusterU]) {
			for (const auto& overlapsV : overlaps[clusterV]) {

				int64_t labelU = overlapsU.first;
				int64_t labelV = overlapsV.first;
				double countU = overlapsU.second;
				double countV = overlapsV.second;

				if (labelU == labelV)
					numPairsPos[i] += countU*countV;
				else
					numPairsNeg[i] += countU*countV;
			}
		}

		// move all overlaps from v to u
		for (const auto& overlapsV : overlaps[clusterV]) {

			int64_t labelV = overlapsV.first;
			size_t countV = overlapsV.second;

			overlaps[clusterU][labelV] += countV;
		}
		overlaps[clusterV].clear();

		totalNumPairsPos += numPairsPos[i];
		totalNumPairsNeg += numPairsNeg[i];
	}

	// normalize number of pairs, this normalizes the loss and gradient
	for (size_t i = 0; i < numNodes - 1; i++) {

		if (totalNumPairsPos > 0)
			numPairsPos[i] /= totalNumPairsPos;
		if (totalNumPairsNeg > 0)
			numPairsNeg[i] /= totalNumPairsNeg;
	}

	// 2. Compute loss and first part of gradient

	double scoreA = 0;
	double scoreB = 0;
	double scoreC = 0;
	std::vector<double> scoresA(numNodes - 1);
	std::vector<double> scoresB(numNodes - 1);
	std::vector<double> scoresC(numNodes - 1);

	// trailing edge index, follows i such that
	// distance(j) < distance(i) - alpha
	size_t j = 0;
	for (size_t i = 0; i < numNodes - 1; i++) {

		double distance = mst[i*3 + 2];

		// update trailing edge's scores
		while (mst[j*3 + 2] /* distance(j) */ < distance /* distance(i) */ - alpha) {

			scoresA[j] = scoreA;
			scoresB[j] = scoreB;
			scoresC[j] = scoreC;
			j++;
		}

		// update running scores
		scoreA += numPairsNeg[i];
		scoreB += distance*numPairsNeg[i];
		scoreC += distance*distance*numPairsNeg[i];
	}

	// finish pending trailing edges
	for (; j < numNodes - 1; j++) {

		scoresA[j] = scoreA;
		scoresB[j] = scoreB;
		scoresC[j] = scoreC;
	}

	// compute loss
	double loss = 0;
	for (size_t i = 0; i < numNodes - 1; i++) {

		double distance = mst[i*3 + 2];

		loss +=
			numPairsPos[i]*(
				(distance*distance + 2*alpha*distance + alpha*alpha)*scoresA[i] +
				(-2*distance - 2*alpha)*scoresB[i] +
				scoresC[i]
			);
	}

	// for the gradient, we also need the scores summed downwards
	double scoreD = 0;
	double scoreE = 0;
	std::vector<double> scoresD(numNodes - 1);
	std::vector<double> scoresE(numNodes - 1);

	// trailing edge index, follows i such that
	// distance(j) > distance(i) + alpha
	j = numNodes - 2;
	for (size_t i = numNodes - 2;; i--) {

		double distance = mst[i*3 + 2];

		// update trailing edge's scores
		while (mst[j*3 + 2] /* distance(j) */ > distance /* distance(i) */ + alpha) {

			scoresD[j] = scoreD;
			scoresE[j] = scoreE;
			j--;
		}

		// update running scores
		scoreD += numPairsPos[i];
		scoreE += distance*numPairsPos[i];

		if (i == 0)
			break;
	}

	// finish pending trailing edges
	for (;; j--) {

		scoresD[j] = scoreD;
		scoresE[j] = scoreE;

		if (j == 0)
			break;
	}

	// finally, compute the gradients
	for (size_t i = 0; i < numNodes - 1; i++) {

		double distance = mst[i*3 + 2];

		gradients[i] =
			2*numPairsPos[i]*(
				(alpha + distance)*(scoresA[i] - numPairsNeg[i]) -
				(scoresB[i] - distance*numPairsNeg[i])
			) -
			2*numPairsNeg[i]*(
				(alpha - distance)*(scoresD[i] - numPairsPos[i]) +
				(scoresE[i] - distance*numPairsPos[i])
			);
	}

	return loss;
}
