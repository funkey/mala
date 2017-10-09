#include <vector>
#include <map>
#include <boost/pending/disjoint_sets.hpp>
#include "um_scores.h"

void c_um_scores(
	int numNodes,
	const double* emst,
	const float* dist,
	const float* distSquared,
	const int64_t* gtSeg,
	float alpha,
	float* numPairsPos,
	float* numPairsNeg,
	float* scoresA,
	float* scoresB,
	float* scoresC) {

	// labels and counts that each cluster overlaps with in gtSeg
	std::vector<std::map<int64_t, int>> overlaps(numNodes);

	// disjoint sets datastructure to keep track of cluster merging
	std::vector<int> rank(numNodes);
	std::vector<int64_t> parent(numNodes);
	boost::disjoint_sets<int*, int64_t*> clusters(&rank[0], &parent[0]);

	for (int i = 0; i < numNodes; i++) {

		// initially, every node is in its own cluster...
		clusters.make_set(i);

		// ...and overlaps only with one label (gtSeg[i])
		if (gtSeg[i] != 0)
			overlaps[i][gtSeg[i]] = 1;
	}

	// trailing edge index, follows i such that
	// distance(j) < distance(i) - alpha
	int j = 0;

	double scoreA = 0;
	double scoreB = 0;
	double scoreC = 0;

	// for each edge in increasing order
	for (int i = 0; i < numNodes - 1; i++) {

		int64_t u = emst[i*3];
		int64_t v = emst[i*3 + 1];
		double distance = emst[i*3 + 2];

		int64_t clusterU = clusters.find_set(u);
		int64_t clusterV = clusters.find_set(v);

		assert(clusterU != clusterV);

		// link and make sure clusterU is the new root
		clusters.link(clusterU, clusterV);
		if (clusters.find_set(clusterU) == clusterV)
			std::swap(clusterU, clusterV);

		// update trailing edge's scores
		while (emst[j*3 + 2] /* distance(j) */ < distance /* distance(i) */ - alpha) {

			scoresA[j] = scoreA;
			scoresB[j] = scoreB;
			scoresC[j] = scoreC;
			j++;
		}

		// find number of positive and negative pairs merged by (u, v)
		numPairsPos[i] = 0;
		numPairsNeg[i] = 0;
		for (const auto& overlapsU : overlaps[clusterU]) {
			for (const auto& overlapsV : overlaps[clusterV]) {

				int64_t labelU = overlapsU.first;
				int64_t labelV = overlapsV.first;
				int countU = overlapsU.second;
				int countV = overlapsV.second;

				if (labelU == labelV)
					numPairsPos[i] += countU*countV;
				else
					numPairsNeg[i] += countU*countV;
			}
		}

		// update running scores
		scoreA += numPairsNeg[i];
		scoreB += distance*numPairsNeg[i];
		scoreC += distance*distance*numPairsNeg[i];

		// move all overlaps from v to u
		for (const auto& overlapsV : overlaps[clusterV]) {

			int64_t labelV = overlapsV.first;
			int countV = overlapsV.second;

			overlaps[clusterU][labelV] += countV;
		}
		overlaps[clusterV].clear();
	}

	// finish pending trailing edges
	for (; j < numNodes - 1; j++) {

		scoresA[j] = scoreA;
		scoresB[j] = scoreB;
		scoresC[j] = scoreC;
	}

	// Finally, the scores for edge i are the sums over all j with distance(j) <
	// distance(i) + alpha. Remove the contribution of j == i, such that the
	// scores of i have a non-zero gradient on distance(i). The self-comparison
	// case is handled elsewhere.
	for (j = 0; j < numNodes - 1; j++) {

		double distance = emst[j*3 + 2];

		scoresA[j] -= numPairsNeg[j];
		scoresB[j] -= distance*numPairsNeg[j];
		scoresC[j] -= distance*distance*numPairsNeg[j];
	}
}
