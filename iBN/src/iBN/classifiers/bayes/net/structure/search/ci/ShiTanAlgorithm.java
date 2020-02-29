package iBN.classifiers.bayes.net.structure.search.ci;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import JavaMI.MutualInformation;
import JavaMI.ProbabilityState;
import iBN.classifiers.bayes.net.structure.search.local.HillClimberExt;
import iBN.classifiers.bayes.net.structure.search.local.MaximumWeightSpanningTree;
import iBN.core.InstancesHelper;
import iBN.core.ArrayHelper;
import iBN.core.BayesNetUtils;
import iBN.core.IncStatistics;
import iBN.core.InstancesSizeException;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.ParentSet;
import weka.classifiers.bayes.net.search.ci.CISearchAlgorithm;
import weka.classifiers.bayes.net.search.local.LocalScoreSearchAlgorithm;
import weka.classifiers.bayes.net.search.local.Scoreable;
import weka.core.Instances;
import weka.core.SelectedTag;

public class ShiTanAlgorithm extends CISearchAlgorithm {

	private static final long serialVersionUID = 1L;
	private Instances oldInstances;
	private Instances allInstances;

	/**
	 * Confidence level used to measure the degree of the association between nodes
	 */
//	private double m_alpha = 0.9;
	private double m_alpha = 0.99;
	
	private int n_parents = 100000;
//	private int n_parents = 1;

	/**
	 * Instance used to check the conditional independence between nodes
	 */
	private Instances instancesToIndependence;

	/**
	 * Structure to storage the maximum weight spanning tree built using the Chow
	 * and Liu's algorithm
	 */
	private BayesNet mWeightTree;

	/**
	 * Search algorithm used for learning the maximum weight spanning tree.
	 */
	private LocalScoreSearchAlgorithm searchAlgorithm = new MaximumWeightSpanningTree();

	private BayesNet m_bayesNet;

	List<Integer> lastParents = new ArrayList<Integer>();

	Cache m_Cache = null;
	
	@Override
	protected void search(BayesNet bayesNet, Instances instances) throws Exception {
		List<Integer> parents = new ArrayList<Integer>();
		// reorder instance
		instances = BayesNetUtils.reorderInstanceIfNedded(bayesNet, instances);
		addInstances(instances);
		this.m_bayesNet = new BayesNet();
		m_bayesNet.m_Instances = new Instances(instances);
		m_bayesNet.initStructure();
		initWeightTree();

		for (int iNode = 0; iNode < maxn(); iNode++) {
			ParentSet oParentSet = m_bayesNet.getParentSet(iNode);
			for (int iParent = 0; iParent < maxn(); iParent++) {
				instancesToIndependence = instances; // used on first independence test (line 5 - call on
														// isConditionalIndependent method)
				if (iNode != iParent && !isConditionalIndependent(iNode, iParent, new int[0], 0)) {
					if (oldInstances.equals(allInstances)) {// remember that every time that two nodes are independents,
															// set true on cache
						if (!hasSeparatorSetBetween(iNode, iParent, allInstances)) {
							if (!heuristicIND(iNode, iParent, bayesNet)) {
								parents.add(iParent);
								oParentSet.addParent(iParent, instances);
								m_Cache.setFalse(iNode, iParent);
								continue;
							}
						}
					} else if (!oldInstances.equals(allInstances)) {
						if (!m_Cache.get(iNode, iParent)) { // if are dependents on lasts
							if (!heuristicIND(iNode, iParent, bayesNet)) {
								parents.add(iParent);
								oParentSet.addParent(iParent, instances);
								m_Cache.setFalse(iNode, iParent);
								continue;
							}
						} else { // if exists independence on lasts
							if (!hasSeparatorSetBetween(iNode, iParent, allInstances)) { // if independence doesn't
																							// maintain on all instances
								if (!heuristicIND(iNode, iParent, bayesNet)) {
									parents.add(iParent);
									oParentSet.addParent(iParent, instances);
									m_Cache.setFalse(iNode, iParent);
									continue;
								}
							}
						}
					}
				}
				m_Cache.setTrue(iNode, iParent);
			}
		}

//		System.out.println(isDiffParents(parents));

		bayesNet = hillClimbingSearch(bayesNet);
		this.m_bayesNet = new BayesNet();
	}

	int isDiffParents(List<Integer> parents) {
//		if (lastParents.size() == 0) {
//			i = parents.size();
//		} else {
//			for (Integer parent : parents) {
//				if (!lastParents.contains(parent)) {
//					i++;
//				}
//			}
//		}
//		lastParents = parents;
//		return i;
		if (lastParents.size() != 0) {
			for (Integer parent : parents) {
				if (!lastParents.contains(parent)) {
					lastParents = parents;
					return 1;
				}
			}
		}
		lastParents = parents;
		return 0;
	}

	private BayesNet hillClimbingSearch(BayesNet bayesNet) throws Exception {
		HillClimberExt climber = new HillClimberExt();
		climber.setUseArcReversal(true);
		climber.setInitAsNaiveBayes(false);
		climber.setM_bayesNet(m_bayesNet);
		climber.setScoreType(new SelectedTag(Scoreable.BAYES, TAGS_SCORE_TYPE));
		climber.setMaxNrOfParents(n_parents); // By setting it to a value much larger than the number of nodes in the
											// network (the default of 100000 pretty much guarantees this), no
											// restriction on the number of parents is enforced
		bayesNet.setSearchAlgorithm(climber);
		bayesNet.m_Instances = allInstances;
		bayesNet.buildStructure();
		return bayesNet;
	}

	private boolean heuristicIND(int iNode, int iParent, BayesNet bayesNet) throws Exception {
		HashSet<Integer> neighborsiNode = new HashSet<Integer>();
		HashSet<Integer> neighborsiParent = new HashSet<Integer>();

		BayesNetUtils.neighborsOnPath(BayesNetUtils.revert(bayesNet), iNode, iParent);
		neighborsiNode.addAll(new HashSet<Integer>(BayesNetUtils.iHeadNeighbors));
		neighborsiParent.addAll(new HashSet<Integer>(BayesNetUtils.iTailNeighbors));

		// BayesNetUtils.neighborsOnPath(mWeightTree, indexWeightTree.get(iNode),
		// indexWeightTree.get(iParent));
		BayesNetUtils.neighborsOnPath(mWeightTree, iNode, iParent);
		neighborsiNode.addAll(new HashSet<Integer>(BayesNetUtils.iHeadNeighbors));
		neighborsiParent.addAll(new HashSet<Integer>(BayesNetUtils.iTailNeighbors));

		List<HashSet<Integer>> allNeighbors = new ArrayList<HashSet<Integer>>();
		allNeighbors.add(neighborsiNode);
		allNeighbors.add(neighborsiParent);

		for (HashSet<Integer> neighbors : allNeighbors) {
			if (infoChi(iNode, iParent, neighbors.toArray(new Integer[neighbors.size()])) <= 0) {
				return true;
			}
			while (neighbors.size() > 1) {
				int m = Integer.MAX_VALUE;
				double s_m = Double.MAX_VALUE;
				double s = Double.MAX_VALUE;
				for (int i = 0; i < neighbors.size(); i++) {
					List<Integer> neighborsList = new ArrayList<Integer>(neighbors);
					neighborsList.remove(i);
					double s_i = infoChi(iNode, iParent, neighborsList.toArray(new Integer[neighborsList.size()]));
					if (s_i < s_m) {
						s_m = s_i;
						m = i;
					}
				}
				if (s_m <= 0.0) {
					return true;
				} else if (s_m > s) {
					break;
				} else {
					s = s_m;
					List<Integer> neighborsList = new ArrayList<Integer>(neighbors);
					neighborsList.remove(m);
					neighbors = new HashSet<Integer>(neighborsList);
				}
			}
		}
		return false;
	}

	/**
	 * 
	 * @param iNode
	 * @param iParent
	 * @param iNeighborsNodes
	 * @return
	 */
	private double infoChi(int iNode, int iParent, Integer[] iNeighborsNodes) {
		double[] iNodeValues = allInstances.attributeToDoubleArray(iNode); // start repetition

		int nConfigNode = allInstances.attribute(iNode).numValues();

		double[] iParentValues = allInstances.attributeToDoubleArray(iParent);

		int nConfigParent = allInstances.attribute(iParent).numValues();

		if (iNeighborsNodes.length != 0) {
			double[] iMiddleValues = allInstances.attributeToDoubleArray(iNeighborsNodes[0]);
			int nConfigMiddle = allInstances.attribute(iNeighborsNodes[0]).numValues();

			if (iNeighborsNodes.length > 1) {
				for (int i = 1; i < iNeighborsNodes.length; i++) {
					double[] mergedValues = new double[iMiddleValues.length];
					double[] iMergeMiddleValues = allInstances.attributeToDoubleArray(iNeighborsNodes[i]);
					int nMergeConfigMiddle = allInstances.attribute(iNeighborsNodes[i]).numValues();
					ProbabilityState.mergeArrays(iMiddleValues, iMergeMiddleValues, mergedValues);
					iMiddleValues = mergedValues;
					nConfigMiddle = (nConfigMiddle + nMergeConfigMiddle) / 2;
				}
			} // end repetition

			return 2 * allInstances.numInstances()
					* MutualInformation.calculateConditionalMutualInformation(iNodeValues, iParentValues, iMiddleValues)
					- IncStatistics.upperTailofChiSquaredDistribution(m_alpha,
							(nConfigNode - 1) * (nConfigParent - 1) * nConfigMiddle);
		}

		return 2 * allInstances.numInstances()
				* MutualInformation.calculateMutualInformation(iNodeValues, iParentValues)
				- IncStatistics.upperTailofChiSquaredDistribution(m_alpha, (nConfigNode - 1) * (nConfigParent - 1));
	}

	private boolean hasSeparatorSetBetween(int iNode, int iParent, Instances instances) {
		instancesToIndependence = instances;
		int[] iMiddle = new int[maxn() - 2]; // iMiddle will contain all nodes except iNode and iParent
		int position = 0;
		for (int i = 0; i < maxn(); i++) {
			if (i != iNode && i != iParent) {
				iMiddle[position] = i;
				position++;
			}
		}

		int n = iMiddle.length;

		// This code will find all possible iMiddle subsets to verify the independence
		// between the nodes
		// System.out.println(iNode + " : " + iParent);
		for (int i = 0; i < (1 << n); i++) {
			ArrayList<Integer> iMiddleList = new ArrayList<>();
			int m = 1;
			for (int j = 0; j < n; j++) {
				if ((i & m) > 0) {
					iMiddleList.add(iMiddle[j]);
				}
				m = m << 1;
			}
			int[] iMiddleNodes = ArrayHelper.convertArray(iMiddleList);
//			System.out.print("{");
//			for (int j : iMiddleNodes) {
//				System.out.print(j + " ");
//			}
//			System.out.println("}");
			if (isConditionalIndependent(iNode, iParent, iMiddleNodes, iMiddleNodes.length)) {
				return true;
			}
		}
		return false;
	}

	@Override
	protected boolean isConditionalIndependent(int iNode, int iParent, int[] iMiddleNodes, int nAttributesZ) {
		double[] iNodeValues = instancesToIndependence.attributeToDoubleArray(iNode);

		int nConfigNode = instancesToIndependence.attribute(iNode).numValues();

		double[] iParentValues = instancesToIndependence.attributeToDoubleArray(iParent);

		int nConfigParent = instancesToIndependence.attribute(iParent).numValues();

		if (iMiddleNodes.length != 0) {
			double[] iMiddleValues = instancesToIndependence.attributeToDoubleArray(iMiddleNodes[0]);
			int nConfigMiddle = instancesToIndependence.attribute(iMiddleNodes[0]).numValues();

			if (iMiddleNodes.length > 1) {
				for (int i = 1; i < iMiddleNodes.length; i++) {
					double[] mergedValues = new double[iMiddleValues.length];
					double[] iMergeMiddleValues = instancesToIndependence.attributeToDoubleArray(iMiddleNodes[i]);
					int nMergeConfigMiddle = instancesToIndependence.attribute(iMiddleNodes[i]).numValues();
					ProbabilityState.mergeArrays(iMiddleValues, iMergeMiddleValues, mergedValues);
					iMiddleValues = mergedValues;
					nConfigMiddle = (nConfigMiddle + nMergeConfigMiddle) / 2;
				}
			}

			return 2 * instancesToIndependence.numInstances()
					* MutualInformation.calculateConditionalMutualInformation(iNodeValues, iParentValues,
							iMiddleValues) <= IncStatistics.upperTailofChiSquaredDistribution(m_alpha,
									(nConfigNode - 1) * (nConfigParent - 1) * nConfigMiddle);
		}

		return 2 * instancesToIndependence.numInstances()
				* MutualInformation.calculateMutualInformation(iNodeValues, iParentValues) <= IncStatistics
						.upperTailofChiSquaredDistribution(m_alpha, (nConfigNode - 1) * (nConfigParent - 1));
	}

	/**
	 * Sets whether to init as naive bayes
	 * 
	 * @param bInitAsNaiveBayes whether to init as naive bayes
	 */
	public void setInitAsNaiveBayes(boolean bInitAsNaiveBayes) {
		m_bInitAsNaiveBayes = bInitAsNaiveBayes;
	}

	/**
	 * Sets the confidence level used to measure the degree of the association
	 * between two variable
	 * 
	 * @param alpha confidence level
	 */
	public void setAlpha(double alpha) {
		m_alpha = alpha;
	}

	/**
	 * Gets whether to init as naive bayes
	 *
	 * @return whether to init as naive bayes
	 */
	public boolean getInitAsNaiveBayes() {
		return m_bInitAsNaiveBayes;
	}

	/**
	 * returns the number of attributes
	 * 
	 * @return the number of attributes
	 */
	private int maxn() {
		return allInstances.numAttributes();
	}

	private void addInstances(Instances instances) throws Exception {
		if (allInstances == null && oldInstances == null) {
			allInstances = instances;
			oldInstances = instances;
			initCache();
			return;
		}
		if (instances.numAttributes() != maxn()) {
			throw new InstancesSizeException("Different number of attributes among instances.");
		}
		oldInstances = new Instances(allInstances);
		allInstances.addAll(instances);
	}

	/**
	 * cache for remembering the independence conditional among nodes for
	 * incremental steps
	 */
	class Cache {

		/** change in parent nodes due to conditional independence **/
		boolean[][] independentsNodes;

		/**
		 * c'tor
		 * 
		 * @param nNrOfNodes number of nodes in network, used to determine memory size
		 *                   to reserve
		 */
		Cache(int nNrOfNodes) {
			independentsNodes = new boolean[nNrOfNodes][nNrOfNodes];
			setInitialValues(nNrOfNodes);
		}

		/**
		 * set true to cache entry for the same nodes
		 * 
		 * @param node - checked start node independentParent - independent end node
		 * @return cache value
		 */
		private void setInitialValues(int nNrOfNodes) {
			for (int iNode = 0; iNode < nNrOfNodes; iNode++) {
				setTrue(iNode, iNode);
			}
		}

		/**
		 * set cache entry
		 * 
		 * @param node - checked start node independentParent - independent end node
		 * @return cache value
		 */
		public void setTrue(int node, int independentParent) {
			independentsNodes[node][independentParent] = true;
		} // set true

		/**
		 * set cache entry
		 * 
		 * @param node - checked start node independentParent - independent end node
		 * @return cache value
		 */
		public void setFalse(int node, int independentParent) {
			independentsNodes[node][independentParent] = false;
		} // set true

		/**
		 * get cache entry
		 * 
		 * @param node - start node to check the independence independentParent - end
		 *             node to check the independence
		 * @return cache value
		 */
		public boolean get(int node, int independentParent) {
			return independentsNodes[node][independentParent];
		} // get
	} // class Cache

	/**
	 * initCache initializes the cache
	 * 
	 * @param bayesNet  Bayes network to be learned
	 * @param instances data set to learn from
	 * @throws Exception if something goes wrong
	 */
	private void initCache() throws Exception {
		m_Cache = new Cache(maxn());
	} // initCache

	/**
	 * Method to generate the maximum weight spanning tree
	 * 
	 * @throws Exception
	 */
	private void initWeightTree() throws Exception {
		mWeightTree = new BayesNet();
		searchAlgorithm.buildStructure(mWeightTree, allInstances);
	}
}
