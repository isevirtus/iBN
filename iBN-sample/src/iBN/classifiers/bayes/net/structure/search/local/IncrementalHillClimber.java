package iBN.classifiers.bayes.net.structure.search.local;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;

import iBN.core.BayesNetUtils;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.ParentSet;
import weka.classifiers.bayes.net.search.local.LocalScoreSearchAlgorithm;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

/**
 * This Bayes Network learning algorithm uses a incremental hill climbing
 * algorithm adding, deleting and reversing arcs. The search is not restricted
 * by an order on the variables (unlike K2). The difference with B and B2 is
 * that this hill climber also considers arrows part of the naive Bayes
 * structure for deletion.
 * 
 * @author Luiz Pereira (luizantonio@copin.ufcg.edu.br)
 * @version $Revision: 0 $
 */
public class IncrementalHillClimber extends LocalScoreSearchAlgorithm {

	/** for serialization */
	private static final long serialVersionUID = 6277560184070695032L;

	/** cache for storing score differences **/
	Cache m_Cache = null;

	/** use the arc reversal operator **/
	boolean m_bUseArcReversal = false;

	/** list of sequence of operations used to construct the final model **/
	List<Operation> m_searchPath = new ArrayList<Operation>();

	/**
	 * list of operations used to construct the neighbourhood of each model present
	 * on search path (each operation on search path is used to construct a model)
	 **/
	List<ArrayList<Operation>> setOfBuildingOperators = new ArrayList<ArrayList<Operation>>();

	/**
	 * list of operations used to construct the neighbourhood of a model present on
	 * search path
	 **/
	ArrayList<Operation> buildingOperators = new ArrayList<Operation>();

	/**
	 * the number of operations closest to the best one among the operations
	 * necessary to generate the neighbourhood of a model
	 **/
	int k = -1;
	
	/**
	 **/
	BayesNet m_0;

	/**
	 * list of k operations closest to the best one used to construct the
	 * neighbourhood of a model present on search path
	 **/
	List<ArrayList<Operation>> setOfBestKBuildingOperators = new ArrayList<ArrayList<Operation>>();

	/** the number of operations on bestKBuildingOperators used **/
	int q = -1;
	
	boolean isFirstIteration = true;

	/**
	 * the Operation class contains info on operations performed on the current
	 * Bayesian network.
	 */
	class Operation implements Serializable, Comparable<Operation> {

		/** for serialization */
		static final long serialVersionUID = -4880888790432547895L;

		// constants indicating the type of an operation
		final static int OPERATION_ADD = 0;
		final static int OPERATION_DEL = 1;
		final static int OPERATION_REVERSE = 2;

		/**
		 * c'tor
		 */
		public Operation() {
		}

		/**
		 * c'tor + initializers
		 * 
		 * @param nTail
		 * @param nHead
		 * @param nOperation
		 */
		public Operation(int nTail, int nHead, int nOperation) {
			m_nHead = nHead;
			m_nTail = nTail;
			m_nOperation = nOperation;
		}

		/**
		 * compare this operation with another
		 * 
		 * @param other operation to compare with
		 * @return true if operation is the same
		 */
		public boolean equals(Operation other) {
			if (other == null) {
				return false;
			}
			return ((m_nOperation == other.m_nOperation) && (m_nHead == other.m_nHead) && (m_nTail == other.m_nTail));
		} // equals

		/** number of the tail node **/
		public int m_nTail;

		/** number of the head node **/
		public int m_nHead;

		/** type of operation (ADD, DEL, REVERSE) **/
		public int m_nOperation;

		/** change of score due to this operation **/
		public double m_fDeltaScore = -1E100;

		@Override
		public int compareTo(Operation oOperation) {
			if (this.m_fDeltaScore > oOperation.m_fDeltaScore) {
				return -1;
			}
			if (this.m_fDeltaScore < oOperation.m_fDeltaScore) {
				return 1;
			}
			return 0;
		}

	} // class Operation

	/**
	 * cache for remembering the change in score for steps in the search space
	 */
	class Cache implements RevisionHandler {

		/** change in score due to adding an arc **/
		double[][] m_fDeltaScoreAdd;
		/** change in score due to deleting an arc **/
		double[][] m_fDeltaScoreDel;

		/**
		 * c'tor
		 * 
		 * @param nNrOfNodes number of nodes in network, used to determine memory size
		 *                   to reserve
		 */
		Cache(int nNrOfNodes) {
			m_fDeltaScoreAdd = new double[nNrOfNodes][nNrOfNodes];
			m_fDeltaScoreDel = new double[nNrOfNodes][nNrOfNodes];
		}

		/**
		 * set cache entry
		 * 
		 * @param oOperation operation to perform
		 * @param fValue     value to put in cache
		 */
		public void put(Operation oOperation, double fValue) {
			if (oOperation.m_nOperation == Operation.OPERATION_ADD) {
				m_fDeltaScoreAdd[oOperation.m_nTail][oOperation.m_nHead] = fValue;
			} else {
				m_fDeltaScoreDel[oOperation.m_nTail][oOperation.m_nHead] = fValue;
			}
		} // put

		/**
		 * get cache entry
		 * 
		 * @param oOperation operation to perform
		 * @return cache value
		 */
		public double get(Operation oOperation) {
			switch (oOperation.m_nOperation) {
			case Operation.OPERATION_ADD:
				return m_fDeltaScoreAdd[oOperation.m_nTail][oOperation.m_nHead];
			case Operation.OPERATION_DEL:
				return m_fDeltaScoreDel[oOperation.m_nTail][oOperation.m_nHead];
			case Operation.OPERATION_REVERSE:
				return m_fDeltaScoreDel[oOperation.m_nTail][oOperation.m_nHead]
						+ m_fDeltaScoreAdd[oOperation.m_nHead][oOperation.m_nTail];
			}
			// should never get here
			return 0;
		} // get

		/**
		 * Returns the revision string.
		 * 
		 * @return the revision
		 */
		@Override
		public String getRevision() {
			return RevisionUtils.extract("$Revision: 10154 $");
		}
	} // class Cache
	
	boolean isTabuActivate = false;

	/**
	 * search determines the network structure/graph of the network with the Taby
	 * algorithm.
	 * 
	 * @param bayesNet  the network to use
	 * @param instances the data to use
	 * @throws Exception if something goes wrong
	 */
	@Override
	protected void search(BayesNet bayesNet, Instances instances) throws Exception {
		instances = BayesNetUtils.reorderInstanceIfNedded(bayesNet, instances);
		if (bayesNet.m_Instances.isEmpty()) {
			m_0 = new BayesNet();
			m_0.m_Instances = instances;
			m_0.initStructure();
		} else {
			m_0.m_Instances.addAll(instances);
			if (isFirstIteration) {
				isFirstIteration = false;
			}
		}
		bayesNet.m_Instances.addAll(instances);
		
		initCacheForM_0(m_0, m_0.m_Instances);
		
		int j = m_TOCOHeuristic();

		int sizeSearchPath = m_searchPath.size();
		if (j < sizeSearchPath || j == 0) {
			for (int i = sizeSearchPath - 1; i >= j; i--) { // set new M_ini
				Operation oppositeOperation = getOppositeOperation(m_searchPath.get(i));
				performOperation(bayesNet, bayesNet.m_Instances, oppositeOperation);
				m_searchPath.remove(i);
				if (j != 0) {
					if (i < sizeSearchPath - 1) {
						setOfBuildingOperators.remove(i+1);
						setOfBestKBuildingOperators.remove(i+1);
					}
				} else {
					setOfBuildingOperators.remove(i);
					setOfBestKBuildingOperators.remove(i);
				}
			}
			if (j != 0) {
				m_Cache = null;
				initCache(bayesNet, bayesNet.m_Instances);
				isTabuActivate = true;
			} else {
				isTabuActivate = false;
			}
			buildingOperators = new ArrayList<Operation>();
			Operation oOperation = getOptimalOperation(bayesNet, instances);
			while ((oOperation != null) && (oOperation.m_fDeltaScore > 0)) {
				m_searchPath.add(oOperation);
				if (isTabuActivate) {
					setOfBuildingOperators.remove(setOfBuildingOperators.size()-1);
					setOfBuildingOperators.add(buildingOperators);
					setOfBestKBuildingOperators.remove(setOfBestKBuildingOperators.size()-1);
					setOfBestKBuildingOperators.add(getBestKBuildingOperators());
					isTabuActivate = false;
				} else {
					setOfBuildingOperators.add(buildingOperators);
					setOfBestKBuildingOperators.add(getBestKBuildingOperators());
				}
				buildingOperators = new ArrayList<Operation>();
				performOperation(bayesNet, instances, oOperation);
				oOperation = getOptimalOperation(bayesNet, instances);
			}
		}

		// free up memory
		m_Cache = null;
	} // search

	private ArrayList<Operation> getBestKBuildingOperators() {
		ArrayList<Operation> bestK = new ArrayList<Operation>(buildingOperators);
		Collections.sort(bestK);
		if (k > (bestK.size() - 1)) {
			return new ArrayList<Operation>(bestK);
		}
		return new ArrayList<Operation>(bestK.subList(0, k));
	}

	int m_TOCOHeuristic() throws Exception {
		int j = 0;
		List<Operation> intersection = new ArrayList<Operation>();
		for (int i = 0; i < m_searchPath.size(); i++) {
			ArrayList<Operation> buildingOperatorsAux = new ArrayList<Operation>(setOfBuildingOperators.get(i));
			for (Operation operation : m_searchPath) { // intersection
				for (Operation operation2 : buildingOperatorsAux) {
					if (operation.equals(operation2)) {
						intersection.add(operation2);
					}
				}
			}

			Operation higherScore = null;
			for (Operation operation : intersection) {
				if (higherScore == null) {
					higherScore = operation;
				} else {
					if (m_Cache.get(operation) > m_Cache.get(higherScore)) {
						higherScore = operation;
					}
				}
			}

			if (m_searchPath.get(i).equals(higherScore)) {
				j = i + 1;
			} else {
				break;
			}

			intersection = new ArrayList<Operation>();
		}

		return j;
	}

	private Operation getOppositeOperation(Operation oOperation) {
		switch (oOperation.m_nOperation) {
		case Operation.OPERATION_ADD:
			return new Operation(oOperation.m_nTail, oOperation.m_nHead, Operation.OPERATION_DEL);
		case Operation.OPERATION_DEL:
			return new Operation(oOperation.m_nTail, oOperation.m_nHead, Operation.OPERATION_ADD);
		case Operation.OPERATION_REVERSE:
			return new Operation(oOperation.m_nHead, oOperation.m_nTail, Operation.OPERATION_REVERSE);
		}
		return null;
	}

	/**
	 * initCache initializes the cache
	 * 
	 * @param bayesNet  Bayes network to be learned
	 * @param instances data set to learn from
	 * @throws Exception if something goes wrong
	 */
	void initCache(BayesNet bayesNet, Instances instances) throws Exception {
		// determine base scores
		double[] fBaseScores = new double[instances.numAttributes()];
		int nNrOfAtts = instances.numAttributes();

		m_Cache = new Cache(nNrOfAtts);

		for (int iAttribute = 0; iAttribute < nNrOfAtts; iAttribute++) {
			updateCache(iAttribute, nNrOfAtts, bayesNet.getParentSet(iAttribute));
		}

		for (int iAttribute = 0; iAttribute < nNrOfAtts; iAttribute++) {
			fBaseScores[iAttribute] = calcNodeScore(iAttribute);
		}

		for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
			for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
				if (iAttributeHead != iAttributeTail) {
					Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_ADD);
					m_Cache.put(oOperation,
							calcScoreWithExtraParent(iAttributeHead, iAttributeTail) - fBaseScores[iAttributeHead]);
				}
			}
		}

	} // initCache

	/**
	 * check whether the operation is not in the forbidden. For base hill climber,
	 * there are no restrictions on operations, so we always return true.
	 * 
	 * @param oOperation operation to be checked
	 * @return true if operation is not in the tabu list
	 */
	boolean isNotTabu(Operation oOperation) {
		if (!isFirstIteration && m_searchPath.size() > 0 && isTabuActivate) {
			int i = m_searchPath.size(); // get step of learning
			ArrayList<Operation> bestKOperations = setOfBestKBuildingOperators.get(i);
			if (q < k) {
				if (q < (bestKOperations.size() - 1)) {
					bestKOperations = new ArrayList<Operation>(bestKOperations.subList(0, q));
				}
			}
			for (Operation operation : bestKOperations) {
				if (operation.equals(oOperation)) {
					return true;
				}
			}
			return false;
		}
		return true;
	} // isNotTabu

	/**
	 * getOptimalOperation finds the optimal operation that can be performed on the
	 * Bayes network that is not in the tabu list.
	 * 
	 * @param bayesNet  Bayes network to apply operation on
	 * @param instances data set to learn from
	 * @return optimal operation found
	 * @throws Exception if something goes wrong
	 */
	Operation getOptimalOperation(BayesNet bayesNet, Instances instances) throws Exception {
		Operation oBestOperation = new Operation();

		// Add???
		oBestOperation = findBestArcToAdd(bayesNet, instances, oBestOperation);
		// Delete???
		oBestOperation = findBestArcToDelete(bayesNet, instances, oBestOperation);
		// Reverse???
		if (getUseArcReversal()) {
			oBestOperation = findBestArcToReverse(bayesNet, instances, oBestOperation);
		}

		// did we find something?
		if (oBestOperation.m_fDeltaScore == -1E100) {
			return null;
		}

		return oBestOperation;
	} // getOptimalOperation

	/**
	 * performOperation applies an operation on the Bayes network and update the
	 * cache.
	 * 
	 * @param bayesNet   Bayes network to apply operation on
	 * @param instances  data set to learn from
	 * @param oOperation operation to perform
	 * @throws Exception if something goes wrong
	 */
	void performOperation(BayesNet bayesNet, Instances instances, Operation oOperation) throws Exception {
		// perform operation
		switch (oOperation.m_nOperation) {
		case Operation.OPERATION_ADD:
			applyArcAddition(bayesNet, oOperation.m_nHead, oOperation.m_nTail, instances);
//			System.out.print("Add " + bayesNet.getNodeName(oOperation.m_nTail) + " -> "
//					+ bayesNet.getNodeName(oOperation.m_nHead) + "\n");
			break;
		case Operation.OPERATION_DEL:
			applyArcDeletion(bayesNet, oOperation.m_nHead, oOperation.m_nTail, instances);
//			System.out.print("Del " + bayesNet.getNodeName(oOperation.m_nTail) + " -> "
//					+ bayesNet.getNodeName(oOperation.m_nHead) + "\n");
			break;
		case Operation.OPERATION_REVERSE:
			applyArcDeletion(bayesNet, oOperation.m_nHead, oOperation.m_nTail, instances);
			applyArcAddition(bayesNet, oOperation.m_nTail, oOperation.m_nHead, instances);
//			System.out.print("Rev " + bayesNet.getNodeName(oOperation.m_nTail) + " -> "
//					+ bayesNet.getNodeName(oOperation.m_nHead) + "\n");
			break;
		}
	} // performOperation

	/**
	 * 
	 * @param bayesNet
	 * @param iHead
	 * @param iTail
	 * @param instances
	 */
	void applyArcAddition(BayesNet bayesNet, int iHead, int iTail, Instances instances) {
		ParentSet bestParentSet = bayesNet.getParentSet(iHead);
		bestParentSet.addParent(iTail, instances);
		updateCache(iHead, instances.numAttributes(), bestParentSet);
	} // applyArcAddition

	/**
	 * 
	 * @param bayesNet
	 * @param iHead
	 * @param iTail
	 * @param instances
	 */
	void applyArcDeletion(BayesNet bayesNet, int iHead, int iTail, Instances instances) {
		ParentSet bestParentSet = bayesNet.getParentSet(iHead);
		bestParentSet.deleteParent(iTail, instances);
		updateCache(iHead, instances.numAttributes(), bestParentSet);
	} // applyArcAddition

	/**
	 * find best (or least bad) arc addition operation
	 * 
	 * @param bayesNet       Bayes network to add arc to
	 * @param instances      data set
	 * @param oBestOperation
	 * @return Operation containing best arc to add, or null if no arc addition is
	 *         allowed (this can happen if any arc addition introduces a cycle, or
	 *         all parent sets are filled up to the maximum nr of parents).
	 */
	Operation findBestArcToAdd(BayesNet bayesNet, Instances instances, Operation oBestOperation) {
		int nNrOfAtts = instances.numAttributes();
		// find best arc to add
		for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
			if (bayesNet.getParentSet(iAttributeHead).getNrOfParents() < m_nMaxNrOfParents) {
				for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
					if (addArcMakesSense(bayesNet, instances, iAttributeHead, iAttributeTail)) {
						Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_ADD);
						if (isNotTabu(oOperation)) {
							buildingOperators.add(oOperation); // storing the operations to build the neighbourhood
							if (m_Cache.get(oOperation) > oBestOperation.m_fDeltaScore) {
								oBestOperation = oOperation;
								oBestOperation.m_fDeltaScore = m_Cache.get(oOperation);
							}
						}
					}
				}
			}
		}
		return oBestOperation;
	} // findBestArcToAdd

	/**
	 * find best (or least bad) arc deletion operation
	 * 
	 * @param bayesNet       Bayes network to delete arc from
	 * @param instances      data set
	 * @param oBestOperation
	 * @return Operation containing best arc to delete, or null if no deletion can
	 *         be made (happens when there is no arc in the network yet).
	 */
	Operation findBestArcToDelete(BayesNet bayesNet, Instances instances, Operation oBestOperation) {
		int nNrOfAtts = instances.numAttributes();
		// find best arc to delete
		for (int iNode = 0; iNode < nNrOfAtts; iNode++) {
			ParentSet parentSet = bayesNet.getParentSet(iNode);
			for (int iParent = 0; iParent < parentSet.getNrOfParents(); iParent++) {
				Operation oOperation = new Operation(parentSet.getParent(iParent), iNode, Operation.OPERATION_DEL);
				if (isNotTabu(oOperation)) {
					buildingOperators.add(oOperation); // storing the operations to build the neighbourhood
					if (m_Cache.get(oOperation) > oBestOperation.m_fDeltaScore) {
						oBestOperation = oOperation;
						oBestOperation.m_fDeltaScore = m_Cache.get(oOperation);
					}
				}
			}
		}
		return oBestOperation;
	} // findBestArcToDelete

	/**
	 * find best (or least bad) arc reversal operation
	 * 
	 * @param bayesNet       Bayes network to reverse arc in
	 * @param instances      data set
	 * @param oBestOperation
	 * @return Operation containing best arc to reverse, or null if no reversal is
	 *         allowed (happens if there is no arc in the network yet, or when any
	 *         such reversal introduces a cycle).
	 */
	Operation findBestArcToReverse(BayesNet bayesNet, Instances instances, Operation oBestOperation) {
		int nNrOfAtts = instances.numAttributes();
		// find best arc to reverse
		for (int iNode = 0; iNode < nNrOfAtts; iNode++) {
			ParentSet parentSet = bayesNet.getParentSet(iNode);
			for (int iParent = 0; iParent < parentSet.getNrOfParents(); iParent++) {
				int iTail = parentSet.getParent(iParent);
				// is reversal allowed?
				if (reverseArcMakesSense(bayesNet, instances, iNode, iTail)
						&& bayesNet.getParentSet(iTail).getNrOfParents() < m_nMaxNrOfParents) {
					// go check if reversal results in the best step forward
					Operation oOperation = new Operation(parentSet.getParent(iParent), iNode,
							Operation.OPERATION_REVERSE);
					if (isNotTabu(oOperation)) {
						buildingOperators.add(oOperation); // storing the operations to build the neighbourhood
						if (m_Cache.get(oOperation) > oBestOperation.m_fDeltaScore) {
							oBestOperation = oOperation;
							oBestOperation.m_fDeltaScore = m_Cache.get(oOperation);
						}
					}
				}
			}
		}
		return oBestOperation;
	} // findBestArcToReverse

	/**
	 * update the cache due to change of parent set of a node
	 * 
	 * @param iAttributeHead node that has its parent set changed
	 * @param nNrOfAtts      number of nodes/attributes in data set
	 * @param parentSet      new parents set of node iAttributeHead
	 */
	void updateCache(int iAttributeHead, int nNrOfAtts, ParentSet parentSet) {
		// update cache entries for arrows heading towards iAttributeHead
		double fBaseScore = calcNodeScore(iAttributeHead);
		int nNrOfParents = parentSet.getNrOfParents();
		for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
			if (iAttributeTail != iAttributeHead) {
				if (!parentSet.contains(iAttributeTail)) {
					// add entries to cache for adding arcs
					if (nNrOfParents < m_nMaxNrOfParents) {
						Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_ADD);
						m_Cache.put(oOperation, calcScoreWithExtraParent(iAttributeHead, iAttributeTail) - fBaseScore);
					}
				} else {
					// add entries to cache for deleting arcs
					Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_DEL);
					m_Cache.put(oOperation, calcScoreWithMissingParent(iAttributeHead, iAttributeTail) - fBaseScore);
				}
			}
		}
	} // updateCache

	/**
	 * Sets the max number of parents
	 * 
	 * @param nMaxNrOfParents the max number of parents
	 */
	public void setMaxNrOfParents(int nMaxNrOfParents) {
		m_nMaxNrOfParents = nMaxNrOfParents;
	}

	/**
	 * Gets the max number of parents.
	 * 
	 * @return the max number of parents
	 */
	public int getMaxNrOfParents() {
		return m_nMaxNrOfParents;
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
	 * Gets whether to init as naive bayes
	 * 
	 * @return whether to init as naive bayes
	 */
	public boolean getInitAsNaiveBayes() {
		return m_bInitAsNaiveBayes;
	}

	/**
	 * get use the arc reversal operation
	 * 
	 * @return whether the arc reversal operation should be used
	 */
	public boolean getUseArcReversal() {
		return m_bUseArcReversal;
	} // getUseArcReversal

	/**
	 * set use the arc reversal operation
	 * 
	 * @param bUseArcReversal whether the arc reversal operation should be used
	 */
	public void setUseArcReversal(boolean bUseArcReversal) {
		m_bUseArcReversal = bUseArcReversal;
	} // setUseArcReversal

	public int getK() {
		return k;
	}

	public void setK(int k) {
		this.k = k+1;
	}

	public int getQ() {
		return q;
	}

	public void setQ(int q) {
		this.q = q;
	}

	/**
	 * @return a string to describe the Use Arc Reversal option.
	 */
	public String useArcReversalTipText() {
		return "When set to true, the arc reversal operation is used in the search.";
	} // useArcReversalTipText

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 10154 $");
	}

	private void initCacheForM_0(BayesNet bayesNet, Instances instances) {
		double[] fBaseScores = new double[instances.numAttributes()];
		int nNrOfAtts = instances.numAttributes();

		m_Cache = new Cache(nNrOfAtts);

		for (int iAttribute = 0; iAttribute < nNrOfAtts; iAttribute++) {
			updateCacheForM_0(iAttribute, nNrOfAtts, bayesNet.getParentSet(iAttribute));
		}

		for (int iAttribute = 0; iAttribute < nNrOfAtts; iAttribute++) {
			fBaseScores[iAttribute] = calcNodeScoreForM_0(iAttribute);
		}

		for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
			for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
				if (iAttributeHead != iAttributeTail) {
					Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_ADD);
					m_Cache.put(oOperation,
							calcScoreWithExtraParentForM_0(iAttributeHead, iAttributeTail) - fBaseScores[iAttributeHead]);
				}
			}
		}
	}

	private void updateCacheForM_0(int iAttributeHead, int nNrOfAtts, ParentSet parentSet) {
		// TODO Auto-generated method stub
		double fBaseScore = calcNodeScoreForM_0(iAttributeHead); //FALTA
		int nNrOfParents = parentSet.getNrOfParents();
		for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
			if (iAttributeTail != iAttributeHead) {
				if (!parentSet.contains(iAttributeTail)) {
					// add entries to cache for adding arcs
					if (nNrOfParents < m_nMaxNrOfParents) {
						Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_ADD);
						m_Cache.put(oOperation, calcScoreWithExtraParentForM_0(iAttributeHead, iAttributeTail) - fBaseScore); //FALTA
					}
				} else {
					// add entries to cache for deleting arcs
					Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_DEL);
					m_Cache.put(oOperation, calcScoreWithMissingParentForM_0(iAttributeHead, iAttributeTail) - fBaseScore); //FALTA
				}
			}
		}
	}

	private double calcNodeScoreForM_0(int nNode) {
		Instances instances = m_0.m_Instances;
	    ParentSet oParentSet = m_0.getParentSet(nNode);

	    // determine cardinality of parent set & reserve space for frequency counts
	    int nCardinality = oParentSet.getCardinalityOfParents();
	    int numValues = instances.attribute(nNode).numValues();
	    int[] nCounts = new int[nCardinality * numValues];

	    // initialize (don't need this?)
	    for (int iParent = 0; iParent < nCardinality * numValues; iParent++) {
	      nCounts[iParent] = 0;
	    }

	    // estimate distributions
	    Enumeration<Instance> enumInsts = instances.enumerateInstances();

	    while (enumInsts.hasMoreElements()) {
	      Instance instance = enumInsts.nextElement();

	      // updateClassifier;
	      double iCPT = 0;

	      for (int iParent = 0; iParent < oParentSet.getNrOfParents(); iParent++) {
	        int nParent = oParentSet.getParent(iParent);

	        iCPT = iCPT * instances.attribute(nParent).numValues()
	          + instance.value(nParent);
	      }

	      nCounts[numValues * ((int) iCPT) + (int) instance.value(nNode)]++;
	    }

	    return calcScoreOfCounts(nCounts, nCardinality, numValues, instances);
	}
	

	private double calcScoreWithExtraParentForM_0(int nNode, int nCandidateParent) {
		ParentSet oParentSet = m_0.getParentSet(nNode);

	    // sanity check: nCandidateParent should not be in parent set already
	    if (oParentSet.contains(nCandidateParent)) {
	      return -1e100;
	    }

	    // set up candidate parent
	    oParentSet.addParent(nCandidateParent, m_0.m_Instances);

	    // calculate the score
	    double logScore = calcNodeScoreForM_0(nNode);

	    // delete temporarily added parent
	    oParentSet.deleteLastParent(m_0.m_Instances);

	    return logScore;
	}
	

	private double calcScoreWithMissingParentForM_0(int nNode, int nCandidateParent) {
		ParentSet oParentSet = m_0.getParentSet(nNode);

	    // sanity check: nCandidateParent should be in parent set already
	    if (!oParentSet.contains(nCandidateParent)) {
	      return -1e100;
	    }

	    // set up candidate parent
	    int iParent = oParentSet.deleteParent(nCandidateParent,
	      m_0.m_Instances);

	    // calculate the score
	    double logScore = calcNodeScoreForM_0(nNode);

	    // restore temporarily deleted parent
	    oParentSet.addParent(nCandidateParent, iParent, m_0.m_Instances);

	    return logScore;
	}
	
} // HillClimber
