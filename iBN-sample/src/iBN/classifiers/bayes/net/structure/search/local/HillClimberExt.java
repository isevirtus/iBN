package iBN.classifiers.bayes.net.structure.search.local;

import java.io.Serializable;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.ParentSet;
import weka.classifiers.bayes.net.search.local.LocalScoreSearchAlgorithm;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

public class HillClimberExt extends LocalScoreSearchAlgorithm {

	/** For serialization */
	private static final long serialVersionUID = -6309548242850049997L;
	
	/**
	 * The Operation class contains info on operations performed on the current Bayesian network.
	 */
	class Operation implements Serializable, RevisionHandler {

		public int m_nTail;

		public int m_nHead;

		public int m_nOperation;

		/** change of score due to this operation **/
		public double m_fDeltaScore = -1E100;

		static final long serialVersionUID = -4880888790432547895L;

		final static int OPERATION_ADD = 0;
		final static int OPERATION_DEL = 1;
		final static int OPERATION_REVERSE = 2;

		public Operation() {
		}

		public Operation(int nTail, int nHead, int nOperation) {
			m_nHead = nHead;
			m_nTail = nTail;
			m_nOperation = nOperation;
		}

		public boolean equals(Operation other) {
			if (other == null) {
				return false;
			}
			return ((m_nOperation == other.m_nOperation) && (m_nHead == other.m_nHead) && (m_nTail == other.m_nTail));
		}

		public String getRevision() {
			return RevisionUtils.extract("$Revision: 8034 $");
		}
	}
	
	/** cache for remembering the change in score for steps in the search space */
	class Cache implements RevisionHandler {

		/** change in score due to adding an arc **/
		double[][] m_fDeltaScoreAdd;
		/** change in score due to deleting an arc **/
		double[][] m_fDeltaScoreDel;

		Cache(int nNrOfNodes) {
			m_fDeltaScoreAdd = new double[nNrOfNodes][nNrOfNodes];
			m_fDeltaScoreDel = new double[nNrOfNodes][nNrOfNodes];
		}

		public void put(Operation oOperation, double fValue) {
			if (oOperation.m_nOperation == Operation.OPERATION_ADD) {
				m_fDeltaScoreAdd[oOperation.m_nTail][oOperation.m_nHead] = fValue;
			} else {
				m_fDeltaScoreDel[oOperation.m_nTail][oOperation.m_nHead] = fValue;
			}
		}

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
		}

		public String getRevision() {
			return RevisionUtils.extract("$Revision: 8034 $");
		}
	}
	
	/** cache for storing score differences **/
	Cache m_Cache = null;

	/** use the arc reversal operator **/
	boolean m_bUseArcReversal = false;
	
	/**
	 * search determines the network structure/graph of the network with the Taby
	 * algorithm.
	 * 
	 * @param bayesNet  the network to use
	 * @param instances the data to use
	 * @throws Exception if something goes wrong
	 */
	public void search(BayesNet bayesNet, Instances instances) throws Exception {
		initCache(bayesNet, instances);

		// go do the search
		Operation oOperation = getOptimalOperation(bayesNet, instances);
		while ((oOperation != null) && (oOperation.m_fDeltaScore > 0)) {
			performOperation(bayesNet, instances, oOperation);
			oOperation = getOptimalOperation(bayesNet, instances);
		}
		
		// free up memory
		m_Cache = null;
	} // search
	
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
			if (bayesNet.getDebug()) {
				System.out.print("Add " + oOperation.m_nHead + " -> " + oOperation.m_nTail);
			}
			break;
		case Operation.OPERATION_DEL:
			applyArcDeletion(bayesNet, oOperation.m_nHead, oOperation.m_nTail, instances);
			if (bayesNet.getDebug()) {
				System.out.print("Del " + oOperation.m_nHead + " -> " + oOperation.m_nTail);
			}
			break;
		case Operation.OPERATION_REVERSE:
			applyArcDeletion(bayesNet, oOperation.m_nHead, oOperation.m_nTail, instances);
			applyArcAddition(bayesNet, oOperation.m_nTail, oOperation.m_nHead, instances);
			if (bayesNet.getDebug()) {
				System.out.print("Rev " + oOperation.m_nHead + " -> " + oOperation.m_nTail);
			}
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
						if (m_Cache.get(oOperation) > oBestOperation.m_fDeltaScore) {
							if (isNotTabu(oOperation)) {
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
				if (m_Cache.get(oOperation) > oBestOperation.m_fDeltaScore) {
					if (isNotTabu(oOperation)) {
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
					if (m_Cache.get(oOperation) > oBestOperation.m_fDeltaScore) {
						if (isNotTabu(oOperation)) {
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
					// AQUI ATUALIZA OU ADD NO CACHE O SCORE DA OPERAÇĂO DE ADIÇĂO PARA TODOS OS
					// OUTROS NÓS QUE NĂO SĂO PAIS DA VARIÁVEL NA REDE ATUAL
					// add entries to cache for adding arcs
					if (nNrOfParents < m_nMaxNrOfParents) {
						Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_ADD);
						m_Cache.put(oOperation, calcScoreWithExtraParent(iAttributeHead, iAttributeTail) - fBaseScore);
					}
				} else {
					// SE A VARIAVÉL FO FOR FOR PAI DE VARIÁVEL JÁ NA REDE BASE, ATUALIZA OU ADD NO
					// CACHE O SCORE DA OPERAÇĂO DE REMOÇĂO
					// add entries to cache for deleting arcs
					Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_DEL);
					m_Cache.put(oOperation, calcScoreWithMissingParent(iAttributeHead, iAttributeTail) - fBaseScore);
				}
			}
		}
	}

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
	

	/** Bayes net contendo todos os pais possíveis **/
	BayesNet m_bayesNet = null;

	public BayesNet getM_bayesNet() {
		return m_bayesNet;
	}

	public void setM_bayesNet(BayesNet m_bayesNet) {
		this.m_bayesNet = m_bayesNet;
	}
	
	/**
	 * check whether the operation is not in the forbidden.
	 * 
	 * @param oOperation operation to be checked
	 * @return true if operation is not in the tabu list
	 */
	boolean isNotTabu(Operation oOperation) {
		//if reverse
		if (oOperation.m_nOperation == 2) {
			if (!m_bayesNet.getParentSet(oOperation.m_nTail).contains(oOperation.m_nHead)) {
				return false;
			}
		//if add
		} else if (oOperation.m_nOperation == 0) {
			if (!m_bayesNet.getParentSet(oOperation.m_nHead).contains(oOperation.m_nTail)) {
				return false;
			}
		}
		return true;
	} // isNotTabu
}
