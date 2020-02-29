package iBN.classifiers.bayes;

import java.util.ArrayList;
import java.util.List;

import iBN.core.BayesNetUtils;
import iBN.core.IncStatistics;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.BIFReader;
import weka.core.Instances;

public class IncrementalBayesNet extends BayesNet {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Initial Bayesian Network.
	 */
	BayesNet bayesNet;

	/**
	 * The datasets headers for the purposes of learning Bayesian Network structures
	 * 
	 */
	Instances m_Instances;

	/**
	 * The size of subsets
	 */
	int k;

	/**
	 * The max size of instances
	 */
	int max_k;

	double incrementalIndex;

	/**
	 * The init index of instances
	 */
	int init_Index;

	/**
	 * buildStructure determines the network structure/graph of the network. The
	 * default behavior is creating a network where all nodes have the first node as
	 * its parent (i.e., a BayesNet that behaves like a naive Bayes classifier).
	 * This method can be overridden by derived classes to restrict the class of
	 * network structures that are acceptable.
	 * 
	 * @throws Exception in case of an error
	 */
	@Override
	public void buildStructure() throws Exception {
		if (max_k == 0) {
			max_k = m_Instances.size();
		}

		// Biff file to compare with original version
//		bayesNet.setBIFFile(BayesNetUtils.writeBIFFFile(BayesNetUtils.getBIFFVersion("bayesnets/original/alarm.xml")));
//		BayesNetUtils.deleteBIFFFile();
		// end set biff file

		// Biff file to compare with batch version
//		bayesNet.setBIFFile(BayesNetUtils
//				.writeBIFFFile(BayesNetUtils.getBatchBIFFVersion("bayesnets/alarm_only_nodes.xml", m_Instances)));
		// end set biff file

		int aux = k;
		k = init_Index;
//		System.out.println("MDL MIT LL A");
//		System.out.println("MDL MIT Extra Perdido Invertido Precisao Cobertura F LL A");
//		System.out.println("A LL MDL Precisao Cobertura F Extra Perdido Invertido");
		System.out.println("A LL MDL");
		while ((k + aux) <= max_k) {
			Instances instances = new Instances(m_Instances);
			instances.delete();
			instances.addAll(m_Instances.subList(k, (int) k + aux));
			getSearchAlgorithm().buildStructure(bayesNet, instances);
			bayesNet.initCPTs();
			bayesNet.estimateCPTs();
//			double[] precisionRecallFMeasure = BayesNetUtils.precisionRecallFMeasure(
//					(weka.classifiers.bayes.BayesNet) new BIFReader().processFile("bayesnets/to_compare.xml"),
//					bayesNet);
//			System.out.println(IncStatistics.measureMDLScore(bayesNet) + " " + IncStatistics.measureMITScore(bayesNet) + " " + IncStatistics.calculateLogLoss(bayesNet, m_Instances, max_k) + " " + IncStatistics.calculateAccurary(bayesNet, m_Instances, max_k));
//			System.out.println(IncStatistics.measureMDLScore(bayesNet) + " " + IncStatistics.measureMITScore(bayesNet)
//					+ " " + bayesNet.measureExtraArcs() + " " + bayesNet.measureMissingArcs() + " "
//					+ bayesNet.measureReversedArcs() + " " + precisionRecallFMeasure[0] + " "
//					+ precisionRecallFMeasure[1] + " " + precisionRecallFMeasure[2] + " "
//					+ IncStatistics.calculateLogLossWithMultiClass(bayesNet, m_Instances, max_k) + " "
//					+ IncStatistics.calculateAccuraryWithMultiClass(bayesNet, m_Instances, max_k, 5));
//			System.out.println(IncStatistics.calculateAccuraryWithMultiClass(bayesNet, m_Instances, max_k, 5)
//					+ " " + IncStatistics.calculateLogLossWithMultiClass(bayesNet, m_Instances, max_k) + " "
//					+ IncStatistics.measureMDLScore(bayesNet) + " " + precisionRecallFMeasure[0] + " "
//					+ precisionRecallFMeasure[1] + " " + precisionRecallFMeasure[2] + " "
//					+ bayesNet.measureExtraArcs() + " " + bayesNet.measureMissingArcs() + " "
//					+ bayesNet.measureReversedArcs());
			System.out.println(IncStatistics.calculateAccurary(bayesNet, m_Instances, max_k)
					+ " " + IncStatistics.calculateLogLoss(bayesNet, m_Instances, max_k) + " "
					+ IncStatistics.measureMDLScore(bayesNet));
			k = k + aux;
//			BayesNetUtils.visualizeBayesNet(bayesNet, "Aprendizagem com " + k + " instâncias");
		}
		BayesNetUtils.deleteBIFFFile();
		BayesNetUtils.saveDataToConfusionMatrix(bayesNet, m_Instances, max_k, "ST_100_car_dissim", Integer.MIN_VALUE);
//		BayesNetUtils.saveDataToConfusionMatrix(bayesNet, m_Instances, max_k, "ST_100_dissim_partial_nursery_1_9_rq6", Integer.MIN_VALUE);
	}

	public BayesNet getBayesNet() {
		return bayesNet;
	}

	public void setBayesNet(BayesNet bayesNet) {
		this.bayesNet = bayesNet;
	}

	public Instances getInstances() {
		return m_Instances;
	}

	public void setInstances(Instances instances) {
		this.m_Instances = instances;
	}

	public int getK() {
		return k;
	}

	public void setK(int k) {
		this.k = k;
	}

	public int getMax_k() {
		return max_k;
	}

	public void setMax_k(int max_k) {
		this.max_k = max_k;
	}

	public int getInit_Index() {
		return init_Index;
	}

	public void setInit_Index(int init_Index) {
		this.init_Index = init_Index;
	}

	public double getIncrementalIndex() {
		return incrementalIndex;
	}

	public void setIncrementalIndex(double incrementalIndex) {
		this.incrementalIndex = incrementalIndex;
	}
}
