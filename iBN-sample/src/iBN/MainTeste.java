package iBN;

import iBN.classifiers.bayes.IncrementalBayesNet;
import iBN.core.BayesNetUtils;
import iBN.core.IncStatistics;
import iBN.core.InstancesHelper;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.BIFReader;
import weka.classifiers.bayes.net.search.local.HillClimber;
import weka.core.converters.ConverterUtils.DataSource;

public class MainTeste {

	public static void main(String[] args) throws Exception {
		
		System.out.println(1.0+2.0+5.0/3.0);

		// <-- SHI TAN ALGORITHM -->
//		iBN.classifiers.bayes.net.structure.search.ci.ShiTanAlgorithm algorithm = new iBN.classifiers.bayes.net.structure.search.ci.ShiTanAlgorithm();
//		algorithm.setInitAsNaiveBayes(false);
//		// <-- END OF SHI TAN ALGORITHM -->
//
//		// <-- IHCS ALGORITHM -->
////		iBN.classifiers.bayes.net.structure.search.local.IncrementalHillClimber algorithm = new iBN.classifiers.bayes.net.structure.search.local.IncrementalHillClimber();
////		algorithm.setUseArcReversal(true);
////		algorithm.setInitAsNaiveBayes(false);
////		algorithm.setMaxNrOfParents(100000);
//////		algorithm.setMaxNrOfParents(1);
////		algorithm.setK(2); // Storage impact
//////		algorithm.setK(Integer.MAX_VALUE-1); // Storage impact
////		algorithm.setQ(2); // Storage impact (q has to be smaller than k)
//////		algorithm.setQ(Integer.MAX_VALUE-1); // Storage impact (q has to be smaller than k)
//		// <-- END IHCS ALGORITHM -->
//
//		BayesNet bayesNet = (BayesNet) new BIFReader().processFile("bayesnets/car_only_nodes.xml");
////		BayesNet bayesNet = (BayesNet) new BIFReader().processFile("bayesnets/car_partial.xml");
//		bayesNet.initCPTs();
////		BayesNetUtils.visualizeBayesNet(bayesNet, "Inicial");
//		DataSource dataSource = new DataSource("datasets/car_all.arff");
////		weka.core.Instances m_Instances = dataSource.getDataSet();
////		weka.core.Instances m_Instances = InstancesHelper.sortInstancesByInstance(dataSource.getDataSet());
//		weka.core.Instances m_Instances = InstancesHelper.reverseSortInstancesByInstance(dataSource.getDataSet());
//		IncrementalBayesNet iBayesNet = new IncrementalBayesNet();
//		iBayesNet.setBayesNet(bayesNet);
//		iBayesNet.setInstances(m_Instances);
//		iBayesNet.setInit_Index(0);
//		iBayesNet.setK(100); // remember of validate this k and max k values
//		iBayesNet.setIncrementalIndex(1); // set to 1.5 to up the step exponentially. 1 to increment only k value
//		iBayesNet.setMax_k(1296);
//		iBayesNet.setSearchAlgorithm(algorithm);
//		iBayesNet.buildStructure();
//		BayesNetUtils.visualizeBayesNet(iBayesNet.getBayesNet(), "Versão Final (Incremental)");
//		System.out.println("<== END OF INCREMENTAL ==>");

		// <-- HCS ALGORITHM -->
//		weka.classifiers.bayes.BayesNet bayesNetToBatch = (weka.classifiers.bayes.BayesNet) new BIFReader()
//				.processFile("bayesnets/nursery_only_nodes.xml");
//		DataSource dataSourceToBatch = new DataSource("datasets/nursery_all.arff");
////		weka.core.Instances m_Instances = dataSourceToBatch.getDataSet();
////		InstancesHelper.printScores(m_Instances);
//		weka.core.Instances m_Instances = InstancesHelper.sortInstancesByInstance(dataSourceToBatch.getDataSet());
////		weka.core.Instances m_Instances = InstancesHelper.reverseSortInstancesByInstance(dataSourceToBatch.getDataSet());
//
//		int init_Index = 0;
//		int max_k = 9720;
//		int k = 100;
//		int aux = k;
////		System.out.println("MDL MIT LL A");
////		System.out.println("A LL MDL Precisao Cobertura F Extra Perdido Invertido");
//		System.out.println("A LL MDL");
//		while (k <= max_k) {
//			HillClimber batchHCS = new HillClimber();
//			batchHCS.setUseArcReversal(true);
//			batchHCS.setInitAsNaiveBayes(false);
//			batchHCS.setMaxNrOfParents(100000);
//			bayesNetToBatch = (weka.classifiers.bayes.BayesNet) new BIFReader()
//					.processFile("bayesnets/nursery_only_nodes.xml");
//			// Biff file to compare with original version
////			bayesNetToBatch.setBIFFile(
////					BayesNetUtils.writeBIFFFile(BayesNetUtils.getBIFFVersion("bayesnets/original/alarm.xml")));
////			BayesNetUtils.deleteBIFFFile();
//			// end set biff file
//			bayesNetToBatch.m_Instances.addAll(m_Instances.subList(init_Index, k));
//			batchHCS.buildStructure(bayesNetToBatch, bayesNetToBatch.m_Instances);
//			bayesNetToBatch.initCPTs();
//			bayesNetToBatch.estimateCPTs();
////			double[] precisionRecallFMeasure = BayesNetUtils.precisionRecallFMeasure(
////					(weka.classifiers.bayes.BayesNet) new BIFReader().processFile("bayesnets/original/alarm.xml"),
////					bayesNetToBatch);
////			System.out.println(IncStatistics.measureMDLScore(bayesNetToBatch) + " " + IncStatistics.measureMITScore(bayesNetToBatch) + " " + IncStatistics.calculateLogLoss(bayesNetToBatch, m_Instances, max_k) + " " + IncStatistics.calculateAccurary(bayesNetToBatch, m_Instances, max_k));
////			System.out.println(IncStatistics.measureMDLScore(bayesNetToBatch) + " "
////					+ IncStatistics.measureMITScore(bayesNetToBatch) + " " + bayesNetToBatch.measureExtraArcs() + " "
////					+ bayesNetToBatch.measureMissingArcs() + " " + bayesNetToBatch.measureReversedArcs() + " "
////					+ precisionRecallFMeasure[0] + " " + precisionRecallFMeasure[1] + " " + precisionRecallFMeasure[2]
////					+ " " + IncStatistics.calculateLogLossWithMultiClass(bayesNetToBatch, m_Instances, max_k) + " "
////					+ IncStatistics.calculateAccuraryWithMultiClass(bayesNetToBatch, m_Instances, max_k, 5));
//
////			System.out.println(IncStatistics.calculateAccuraryWithMultiClass(bayesNetToBatch, m_Instances, max_k, 5)
////					+ " " + IncStatistics.calculateLogLossWithMultiClass(bayesNetToBatch, m_Instances, max_k) + " "
////					+ IncStatistics.measureMDLScore(bayesNetToBatch) + " " + precisionRecallFMeasure[0] + " "
////					+ precisionRecallFMeasure[1] + " " + precisionRecallFMeasure[2] + " "
////					+ bayesNetToBatch.measureExtraArcs() + " " + bayesNetToBatch.measureMissingArcs() + " "
////					+ bayesNetToBatch.measureReversedArcs());
//			
//			System.out.println(IncStatistics.calculateAccurary(bayesNetToBatch, m_Instances, max_k)
//					+ " " + IncStatistics.calculateLogLoss(bayesNetToBatch, m_Instances, max_k) + " "
//					+ IncStatistics.measureMDLScore(bayesNetToBatch));
////			System.out.println(IncStatistics.calculateAccuraryWithMultiClass(bayesNetToBatch, m_Instances, max_k, 5));
//			k = k + aux;
//		}
//		BayesNetUtils.saveDataToConfusionMatrix(bayesNetToBatch, m_Instances, max_k, "HCS_100_nursery_sim", Integer.MIN_VALUE);
////		BayesNetUtils.saveDataToConfusionMatrix(bayesNetToBatch, m_Instances, max_k, "BHCS_100_nursery",
////				Integer.MIN_VALUE);
//		BayesNetUtils.visualizeBayesNet(bayesNetToBatch, "Final (Batch)");
//		System.out.println("<== END OF BATCH HCS ==>");

		// Dar uma olhada nisso. divergence(BayesNet other) in BIFReader, called in
		// measureDivergence() in BayesNet
		/**
		 * calculates the divergence between the probability distribution represented by
		 * this network and that of another, that is, \sum_{x\in X} P(x)log P(x)/Q(x)
		 * where X is the set of values the nodes in the network can take, P(x) the
		 * probability of this network for configuration x Q(x) the probability of the
		 * other network for configuration x
		 * 
		 * @param other network to compare with
		 * @return divergence between this and other Bayes Network
		 */

//		for (int i = 100; i < 1101; i++) {
//			iBN.classifiers.bayes.net.structure.search.local.IncrementalHillClimber algorithm = new iBN.classifiers.bayes.net.structure.search.local.IncrementalHillClimber();
//			algorithm.setUseArcReversal(true);
//			algorithm.setInitAsNaiveBayes(false);
//			algorithm.setMaxNrOfParents(100000);
//			algorithm.setK(i); // Storage impact
//			algorithm.setQ(i); // Storage impact (q has to be smaller than k)
//			// <-- END IHCS ALGORITHM -->
//			
//			BayesNet bayesNet = (BayesNet) new BIFReader().processFile("bayesnets/alarm_only_nodes.xml");
////			bayesNet.initCPTs();
////			BayesNetUtils.visualizeBayesNet(bayesNet, "Inicial");
//			DataSource dataSource = new DataSource("datasets/alarm_all.arff");
//			IncrementalBayesNet iBayesNet = new IncrementalBayesNet();
//			iBayesNet.setBayesNet(bayesNet);
//			iBayesNet.setInstances(dataSource.getDataSet());
//			iBayesNet.setInit_Index(0);
//			iBayesNet.setK(1000); // remember of validate this k and max k values
//			iBayesNet.setIncrementalIndex(1); // set to 1.5 to up the step exponentially. 1 to increment only k value
//			iBayesNet.setMax_k(20000);
//			iBayesNet.setSearchAlgorithm(algorithm);
//			iBayesNet.buildStructure();
////			BayesNetUtils.visualizeBayesNet(iBayesNet.getBayesNet(), "Final (Incremental)");
//		}
	}
}
