package iBN.core;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import JavaMI.MutualInformation;
import JavaMI.ProbabilityState;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.search.local.Scoreable;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Statistics;
import weka.core.Utils;

public class IncStatistics extends Statistics {

	protected static final double V99[] = { 6.635, 9.210, 11.345, 13.277, 15.086, 16.812, 18.475, 20.090, 21.666,
			23.209, 24.725, 26.217, 27.688, 29.141, 30.578, 32.000, 33.409, 34.805, 36.191, 37.566, 38.932, 40.289,
			41.638, 42.980, 44.314, 45.642, 46.963, 48.278, 49.588, 50.892, 52.191, 53.486, 54.776, 56.061, 57.342,
			58.619, 59.893, 61.162, 62.428, 63.691, 64.950, 66.206, 67.459, 68.710, 69.957, 71.201, 72.443, 73.683,
			74.919, 76.154, 77.386, 78.616, 79.843, 81.069, 82.292, 83.513, 84.733, 85.950, 87.166, 88.379, 89.591,
			90.802, 92.010, 93.217, 94.422, 95.626, 96.828, 98.028, 99.228, 100.425, 101.621, 102.816, 104.010, 105.202,
			106.393, 107.583, 108.771, 109.958, 111.144, 112.329, 113.512, 114.695, 115.876, 117.057, 118.236, 119.414,
			120.591, 121.767, 122.942, 124.116, 125.289, 126.462, 127.633, 128.803, 129.973, 131.141, 132.309, 133.476,
			134.642, 135.807 };

	/**
	 * Returns the upper-tail critical values of the Chi-Square distribution. The
	 * upper-tail critical values of chi-square distribution for # degrees of
	 * freedom below 100 are tabulated exact values (only to significance level
	 * equals to .99 - used on experiment). The values for # degrees of freedom
	 * above 100 are calculated based on Wilson-Hilferty approximation
	 * 
	 * Source to tabulated values:
	 * https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm Access on
	 * 19 nov 2018
	 * 
	 * @param x the significance level
	 * @param v the number of degrees of freedom
	 * @return the critical value of chi-square distribution
	 */
	public static double upperTailofChiSquaredDistribution(double x, double v) {
		if (x < 0.0 || v < 1.0) {
			return 0.0;
		}
		if (x == 0.99 && v <= 100) {
			return V99[(int) (v - 1)];
		}
		return v * Math.pow((1 - (2 / (9 * v)) + normalInverse(x) * Math.sqrt(2 / (9 * v))), 3);
	}

	/**
	 * measureScore returns the log of the quality of a network (e.g. the posterior
	 * probability of the network, or the MDL value).
	 * 
	 * @param nType score type (Bayes, MDL, etc) to calculate score with m_BayesNet
	 *              network that will measure its quality
	 * @return log score.
	 */
	public static double measureScore(BayesNet m_BayesNet, int nType) {
		try {
			if (m_BayesNet.m_Distributions == null) {
				return 0;
			}
			if (nType < 0) {
				nType = Scoreable.BAYES;
			}
			double fLogScore = 0.0;
			Instances instances = m_BayesNet.m_Instances;
			for (int iAttribute = 0; iAttribute < instances.numAttributes(); iAttribute++) {
				int nCardinality = m_BayesNet.getParentSet(iAttribute).getCardinalityOfParents();
				for (int iParent = 0; iParent < nCardinality; iParent++) {
					fLogScore += ((Scoreable) m_BayesNet.m_Distributions[iAttribute][iParent]).logScore(nType,
							nCardinality);
				}

				switch (nType) {
				case (Scoreable.MDL): {
					fLogScore -= 0.5 * m_BayesNet.getParentSet(iAttribute).getCardinalityOfParents()
							* (instances.attribute(iAttribute).numValues() - 1)
							* Math.log(m_BayesNet.getNumInstances());
				}
					break;
				case (Scoreable.AIC): {
					fLogScore -= m_BayesNet.getParentSet(iAttribute).getCardinalityOfParents()
							* (instances.attribute(iAttribute).numValues() - 1);
				}
					break;
				}
			}
			return fLogScore;
		} catch (ArithmeticException ex) {
			return Double.NaN;
		}
	} // measureBayesScore

	/**
	 */
	static double logLikelihoodScore(BayesNet m_BayesNet) {
		try {
			double llScore = 0.0;
			Instances instances = m_BayesNet.m_Instances;
			for (int iAttribute = 0; iAttribute < instances.numAttributes(); iAttribute++) {
				int nCardinality = m_BayesNet.getParentSet(iAttribute).getCardinalityOfParents();
				for (int iParent = 0; iParent < nCardinality; iParent++) {
					llScore += ((Scoreable) m_BayesNet.m_Distributions[iAttribute][iParent]).logScore(Scoreable.MDL,
							nCardinality);
				}
			}
			return llScore;
		} catch (ArithmeticException ex) {
			return Double.NaN;
		}
	} // measureBayesScore

	/**
	 */
	public static double measureMDLScore(BayesNet m_BayesNet) {
		try {
			if (m_BayesNet.m_Distributions == null) {
				return 0;
			}
			double llScore = logLikelihoodScore(m_BayesNet);
			double b = 0.0;
			Instances instances = m_BayesNet.m_Instances;
			for (int iAttribute = 0; iAttribute < instances.numAttributes(); iAttribute++) {
				b = b + (instances.attribute(iAttribute).numValues() - 1)
						* m_BayesNet.getParentSet(iAttribute).getCardinalityOfParents();
			}
			int numInstances = m_BayesNet.m_Instances.size();
			double mdlScore = llScore - (0.5 * Math.log(numInstances) * b);
			return mdlScore;
		} catch (ArithmeticException ex) {
			return Double.NaN;
		}
	} // measureBayesScore

	/**
	 */
	public static double measureMITScore(BayesNet m_BayesNet) {
		try {
			double mitScore = 0.0;
			Instances instances = m_BayesNet.m_Instances;
			for (int iAttribute = 0; iAttribute < instances.numAttributes(); iAttribute++) {
				double[] iAttributeValues = instances.attributeToDoubleArray(iAttribute);
				int nConfigAttribute = instances.attribute(iAttribute).numValues();
				double[] iParentValues = new double[instances.size()];
				int nConfigParent = 0;
				for (int i = 0; i < m_BayesNet.getParentSet(iAttribute).getNrOfParents(); i++) {
					if (i == 0) {
						iParentValues = instances
								.attributeToDoubleArray(m_BayesNet.getParentSet(iAttribute).getParent(i));
						nConfigParent = instances.attribute(m_BayesNet.getParentSet(iAttribute).getParent(i))
								.numValues();
					} else {
						double[] mergedValues = new double[iParentValues.length];
						double[] iMergeMiddleValues = instances
								.attributeToDoubleArray(m_BayesNet.getParentSet(iAttribute).getParent(i));
						int nMergeConfigMiddle = instances.attribute(m_BayesNet.getParentSet(iAttribute).getParent(i))
								.numValues();
						ProbabilityState.mergeArrays(iParentValues, iMergeMiddleValues, mergedValues);
						iParentValues = mergedValues;
						nConfigParent = (nConfigParent + nMergeConfigMiddle) / 2;
					}
				}
				mitScore += 2 * instances.numInstances()
						* MutualInformation.calculateMutualInformation(iAttributeValues, iParentValues)
						- upperTailofChiSquaredDistribution(0.99, (nConfigAttribute - 1) * (nConfigParent - 1));
			}
			return mitScore;
		} catch (ArithmeticException ex) {
			return Double.NaN;
		}
	} // measureBayesScore

	public static double calculateLogLoss(BayesNet bayesNet, Instances m_Instances, int max_k) throws Exception {
		double logLoss = 0;
		m_Instances.setClassIndex(m_Instances.numAttributes() - 1);
		bayesNet.m_Instances.setClassIndex(m_Instances.numAttributes() - 1);
		boolean isBinaryClass = false;
		if (m_Instances.numClasses() == 2) {
			isBinaryClass = true;
		}
		for (int i = max_k; i < m_Instances.size(); i++) {
			Instance randonInstance = m_Instances.get(i);
			double[] distributionForInstance = distributionForInstance(bayesNet, randonInstance);
			int classValue = (int) randonInstance.classValue();
			double prediction = 0.0;
			if (isBinaryClass) {
				int predictValue = (int) classifyInstance(bayesNet, randonInstance);
				if (classValue != predictValue) {
					prediction = 1 - distributionForInstance[classValue];
				} else {
					prediction = distributionForInstance[classValue];
				}
			} else {
				prediction = distributionForInstance[classValue];
			}
			if (prediction != 0) {
				logLoss += Math.log(prediction);
			} else {
				logLoss += 0;
			}
		}
		double size = m_Instances.size() - max_k;
		return logLoss * (-1 / size);
	}

	public static double classifyInstance(BayesNet bayesNet, Instance instance) throws Exception {
		double[] dist = distributionForInstance(bayesNet, instance);
		if (dist == null) {
			throw new Exception("Null distribution predicted");
		}
		switch (instance.classAttribute().type()) {
		case Attribute.NOMINAL:
			double max = 0;
			int maxIndex = 0;

			for (int i = 0; i < dist.length; i++) {
				if (dist[i] > max) {
					maxIndex = i;
					max = dist[i];
				}
			}
			if (max > 0) {
				return maxIndex;
			} else {
				return Utils.missingValue();
			}
		case Attribute.NUMERIC:
		case Attribute.DATE:
			return dist[0];
		default:
			return Utils.missingValue();
		}
	}

	public static double[] distributionForInstance(BayesNet bayesNet, Instance instance) throws Exception {
		instance = normalizeInstance(bayesNet, instance);
		BayesNetEstimator m_BayesNetEstimator = bayesNet.getEstimator();
		return m_BayesNetEstimator.distributionForInstance(bayesNet, instance);
	}

	/**
	 * ensure that all variables are nominal and that there are no missing values
	 * 
	 * @param instance instance to check and quantize and/or fill in missing values
	 * @return filtered instance
	 * @throws Exception if a filter (Discretize, ReplaceMissingValues) fails
	 */
	protected static Instance normalizeInstance(BayesNet bayesNet, Instance instance) throws Exception {
		return instance;
	} // normalizeInstance

	public static double calculateLogLossWithMultiClass(BayesNet bayesNet, Instances m_Instances, int max_k)
			throws Exception {
//		int[] classesIndex = getAlarmClassesIndex();
		int[] classesIndex = getAsiaClassesIndex();
		double logLoss = 0;

		for (int j = max_k; j < m_Instances.size(); j++) {
			double logLossAux = 0;
			for (int i : classesIndex) {
				m_Instances.setClassIndex(i);
				bayesNet.m_Instances.setClassIndex(i);
				boolean isBinaryClass = false;
				if (m_Instances.numClasses() == 2) {
					isBinaryClass = true;
				}
				Instance randonInstance = m_Instances.get(j);
				double[] distributionForInstance = distributionForInstance(bayesNet, randonInstance);
				int classValue = (int) randonInstance.classValue();
				double prediction = 0.0;
				if (isBinaryClass) {
					int predictValue = (int) classifyInstance(bayesNet, randonInstance);
					if (classValue != predictValue) {
						prediction = 1 - distributionForInstance[classValue];
					} else {
						prediction = distributionForInstance[classValue];
					}
				} else {
					prediction = distributionForInstance[classValue];
				}
				if (prediction != 0) {
					logLossAux += Math.log(prediction);
				} else {
					logLossAux += 0;
				}
			}
			logLoss += logLossAux * (-1 / (double) classesIndex.length);
		}
		double size = m_Instances.size() - max_k;
		return logLoss * (-1 / size);
	}

	private static int[] getAlarmClassesIndex() {
		int[] classesIndex = new int[8];
		classesIndex[0] = 16;
		classesIndex[1] = 17;
		classesIndex[2] = 18;
		classesIndex[3] = 19;
		classesIndex[4] = 20;
		classesIndex[5] = 21;
		classesIndex[6] = 22;
		classesIndex[7] = 23;
		return classesIndex;
	}
	
	private static int[] getAsiaClassesIndex() {
		int[] classesIndex = new int[2];
		classesIndex[0] = 6;
		classesIndex[1] = 7;
		return classesIndex;
	}

	public static int[] dataPredConfusionMatrix;
	public static int[] dataRealConfusionMatrix;

	public static double calculateAccurary(BayesNet bayesNet, Instances m_Instances, int max_k) throws Exception {
		double TP = 0;
		double FP = 0;

		m_Instances.setClassIndex(m_Instances.numAttributes() - 1);
		bayesNet.m_Instances.setClassIndex(m_Instances.numAttributes() - 1);

		// Confusion matrix
		dataPredConfusionMatrix = new int[m_Instances.size() - max_k];
		dataRealConfusionMatrix = new int[m_Instances.size() - max_k];

		for (int i = max_k; i < m_Instances.size(); i++) {
			Instance randomInstance = m_Instances.get(i);
			int classValue = (int) randomInstance.classValue();
			int classifyInstance = (int) classifyInstance(bayesNet, randomInstance);
			if (classValue == classifyInstance) {
				TP++;
			} else {
				FP++;
			}

			dataPredConfusionMatrix[i - max_k] = classifyInstance;
			dataRealConfusionMatrix[i - max_k] = classValue;
		}
		return (TP) / (TP + FP);
	}

	public static double calculateAccuraryWithMultiClass(BayesNet bayesNet, Instances m_Instances, int max_k, int indexClassToConfMtrx)
			throws Exception {
//		int[] classesIndex = getAlarmClassesIndex();
		int[] classesIndex = getAsiaClassesIndex();

		// Confusion matrix
		dataPredConfusionMatrix = new int[m_Instances.size() - max_k];
		dataRealConfusionMatrix = new int[m_Instances.size() - max_k];

		double accuracy = 0;

		for (int i : classesIndex) {
			m_Instances.setClassIndex(i);
			bayesNet.m_Instances.setClassIndex(i);
			double TP = 0;
			double FP = 0;
			for (int j = max_k; j < m_Instances.size(); j++) {
				Instance randomInstance = m_Instances.get(j);
				int classValue = (int) randomInstance.classValue();
				int classifyInstance = (int) classifyInstance(bayesNet, randomInstance);
				if (classValue == classifyInstance) {
					TP++;
				} else {
					FP++;
				}
				
				if (i == classesIndex[indexClassToConfMtrx]) {
					dataPredConfusionMatrix[j - max_k] = classifyInstance;
					dataRealConfusionMatrix[j - max_k] = classValue;
				}
			}
			accuracy += (TP) / (TP + FP);
		}
		return accuracy / classesIndex.length;
	}

	/**
	 * Updates the classifier with the given instance.
	 * 
	 * @param instance the new training instance to include in the model
	 * @throws Exception if the instance could not be incorporated in the model.
	 */
	static void updateClassifier(BayesNet bayesNet, Instance instance) throws Exception {
		instance = normalizeInstance(bayesNet, instance);
		BayesNetEstimator m_BayesNetEstimator = bayesNet.getEstimator();
		m_BayesNetEstimator.updateClassifier(bayesNet, instance);
	}

	/**
	 * accuracyIncrease determines how much the accuracy estimate should be
	 * increased due to the contribution of a single given instance.
	 * 
	 * @param instance : instance for which to calculate the accuracy increase.
	 * @return increase in accuracy due to given instance.
	 * @throws Exception passed on by distributionForInstance and classifyInstance
	 */
	static double accuracyIncrease(BayesNet bayesNet, Instance instance) throws Exception {
		double[] fProb = distributionForInstance(bayesNet, instance);
		return fProb[(int) instance.classValue()] * instance.weight();
	}

	public static List<Integer> getFromFile() {
		List<Integer> randoms = new ArrayList<Integer>();
		Scanner scanner;
		try {
			scanner = new Scanner(new File("docs/random1728.txt"));
			while (scanner.hasNextInt()) {
				randoms.add(scanner.nextInt());
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return randoms;
	}
}
