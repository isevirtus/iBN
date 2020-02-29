package iBN.core;

import weka.core.Instance;
import weka.core.Instances;

public class InstancesHelper {

	/**
	 * Similar -> Dissimilar
	 * @param instances
	 * @return
	 */
	public static Instances sortInstancesByInstance(Instances instances) {
		int nRandon = 1000;

		Instance randomInstance = instances.get(nRandon);
		int[] scores = new int[instances.size()];
		Instance[] backup = new Instance[instances.size()];
		int j = 0;
		for (Instance inst : instances) {
			backup[j] = inst;
			scores[j] = calcScore(inst, randomInstance);
			j++;
		}
		
		instances.delete();
		
		int aux = instances.numAttributes();
		while (aux > -1) {
			for (int i = 0; i < scores.length; i++) {
				if (scores[i] == aux) {
//					System.out.println(aux);
					instances.add(backup[i]);
				}
			}
			aux--;
		}
		return instances;
	}
	
	/**
	 * Dissimilar -> Similar 
	 * @param instances
	 * @return
	 */
	public static Instances reverseSortInstancesByInstance(Instances instances) {
		int nRandon = 1000;

		Instance randomInstance = instances.get(nRandon);
		int[] scores = new int[instances.size()];
		Instance[] backup = new Instance[instances.size()];
		int j = 0;
		for (Instance inst : instances) {
			backup[j] = inst;
			scores[j] = calcScore(inst, randomInstance);
			j++;
		}
		
		instances.delete();
		
		int aux = 0;
		while (aux <= instances.numAttributes()) {
			for (int i = 0; i < scores.length; i++) {
				if (scores[i] == aux) {
//					System.out.println(aux);
					instances.add(backup[i]);
				}
			}
			aux++;
		}
		return instances;
	}

	private static int calcScore(Instance inst, Instance randomInstance) {
		int score = 0;
		for (int i = 0; i < randomInstance.numValues(); i++) {
			if (randomInstance.value(i) == inst.value(i)) {
				score++;
			}
		}
		return score;
	}

	public static void printScores(Instances instances) {
		int nRandon = 1;
		Instance randomInstance = instances.get(nRandon);
		for (Instance inst : instances) {
			System.out.println(calcScore(inst, randomInstance));
		}
	}

}
