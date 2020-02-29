package iBN.classifiers.bayes.net.structure.search.local;

import java.util.LinkedHashSet;
import java.util.Set;

import org.jgrapht.alg.interfaces.SpanningTreeAlgorithm.SpanningTree;
import org.jgrapht.alg.spanning.KruskalMinimumSpanningTree;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.WeightedPseudograph;
import org.jgrapht.traverse.DepthFirstIterator;
import org.jgrapht.traverse.GraphIterator;

import JavaMI.MutualInformation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.local.LocalScoreSearchAlgorithm;
import weka.core.Instances;

public class MaximumWeightSpanningTree extends LocalScoreSearchAlgorithm{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	@Override
	public void buildStructure(BayesNet bayesNet, Instances instances) throws Exception {
		WeightedPseudograph<Integer, DefaultWeightedEdge> weightedGraph = new WeightedPseudograph<>(DefaultWeightedEdge.class);
		KruskalMinimumSpanningTree<Integer, DefaultWeightedEdge> mstAlgo = new KruskalMinimumSpanningTree<>(weightedGraph);
			
		for (int i = 0; i < instances.numAttributes(); i++) { //start addition vertex to WeightedPseudograph
			weightedGraph.addVertex(i);
		} //end addition vertex
		
		for (int i = 0; i < instances.numAttributes(); i++) { //start addition of weight (MI) to each possible edge 
			for (int j = i + 1; j < instances.numAttributes(); j++) {
				DefaultWeightedEdge e = new DefaultWeightedEdge();
				weightedGraph.addEdge(i, j, e);
				double mi = -1 * MutualInformation.calculateMutualInformation(instances.attributeToDoubleArray(i),
						instances.attributeToDoubleArray(j));
				weightedGraph.setEdgeWeight(e, mi);
			}
		} //end start addition of weight
		
		SpanningTree<DefaultWeightedEdge> mst = mstAlgo.getSpanningTree();
		Set<DefaultWeightedEdge> edges = mst.getEdges();
		int[] edgeCount = new int[instances.numAttributes()];
		for (DefaultWeightedEdge edge : edges) {
			edgeCount[weightedGraph.getEdgeSource(edge)]++;
			edgeCount[weightedGraph.getEdgeTarget(edge)]++;
		}
		Set<DefaultWeightedEdge> removeEdges = new LinkedHashSet<DefaultWeightedEdge>(weightedGraph.edgeSet());
		removeEdges.removeAll(edges);
		weightedGraph.removeAllEdges(removeEdges);
		GraphIterator<Integer, DefaultWeightedEdge> iterator = new DepthFirstIterator<Integer, DefaultWeightedEdge>(
				weightedGraph);
		
		bayesNet.m_Instances = instances;
		bayesNet.initStructure();
		
		Integer v;
		Integer t;
		while (iterator.hasNext()) {
			v = iterator.next();
			edges = weightedGraph.edgesOf(v);
			for (DefaultWeightedEdge edge : edges) {
				t = weightedGraph.getEdgeSource(edge);
				if (t == v)
					t = weightedGraph.getEdgeTarget(edge);
				bayesNet.getParentSet(v).addParent(t, instances);
			}
		}
	}
}
