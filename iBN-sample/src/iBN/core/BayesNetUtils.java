package iBN.core;

import java.awt.BorderLayout;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.BIFReader;
import weka.classifiers.bayes.net.search.local.HillClimber;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.gui.graphvisualizer.BIFFormatException;
import weka.gui.graphvisualizer.GraphVisualizer;

public class BayesNetUtils {

	public static List<Integer> iHeadNeighbors;
	public static List<Integer> iTailNeighbors;

	public static String getBatchBIFFVersion(String biffFile, Instances instances) throws Exception {
		BayesNet bayesNet = (BayesNet) new BIFReader().processFile(biffFile);
		bayesNet.m_Instances = instances;
		HillClimber climber = new HillClimber();
		climber.setUseArcReversal(true);
		climber.setInitAsNaiveBayes(false);
		climber.setMaxNrOfParents(100000);
		climber.buildStructure(bayesNet, instances);
		bayesNet.initCPTs();
		bayesNet.estimateCPTs();
//		Instances instances2Acc = new Instances(instances);
//		System.out.println("(BATCH)" + " " + IncStatistics.measureMITScore(bayesNet) + " " + IncStatistics.measureMDLScore(bayesNet) + " " + IncStatistics.calculateLogLossWithMultiClass(bayesNet, instances) + " " + IncStatistics.calculateAccuraryWithMultiClass(bayesNet, instances2Acc, 0.33));
//		BayesNetUtils.visualizeBayesNet(bayesNet, "Versão Final (HCS em Lote)");
//		System.out.println(IncStatistics.calculateLogLossWithMultiClass(bayesNet, instances));
		return bayesNet.toXMLBIF03();

	}
	
	public static String getBIFFVersion(String biffFile) throws Exception {
		BayesNet bayesNet = (BayesNet) new BIFReader().processFile(biffFile);
		return bayesNet.toXMLBIF03();

	}

	public static BayesNet revert(BayesNet bayesNet) throws Exception {
		BayesNet bayesNetReversed = new BayesNet();
		bayesNetReversed.m_Instances = bayesNet.m_Instances;
		bayesNetReversed.initStructure();
		for (int i = 0; i < bayesNet.getNrOfNodes(); i++) {
			for (int j = 0; j < bayesNet.getNrOfParents(i); j++) {
				int p = bayesNet.getParent(i, j);
				bayesNetReversed.getParentSet(p).addParent(i, bayesNetReversed.m_Instances);
			}
		}
		return bayesNetReversed;
	}

	public static void neighborsOnPath(BayesNet bayesNet, int iHead, int iTail) {
		iHeadNeighbors = new ArrayList<Integer>();
		iTailNeighbors = new ArrayList<Integer>();
		boolean[] isVisited = new boolean[bayesNet.getNrOfNodes()];
		ArrayList<Integer> pathList = new ArrayList<Integer>();
		pathList.add(iHead); // add source to path[]
		getAllPathsUtil(iHead, iTail, isVisited, pathList, bayesNet); // Call recursive utility
	}

	private static void getAllPathsUtil(Integer u, Integer d, boolean[] isVisited, List<Integer> localPathList,
			BayesNet bayesNet) {
		isVisited[u] = true; // Mark the current node
		if (u.equals(d)) {
//            System.out.println(localPathList); //path between two nodes
			if (localPathList.size() > 2) {
				iHeadNeighbors.add(localPathList.get(1));
				iTailNeighbors.add(localPathList.get(localPathList.size() - 2));
			}
		}
		for (int i = 0; i < bayesNet.getParentSet(u).getNrOfParents(); i++) { // Recur for all the vertices adjacent to
																				// current vertex
			int p = bayesNet.getParent(u, i);
			if (!isVisited[p]) {
				localPathList.add(p); // store current node in path[]
				getAllPathsUtil(p, d, isVisited, localPathList, bayesNet);
				localPathList.remove(localPathList.indexOf(p)); // remove current node in path[]
			}
		}
		isVisited[u] = false; // Mark the current node
	}

	public static int getIndexByName(BayesNet bayesNet, String name) {
		for (int i = 0; i < bayesNet.getNrOfNodes(); i++) {
			if (bayesNet.getNodeName(i).equals(name)) {
				return i;
			}
		}
		return Integer.MIN_VALUE;
	}

	/**
	 * Pops up a GraphVisualizer for the BayesNet classifier from the currently
	 * selected item in the results list.
	 * 
	 * @param XMLBIF    the description of the graph in XMLBIF ver. 0.3
	 * @param graphName the name of the graph
	 */
	public static void visualizeBayesNet(BayesNet bayesNet, String graphName) {
		String XMLBIF = bayesNet.toXMLBIF03();
		final javax.swing.JFrame jf = new javax.swing.JFrame("Weka Classifier Graph Visualizer: " + graphName);
		jf.setSize(500, 400);
		jf.getContentPane().setLayout(new BorderLayout());
		GraphVisualizer gv = new GraphVisualizer();
		try {
			gv.readBIF(XMLBIF);
		} catch (BIFFormatException be) {
			System.err.println("unable to visualize BayesNet");
			be.printStackTrace();
		}
		gv.layoutGraph();

		jf.getContentPane().add(gv, BorderLayout.CENTER);
		jf.addWindowListener(new java.awt.event.WindowAdapter() {
			@Override
			public void windowClosing(java.awt.event.WindowEvent e) {
				jf.dispose();
			}
		});

		jf.setVisible(true);
	}

	public static Instances reorderInstanceIfNedded(BayesNet initialBayesNet, Instances instances) throws Exception {
		int[] s = new int[initialBayesNet.getNrOfNodes()];
		for (int i = 0; i < initialBayesNet.getNrOfNodes(); i++) {
			for (int j = 0; j < instances.numAttributes(); j++) {
				if (initialBayesNet.getNodeName(i).equals(instances.attribute(j).name())) {
					s[i] = j + 1;
				}
			}
		}
		instances = reorderLabels(instances, s);
		return instances;
	}

	/**
	 * ReorderLabels - swap values of y[1] to y[L] according to s[].
	 * 
	 * @param s new indices order (supposing that it contains the first s.length
	 *          indices)
	 */
	private static Instances reorderLabels(Instances instances, int s[]) throws Exception {
		int L = s.length;
		Reorder f = new Reorder();
		String range = "";
		for (int j = 0; j < L - 1; j++) {
			range += String.valueOf(s[j]) + ",";
		}
		range = range + String.valueOf(s[L - 1]);
		f.setAttributeIndices(range);
		f.setInputFormat(instances);
		instances = Filter.useFilter(instances, f);
		return instances;
	}

	public static void generateScrumBNXML() {
		List<String> nodeNames = new ArrayList<String>();
//		nodeNames.add("Complexidade de Requisitos");
//		nodeNames.add("Ambiente de Negócio");
//		nodeNames.add("Inovação de Requisitos");
//		nodeNames.add("Volatilidade de Requisitos");
//		nodeNames.add("Qualidade do Product Owner");
//		nodeNames.add("Inspenção");
//		nodeNames.add("Monitoramento do Progresso");
//		nodeNames.add("Adaptação");
//		nodeNames.add("Reunião de Revisão");
//		nodeNames.add("Definição Inicial do Backlog do Produto");
//		nodeNames.add("Detalhado");
//		nodeNames.add("Estimado");
//		nodeNames.add("Ordenado");
//		nodeNames.add("Backlog do Produto");
//		nodeNames.add("Efetividade da Definição Contínua do Backlog do Produto");
//		nodeNames.add("Desempenho Passado");
//		nodeNames.add("Capacidade Projetada");
//		nodeNames.add("Backlog da Sprint");
//		nodeNames.add("Objetivo da Sprint");
//		nodeNames.add("Reunião de Planejamento");
//		nodeNames.add("Autonomia");
//		nodeNames.add("Coesão");
//		nodeNames.add("Auto-gerenciamento");
//		nodeNames.add("Adaptabilidade");
//		nodeNames.add("Compartilhamento de Liderança");
//		nodeNames.add("Conhecimento");
//		nodeNames.add("Colaboração");
//		nodeNames.add("Orientação");
//		nodeNames.add("Personalidade");
//		nodeNames.add("Coordenação");
//		nodeNames.add("Comunicação");
//		nodeNames.add("Distribuição");
//		nodeNames.add("Canais");
//		nodeNames.add("Reunião Diária");
//		nodeNames.add("Monitoramento");
//		nodeNames.add("Presença");
//		nodeNames.add("Trabalho em Equipe");
//		nodeNames.add("Qualidade do Processo de Desenvolvimento e Teste");
//		nodeNames.add("Qualidade da Execução da Sprint");
//		nodeNames.add("Qualidade do Incremento");
//		nodeNames.add("Novos Itens de Backlog Prontos");
//		nodeNames.add("Satisfação do Cliente");

		// Alarm
		nodeNames.add("HISTORY");
		nodeNames.add("CVP");
		nodeNames.add("PCWP");
		nodeNames.add("HYPOVOLEMIA");
		nodeNames.add("LVEDVOLUME");
		nodeNames.add("LVFAILURE");
		nodeNames.add("STROKEVOLUME");
		nodeNames.add("ERRLOWOUTPUT");
		nodeNames.add("HRBP");
		nodeNames.add("HREKG");
		nodeNames.add("ERRCAUTER");
		nodeNames.add("HRSAT");
		nodeNames.add("INSUFFANESTH");
		nodeNames.add("ANAPHYLAXIS");
		nodeNames.add("TPR");
		nodeNames.add("EXPCO2");
		nodeNames.add("KINKEDTUBE");
		nodeNames.add("MINVOL");
		nodeNames.add("FIO2");
		nodeNames.add("PVSAT");
		nodeNames.add("SAO2");
		nodeNames.add("PAP");
		nodeNames.add("PULMEMBOLUS");
		nodeNames.add("SHUNT");
		nodeNames.add("INTUBATION");
		nodeNames.add("PRESS");
		nodeNames.add("DISCONNECT");
		nodeNames.add("MINVOLSET");
		nodeNames.add("VENTMACH");
		nodeNames.add("VENTTUBE");
		nodeNames.add("VENTLUNG");
		nodeNames.add("VENTALV");
		nodeNames.add("ARTCO2");
		nodeNames.add("CATECHOL");
		nodeNames.add("HR");
		nodeNames.add("CO");
		nodeNames.add("BP");

		System.out.println("<?xml version=\"1.0\"?>\n" + "<!-- DTD for the XMLBIF 0.3 format -->\n"
				+ "<!DOCTYPE BIF [\n" + "	<!ELEMENT BIF ( NETWORK )*>\n"
				+ "	      <!ATTLIST BIF VERSION CDATA #REQUIRED>\n"
				+ "	<!ELEMENT NETWORK ( NAME, ( PROPERTY | VARIABLE | DEFINITION )* )>\n"
				+ "	<!ELEMENT NAME (#PCDATA)>\n" + "	<!ELEMENT VARIABLE ( NAME, ( OUTCOME |  PROPERTY )* ) >\n"
				+ "	      <!ATTLIST VARIABLE TYPE (nature|decision|utility) \"nature\">\n"
				+ "	<!ELEMENT OUTCOME (#PCDATA)>\n" + "	<!ELEMENT DEFINITION ( FOR | GIVEN | TABLE | PROPERTY )* >\n"
				+ "	<!ELEMENT FOR (#PCDATA)>\n" + "	<!ELEMENT GIVEN (#PCDATA)>\n" + "	<!ELEMENT TABLE (#PCDATA)>\n"
				+ "	<!ELEMENT PROPERTY (#PCDATA)>\n" + "]>\n" + "\n" + "\n" + "<BIF VERSION=\"0.3\">\n" + "<NETWORK>\n"
				+ "<NAME>Scrum Assessment</NAME>");

		for (int i = 0; i < nodeNames.size(); i++) {
			System.out.println("<VARIABLE TYPE=\"nature\">\n" + "<NAME>" + nodeNames.get(i) + "</NAME>\n"
					+ "<OUTCOME>0</OUTCOME>\n" + "<OUTCOME>1</OUTCOME>\n" + "<OUTCOME>2</OUTCOME>\n"
					+ "<PROPERTY>position = (" + i * 75 + ",10)</PROPERTY>\n" + "</VARIABLE>");
		}

		for (int i = 0; i < nodeNames.size(); i++) {
			System.out.println("<DEFINITION>\n" + "<FOR>" + nodeNames.get(i) + "</FOR>\n" + "<TABLE>\n"
					+ "0.3333333333333333 0.3333333333333333 0.3333333333333333 \n" + "</TABLE>\n" + "</DEFINITION>");
		}

		System.out.println("</NETWORK>\n" + "</BIF>");
	}

	public static String writeBIFFFile(String xmlTxt) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter("bayesnets/to_compare.xml"));
		writer.write(xmlTxt);
		writer.close();
		return "bayesnets/to_compare.xml";
	}

	public static void deleteBIFFFile() {
		File file = new File("bayesnets/to_compare.xml");
		file.delete();
	}

	/**
	 * 
	 * @param oBayesNet rede original
	 * @param lBayesNet rede aprendida
	 * @return
	 * @throws Exception
	 */
	public static double[] precisionRecallFMeasure(BayesNet oBayesNet, BayesNet lBayesNet) throws Exception {
		double nTP = 0;
		double nFN = 0;
		for (int iAttribute = 0; iAttribute < oBayesNet.getNrOfNodes(); iAttribute++) {
			for (int iParent = 0; iParent < oBayesNet.getParentSet(iAttribute).getNrOfParents(); iParent++) {
				int nParent = oBayesNet.getParentSet(iAttribute).getParent(iParent);
				if (lBayesNet.getParentSet(iAttribute).contains(nParent)) {
					nTP++;
				} else {
					nFN++;
				}
			}
		}
		
		double nFP = 0;
		for (int iAttribute = 0; iAttribute < lBayesNet.getNrOfNodes(); iAttribute++) {
			for (int iParent = 0; iParent < lBayesNet.getParentSet(iAttribute).getNrOfParents(); iParent++) {
				int nParent = lBayesNet.getParentSet(iAttribute).getParent(iParent);
				if (!oBayesNet.getParentSet(iAttribute).contains(nParent)) {
					nFP++;
				}
			}
		}
		double precision = nTP/(nTP+nFP);
		double recall = nTP/(nTP+nFN);
		double f_measure = ((precision * recall)/(precision + recall))*2;
		return new double[] {precision, recall, f_measure};
	}

	public static void saveDataToConfusionMatrix(BayesNet bayesNetToBatch, Instances m_Instances, int max_k, String fileName, int indexClassToConfMtrx) throws Exception {
		int[] dataPredConfusionMatrix = IncStatistics.dataPredConfusionMatrix;
		int[] dataRealConfusionMatrix = IncStatistics.dataRealConfusionMatrix;
		
		if (indexClassToConfMtrx != Integer.MIN_VALUE) {
			m_Instances.setClassIndex(indexClassToConfMtrx);
		}
		
		String csv = "Pred,Real\n";
		for (int i = 0; i < dataRealConfusionMatrix.length; i++) {
			csv += m_Instances.classAttribute().value(dataRealConfusionMatrix[i]).toLowerCase() + "," + m_Instances.classAttribute().value(dataPredConfusionMatrix[i]).toLowerCase() + "\n";
		}
		
		BufferedWriter writer = new BufferedWriter(new FileWriter("D:/Documentos/CSV/cM/"+ fileName + ".csv"));
		writer.write(csv);
		writer.close();
		
	}
}
