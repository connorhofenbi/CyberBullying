package cyberMain;

import java.awt.BorderLayout;
import java.io.FileNotFoundException;

import javax.swing.JFrame;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.meta.Bagging;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Utils;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class MainRunnable {
	
	
	private static DataHandler originalData;
	private static Instances[][] testtrainSetsIntro;
	private static Instances[][] testtrainSetsExtro;
	private static ClassificationAndValidation introvertModel;
	private static ClassificationAndValidation extrovertModel;
	private final static String dataLink = "/Users/connorhofenbitzer/Desktop/sortedData.arff";

	public static void main(String[] args) {
		
		//load the data
		originalData = new DataHandler(dataLink);
		introvertModel = new ClassificationAndValidation();
		extrovertModel = new ClassificationAndValidation();

		try {
			
			//set the class index
			originalData.getDataset().setClassIndex(0);
			
			//split into stratas
			Instances[] dataSplitIntoStratas = originalData.createDemoStratas();
			
			//run cross validation splits
			testtrainSetsIntro = introvertModel.crossValSplit(dataSplitIntoStratas[0], 10);
			testtrainSetsExtro = extrovertModel.crossValSplit(dataSplitIntoStratas[1], 10);

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		//models to test based of haungs paper
		Classifier[] models = {
				new J48(),
				new RandomForest(),
				new Bagging(),
				new NaiveBayes(),
				new ZeroR()
		};
		
		//for generating ROC
		ThresholdCurve introVertROC = null;
		ThresholdCurve extroVertROC = null;
		
		//actually run the models
		for(Classifier i : models) {
			Evaluation introValidate = null;
			Evaluation extroValidate = null;

			//evaluate the introvert model
			for(int h = 0; h < testtrainSetsIntro[0].length; h++) {
				introValidate = introvertModel.classify(i, testtrainSetsIntro[1][h], testtrainSetsIntro[0][h]);
			}
			
			//evaluate the extravert model
			for(int f = 0; f < testtrainSetsExtro[0].length; f++) {
				extroValidate = extrovertModel.classify(i, testtrainSetsExtro[1][f], testtrainSetsExtro[0][f]);	
			}
			
			//generate ROC curves
			introVertROC = new ThresholdCurve();
			extroVertROC = new ThresholdCurve();
			
			//generate the acutal curve
			Instances introvertCurve = introVertROC.getCurve(introValidate.predictions(), 0);
			Instances extrovertCurve = extroVertROC.getCurve(extroValidate.predictions(), 0);
			
			//print out the model
			System.out.println("\n Classifier : " + i.getClass().getName());
			System.out.printf("Introvert accuracy %.2f Root mean squared error %.2f Area under ROC %.2f \n" ,introValidate.pctCorrect(), introValidate.rootMeanSquaredError(), introVertROC.getROCArea(introvertCurve));
			System.out.printf("Extravert accuracy %.2f Root mean squared error %.2f Area under ROC %.2f \n" ,extroValidate.pctCorrect(), extroValidate.rootMeanSquaredError(), extroVertROC.getROCArea(extrovertCurve));
			
			//show the ROC curve
			createVisuilizedGraph(introVertROC, extroVertROC , introvertCurve, extrovertCurve, i.getClass().getName() );
		}
	}	

	/**
	 * Creates a visualization board for ROC curves
	 * @param introvert : introvert data threshold curve
	 * @param extrovert : extrovert data threshold curve
	 * @param introvertCurve : introvert ROC calculator
	 * @param extrovertCurve : extrovert ROC calculator
	 * @param name : name of the plot
	 */
	private static void createVisuilizedGraph(ThresholdCurve introvert, ThresholdCurve extrovert, 
												Instances introvertCurve, Instances extrovertCurve, String name) {
		
		ThresholdVisualizePanel threshold = new ThresholdVisualizePanel();
		threshold.setROCString("Intro ROC area = " + Utils.doubleToString(introvert.getROCArea(introvertCurve), 4)  + 
								" Extro ROC area = " +  Utils.doubleToString(extrovert.getROCArea(extrovertCurve), 4));
		threshold.setName(introvertCurve.relationName());
		
		PlotData2D introvertData = createNewCurvePlot(introvertCurve);
		PlotData2D extrovertData = createNewCurvePlot(extrovertCurve);
		
		try {
			threshold.addPlot(introvertData);
			threshold.addPlot(extrovertData);
			
			JFrame jf = new JFrame("WEKA ROC " + name);
			jf.setSize(500, 400);
			jf.getContentPane().setLayout(new BorderLayout());
			jf.getContentPane().add(threshold, BorderLayout.CENTER);
			jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
			jf.setVisible(true);
		} catch(Exception e ) {
			
		}
		
	}
	
	/**
	 * Create a new plot given a curve
	 * @param curve : plot instances 
	 * @return : a new plot
	 */
	private static PlotData2D createNewCurvePlot(Instances curve) {
		PlotData2D data = new PlotData2D(curve);
		data.setPlotName(curve.relationName());
		data.addInstanceNumberAttribute();
		
		boolean[] connected = new boolean[curve.numInstances()];
		for(int i = 1; i < connected.length; i++) {
			connected[i] = true;
		}
		
		try {
			data.setConnectPoints(connected);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return data;
	}

}

