package cyberMain;

import java.io.FileNotFoundException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Bagging;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class MainRunnable {
	
	/**
	 * Still need to implement : 
	 * folds to personal split algorithm
	 * visualization of data
	 */
	
	
	private static DataHandler data;
	private static Instances[][] testtrainSets;
	private static ClassificationAndValidation actualModel;
	private final static String dataPath = "/Users/connorhofenbitzer/Desktop/FinalData.arff";
	
	public static void main(String[] args) {
		
		//load the data
		data = new DataHandler(dataPath);
		actualModel = new ClassificationAndValidation();
		
		try {
			//set the class index
			data.getDataset().setClassIndex(0);
			
			//run cross validation splits
			testtrainSets = actualModel.crossValSplit(data.getDataset(), 10);
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
		
		//actually run the model
		for(Classifier i : models) {
			Evaluation validate = null;
			for(int h = 0; h < testtrainSets[0].length; h++) {
				validate = actualModel.classify(i, testtrainSets[1][h], testtrainSets[0][h]);
			}
			
			System.out.printf("model %s accuracy %.2f \n", i,validate.pctCorrect());
		}
	}	

}
