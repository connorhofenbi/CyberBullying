package cyberMain;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class ClassificationAndValidation {
	
	/**
	 * Classify a model using training and testing sets 
	 * @param model    : model to train on
	 * @param testing  : data to test on 
	 * @param training : data to train on
	 * @return : the evaluation of the model
	 */
	public Evaluation classify(Classifier model, Instances testing, Instances training) {
		
		//evaluation model to use
		Evaluation eval = null;
		try {
			
			// init the models and build classifiers
			eval = new Evaluation(training);
			model.buildClassifier(training);
			eval.evaluateModel(model, testing);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//return the evaluation
		return eval;
	}
	
	
	/**
	 * Run a cross validation split
	 * 
	 * @param data data to split
	 * @param numFolds : number of folds
	 * @return training set and testing set
	 */
	public Instances[][] crossValSplit(Instances data, int numFolds) {
		Instances[][] split = new Instances[2][numFolds];
		for(int i = 0; i < numFolds; i++) {
			split[0][i] = data.trainCV(numFolds, i);
			split[1][i] = data.testCV(numFolds, i);
		}

		return split;
	}
	
	
}
