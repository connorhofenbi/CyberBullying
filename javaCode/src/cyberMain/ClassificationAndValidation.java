package cyberMain;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.filters.supervised.instance.SMOTE;


public class ClassificationAndValidation {
	
	/**
	 * Classify a model using training and testing sets 
	 * @param model    : model to train on
	 * @param testing  : data to test on 
	 * @param training : data to train on
	 * @return : the evaluation of the model
	 */
	public Evaluation classify(Classifier model, Instances data) {
		
		//evaluation model to use
		Evaluation eval = null;		
		
		//encoperate SMOTE
		SMOTE smote = new SMOTE();
		
		
		try {
			
			// init the models and build classifiers
			eval = new Evaluation(data);
			model.buildClassifier(data);
			
			//encoperate SMOTE
			smote.setInputFormat(data);
			FilteredClassifier fc = new FilteredClassifier();
			
			fc.setClassifier(model);
			fc.setFilter(smote);
			
			eval.crossValidateModel(fc, data, 10, new Random(1));
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//return the evaluation
		return eval;
	}
	

}
