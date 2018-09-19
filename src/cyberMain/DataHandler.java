package cyberMain;

import java.io.File;
import java.io.FileNotFoundException;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class DataHandler {
	
	private Instances dataSet;
	
	public DataHandler(String dataPath) {
		loadDataset(dataPath);
	}
	
	
	/**
	 * Loads a data set at a given path
	 * @param dataPath : path of the data set to load (ARFF format)
	 */
	private void loadDataset(String dataPath) {
		try {
			//create a new ArffLoader to load our data
			ArffLoader loader = new ArffLoader();
			
			//set the source to our data
			loader.setSource(new File(dataPath));
			
			//load the actual dataset
			dataSet = loader.getDataSet();
			
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	

	/**
	 * @return the full dataset
	 * @throws FileNotFoundException
	 */
	public Instances getDataset() throws FileNotFoundException {
		if(dataSet != null) {
			return dataSet;
		}
		throw new FileNotFoundException("please load a file to the dataset");
	}
	
	/**
	 * Splits the data into two stratas of introverts and extroverts
	 * @return : [0] introvert strata [1] extrovert strata
	 */
	public Instances[] createDemoStratas() {
		
		//get the index for outdegree centrality 
		int attIndex = dataSet.attribute("outdegree_centrality_for_reciever").index();
		
		//how we split the data
		double mean = calculateMean();
		
		//copy instances to get attributes aswell
		Instances introverts = new Instances(dataSet);
		Instances extroverts = new Instances(dataSet);
		extroverts.clear();

		//split the data at the mean
		for(int i = 0; i < introverts.size(); i++) {
			
			//remove from introverts and add to extroverts
			if(introverts.get(i).value(attIndex) > mean) {
				extroverts.add(introverts.remove(i));
			}
		}
		

		Instances[] output = new Instances[2];
		output[0] = introverts;
		output[1] = extroverts;
		return output;
		
	}

	/**
	 * Finds the mean of outdegree centrality for reciever
	 * @return : mean of outdegree centrality for reciever
	 */
	public double calculateMean() {
		
		double totaledAttribute = 0.0;
		
		//get the index for outdegree centrality 
		int attIndex = dataSet.attribute("outdegree_centrality_for_reciever").index();
		for(int i = 0; i < dataSet.size(); i++) {
			totaledAttribute += dataSet.get(i).value(attIndex);
		}
		return totaledAttribute/dataSet.size();
	}
	
	
}
