package cyberMain;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class DataHandler {
	
	private int bullyIndex;
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
			
			//grabs the index when bullying ends
			bullyIndex = getSplit(dataSet, "0", 0);
			
			
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

	public double calculateMean() {
		
		double totaledAttribute = 0.0;
		
		//get the index for outdegree centrality 
		int attIndex = dataSet.attribute("outdegree_centrality_for_reciever").index();
		for(int i = 0; i < dataSet.size(); i++) {
			totaledAttribute += dataSet.get(i).value(attIndex);
		}
		return totaledAttribute/dataSet.size();
	}
	
	/**
	 * 
	 * This will be used later
	 * Need to implement folds into creating new sets
	 * 
	 */
	
	/**
	 * Creates the training and testing datasets
	 * @param trainingPercent : percent of the overall dataset to train from 0 to 1
	 * @return index 0 is the training set, index 1 is the testing set
	 */
	public Instances[] createSets(double trainingPercent) {
		
		Instances[] output = new Instances[2];

		//create the training set
		output[0] = createNewSet(trainingPercent);

		//create the testing set
		output[1] = createNewSet(1-trainingPercent);
		
		//return the two instances
		return output;	
	}
	
	/**
	 * Finds a split in the dataset after it finds a given attribute 
	 * @param checkFeature : the feature to check for (pass through numbers as strings)
	 * @param attributeIndex : the index of the attribute searching for
	 * @return : the position of the split
	 */
	private int getSplit(Instances data , String checkFeature, int attributeIndex) {
		
		//iterate through the dataset
		for(int i = 0; i < data.numInstances(); i++) {
			
			//find where the first index of the split is
			if(data.instance(i).toString(attributeIndex).equals(checkFeature)) {
				return i;
			}
		}
		return -1;
	}
	
	/**
	 * Creates a subset of the overall data without replacement
	 * when you delete an index all other indecies shift
	 * so it is without replacement
	 * 
	 * It breaks the data into stratas and takes a percent of each
	 * strata
	 * 
	 * Using this instead of a cross validation split for more control over the splits later down the road
	 * 
	 * @param percent: percent of the full dataset to be contained
	 * @return : a testing dataset
	 */
	private Instances createNewSet(double percent) {

		Instances newSet = null;
		try {
			//using the method so if I need to change how
			//the data is distributed or output it also changes
			//that here
			newSet = this.getDataset();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		Random randomGen = new Random();

		int currIndex = 0;
		int countingsize = (int)(bullyIndex*(1-percent));
		
		//first remove from the bullying instances
		while (currIndex < countingsize) {
			int currRandom = randomGen.nextInt(bullyIndex);
			newSet.delete(currRandom);
			currIndex++;
		}
		
		//indecies have changed so have to recalculate here
		bullyIndex = getSplit(newSet, "0", 0);

		//reinitialize values
		currIndex = 0;
		randomGen = new Random();
		countingsize = (int)( (newSet.numInstances()-bullyIndex)*(percent));
		
		//now remove from the nonBullying numbers
		//did this randomization differently because these random numbers had to be calculated on the fly
		while(currIndex < countingsize) {
			
			//get the next random number
			int currRandom = bullyIndex + randomGen.nextInt((newSet.numInstances() - bullyIndex));
			
			//delete that number and let everything shift
			newSet.delete(currRandom);	
			currIndex++;
		}
		
		return newSet;
	}
	
	
}
