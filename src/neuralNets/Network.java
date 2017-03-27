package neuralNets;
import java.util.Random;

public class Network {
	
	private int numLayers;
	private double[][] biases; 
	private double weights[][][];
	
	public Network(int[] sizes)
	{
		//Random number generator
		Random rng = new Random();
		//Number of layers of neurons
		numLayers = sizes.length;
		//The biases and weights of each neuron in each layer
		biases = new double[numLayers][];
		weights = new double[numLayers][][];
		
		//Initializes the biases of each neuron in each layer
		for(int layer = 1; layer < numLayers; layer++) {
			biases[layer] = new double[sizes[layer]];
			for(int neuron = 0; neuron < sizes[layer]; neuron++) {
				biases[layer][neuron] = rng.nextGaussian();
			}
		}
		
		for(int layer = 1; layer < numLayers; layer++) {
			//Initialize each layer to the number of neurons in the layer
			weights[layer] = new double[sizes[layer]][];
			for(int neuron = 0; neuron < sizes[layer]; neuron++) {
				//Each neuron has a weight for each neuron in the previous layer
				weights[layer][neuron] = new double[sizes[layer - 1]];
				for(int weight = 0; weight < sizes[layer - 1]; weight++) {
					//Each weight is initialized to a value between 0 and 1
					weights[layer][neuron][weight] = rng.nextGaussian();
				}
			}
		}
	}
	

	public static void main(String[] args) {
		

	}

}
