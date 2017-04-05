package neuralNets;
import java.util.Random;
import neuralNets.DataLoader;
import testMnist.TestMnistData;

import cern.colt.function.*;
import cern.colt.matrix.*;
import cern.jet.math.Functions;
import cern.jet.random.Normal;
import cern.jet.random.engine.RandomEngine;

public class Network {
	
	private int numLayers;
	private DoubleMatrix1D[] biases; 
	private DoubleMatrix2D[] weights;
	
	//A function object which performs the sigmoid function.
	private static final DoubleFunction sigmoid = new DoubleFunction() {
		public final double apply(double z) {return 1.0 / (1.0 + Functions.exp.apply(-z));}
	};
	
	//A function object which performs the sigmoid prime function
	private static final DoubleFunction sigmoidPrime = new DoubleFunction() {
		public final double apply(double z) {return sigmoid.apply(z) * (1 - sigmoid.apply(z));}
	};
	
	public Network(int[] sizes)
	{
		//Random number generator
		//Random rng = new Random();
		Normal rng = new Normal(0.0, 1.0, RandomEngine.makeDefault());
		//Number of layers of neurons
		numLayers = sizes.length;
		//The biases and weights of each neuron in each layer
		biases = new DoubleMatrix1D[numLayers];
		weights = new DoubleMatrix2D[numLayers];
		
		//Initializes the biases of each neuron in each layer
		for(int layer = 1; layer < numLayers; layer++) {
			biases[layer] = DoubleFactory1D.dense.make(sizes[layer]);
			biases[layer].assign(rng);
		}
		
		for(int layer = 1; layer < numLayers; layer++) {
			weights[layer] = DoubleFactory2D.dense.make(sizes[layer], sizes[layer - 1]);
			weights[layer].assign(rng);
		}
	}
	
	public DoubleMatrix1D[] getBiases() {
		return biases;
	}
	
	public DoubleMatrix1D getBiases(int layer) {
		return biases[layer];
	}
	
	public DoubleMatrix2D[] getWeights() {
		return weights;
	}
	
	/*
	 * Takes in the input to the network as a vector (1-dimensional matrix).
	 * Returns the output of the network as a vector.
	 */
	public DoubleMatrix1D feedForward(DoubleMatrix1D activations) {
		for(int layer = 1; layer < numLayers; layer++) {
			DoubleMatrix1D weightedActivations = weights[layer].zMult(activations, null);
			activations = weightedActivations.assign(biases[layer], Functions.plus).assign(sigmoid);
		}
		
		return activations;
	}
		
	public DoubleMatrix2D getWeights(int layer) {
		return weights[layer];
	}
	
	public void testMatrices() {
		for(int i = 0; i < numLayers; i++) {
			System.out.println("Biases of layer: " + i);
			System.out.println(biases[i]);
			System.out.println();
			System.out.println("Weights of layer: " + i);
			System.out.println(weights[i]);
			System.out.println();
		}
		
		
		System.out.println(getWeights(1).zMult(getBiases(1), null));
		
	}

	public static void main(String[] args) {
		/*int[] sizes = {10, 10, 10};
		Network n = new Network(sizes);
		
		n.testMatrices(); */
		
		DataLoader load = new DataLoader();
		load.test();
	}

}
