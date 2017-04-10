package neuralNets;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import neuralNets.DataLoader;
import testMnist.TestMnistData;

import cern.colt.function.*;
import cern.colt.matrix.*;
import cern.jet.math.Functions;
import cern.jet.random.Normal;
import cern.jet.random.engine.RandomEngine;

public class Network {
	
	//public static final String DATA_PATH = "C:\\\\Users\\chach\\workspace\\comp4932-neuronet-A3\\Neuralnet_TestData\\";
	public static final String DATA_PATH = "Neuralnet_TestData/";
	public static final String TEST_IMAGE_FILE = "t10k-images.idx3-ubyte";
	public static final String TEST_LABEL_FILE = "t10k-labels.idx1-ubyte";
	public static final String TRAINING_IMAGE_FILE = "train-images.idx3-ubyte";
	public static final String TRAINING_LABEL_FILE = "train-labels.idx1-ubyte";
	
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
	
	public DoubleMatrix2D getWeights(int layer) {
		return weights[layer];
	}
	
	/*
	 * Takes in the input to the network as a vector (1-dimensional matrix).
	 * Returns the output of the network as a vector.
	 */
	private DoubleMatrix1D feedForward(DoubleMatrix1D activations) {
		for(int layer = 1; layer < numLayers; layer++) {
			DoubleMatrix1D weightedActivations = weights[layer].zMult(activations, null);
			activations = weightedActivations.assign(biases[layer], Functions.plus).assign(sigmoid);
		}
		
		return activations;
	}
	
	public void SGD(DoubleMatrix1D[][] training_data, int epochs, int mini_batch_size, double eta, DoubleMatrix1D[][] test_data) {
		int n_test, n;
		
		n = training_data[0].length;
		
		for(int epoch = 1; epoch <= epochs; epoch++) {
			List<DoubleMatrix1D> solution = new ArrayList<DoubleMatrix1D>(), solution2 = new ArrayList<DoubleMatrix1D>();
			for (int i = 0; i < n; i++) {
			    solution.add(i, training_data[0][i]);
			    solution2.add(i, training_data[1][i]);
			}
			Collections.shuffle(solution);
			Collections.shuffle(solution2);
			
			for (int i = 0; i < n; i++) {
			    training_data[0][i] = solution.get(i);
			    training_data[1][i] = solution2.get(i);
			}
			
			DoubleMatrix1D[][][] mini_batches = new DoubleMatrix1D[training_data[0].length / mini_batch_size][2][mini_batch_size];
			
			for(int k = 0, batchNum = 0; k < n; k += mini_batch_size, batchNum++) {
				for(int i = 0; i < mini_batch_size; i++) {
						mini_batches[batchNum][0][i] = training_data[0][k + i];
						mini_batches[batchNum][1][i] = training_data[1][k + i];
						assert mini_batches[batchNum][1][i] != null : "minibatches[" + batchNum + "][1][" + i + "] is null i = " + i + " batchNum = " + batchNum + " k = " + k;
				}
			}
			
			for(DoubleMatrix1D[][] mini_batch : mini_batches) {
				update_mini_batch(mini_batch, eta);
			}
			
			if(test_data != null) {
				n_test = test_data[0].length;
				
				System.out.println("Epoch " + epoch + ": " + evaluate(test_data) + " / " + n_test);
			} else {
				System.out.println("Epoch " + epoch + " complete");
			}
		}
	}
	
	private int evaluate(DoubleMatrix1D[][] test_data) {
		int numCorrect = 0;
		for(int i = 0; i < test_data[0].length; i++) {
			DoubleMatrix1D results = feedForward(test_data[0][i]);
			if(getMax(results) == getMax(test_data[1][i])) {
				numCorrect++;
			}
		}
		
		return numCorrect;
	}

	private int getMax(DoubleMatrix1D vector) {
		double max = Double.MIN_VALUE;
		int maxIndex = -1;
		
		for(int i = 0; i < vector.size(); i++) {
			if(max < vector.getQuick(i)) {
				maxIndex = i;
			}
		}
		
		return maxIndex;
	}

	private void update_mini_batch(DoubleMatrix1D[][] mini_batch, double eta) {
		DoubleMatrix1D[] nabla_b= new DoubleMatrix1D[numLayers];
		DoubleMatrix2D[] nabla_w= new DoubleMatrix2D[numLayers];
		
		for(int layer = 1; layer < numLayers; layer++) {
			nabla_b[layer] = DoubleFactory1D.dense.make(biases[layer].size(), 0);
			nabla_w[layer] = DoubleFactory2D.dense.make(weights[layer].rows(), weights[layer].columns());
		}
		
		DoubleMatrix2D[][] delta_nablas;
		
		for(int batchNum = 0; batchNum < mini_batch[0].length; batchNum++) {
			assert mini_batch[1][batchNum] != null : "NULL mini_batch : mini_batch[1]" + batchNum; 
			delta_nablas = backprop(mini_batch[0][batchNum], mini_batch[1][batchNum]);
			for(int layer = 1; layer < numLayers; layer++) {
				DoubleMatrix1D delta_nabla_b = DoubleFactory2D.dense.diagonal(delta_nablas[0][layer]);
				nabla_b[layer].assign(delta_nabla_b, Functions.plus);
				nabla_w[layer].assign(delta_nablas[1][layer], Functions.plus);
			}
		}
		
		for(int layer = 1; layer < numLayers; layer++) {
			nabla_w[layer].assign(Functions.mult(eta / (double)mini_batch[0].length));
			nabla_b[layer].assign(Functions.mult(eta / (double)mini_batch[0].length));
			
			weights[layer] = weights[layer].assign(nabla_w[layer],Functions.minus);
			biases[layer] = biases[layer].assign(nabla_b[layer], Functions.minus);
		}
	}
		
	private DoubleMatrix2D[][] backprop(DoubleMatrix1D x, DoubleMatrix1D results) {
		DoubleMatrix1D[] nabla_b= new DoubleMatrix1D[numLayers];
		DoubleMatrix2D[] nabla_w= new DoubleMatrix2D[numLayers];
		
		for(int layer = 1; layer < numLayers; layer++) {
			nabla_b[layer] = DoubleFactory1D.dense.make(biases[layer].size(), 0);
			nabla_w[layer] = DoubleFactory2D.dense.make(weights[layer].rows(), weights[layer].columns());
		}
		
		//Feedforward storing activations and zs
		DoubleMatrix1D activation = x;
		DoubleMatrix1D[] activations = new DoubleMatrix1D[numLayers];
		activations[0] = activation;
		
		DoubleMatrix1D[] zs = new DoubleMatrix1D[numLayers];
		
		for(int layer = 1; layer < numLayers; layer++) {
			DoubleMatrix1D z = weights[layer].zMult(activation, null);
			z.assign(biases[layer], Functions.plus);
			zs[layer] = z;
			
			activation = z.assign(sigmoid);
			activations[layer] = activation;
		}
		
		//backward pass
		DoubleMatrix1D delta = cost_derivative(activations[numLayers - 1], results).assign(zs[2].assign(sigmoidPrime), Functions.mult);
		
		nabla_b[numLayers - 1] = delta.copy();
		nabla_w[numLayers - 1] = matMult(delta, activations[numLayers - 2]);
		
		for(int layer = numLayers - 2; layer > 0; layer--) {
			DoubleMatrix1D z = zs[layer];
			DoubleMatrix1D sp = z.copy();
			sp.assign(sigmoidPrime);
			delta = weights[layer + 1].zMult(delta, null, 1, 1, true);
			nabla_b[layer] = delta;
			nabla_w[layer] = matMult(delta, activations[layer - 1]);
		}
		
		DoubleMatrix2D[] temp = new DoubleMatrix2D[numLayers];
		
		
		for(int layer = 1; layer < numLayers; layer++) {
			temp[layer] = DoubleFactory2D.dense.diagonal(nabla_b[layer]);
		}
		
		return new DoubleMatrix2D[][] {temp, nabla_w};
	}
	
	private DoubleMatrix2D matMult(DoubleMatrix1D v1, DoubleMatrix1D v2) {
		DoubleMatrix2D result = DoubleFactory2D.dense.make(v1.size(), v2.size());
		
		for(int i = 0; i < v1.size(); i++) {
			for(int j = 0; j < v2.size(); j++) {
				result.setQuick(i, j, v1.getQuick(i) * v2.getQuick(j));
			}
		}
		
		return result;
	}

	private DoubleMatrix1D cost_derivative(DoubleMatrix1D output, DoubleMatrix1D results) {
		DoubleMatrix1D difference = output.copy();
		return difference.assign(results, Functions.minus);
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
		
		//System.out.println("Weights[1] * biases[1]");
		//System.out.println(getWeights(1).zMult(getBiases(1), null));
		
		System.out.println(feedForward(DoubleFactory1D.dense.make(3, 0.5)));
	}

	public static void main(String[] args) {
		try {
			System.out.println(new File(".").getCanonicalPath());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		int[] sizes = {784, 30, 10};
		Network n = new Network(sizes);
		
		//n.testMatrices(); 
		
		DataLoader load = new DataLoader();
		DoubleMatrix1D[][] testData = load.load(TEST_LABEL_FILE, TEST_IMAGE_FILE);
		DoubleMatrix1D[][] trainingData = load.load(DATA_PATH + TRAINING_LABEL_FILE, DATA_PATH + TRAINING_IMAGE_FILE);

		n.SGD(trainingData, 5, 10, 3.0, testData);
	}

}
