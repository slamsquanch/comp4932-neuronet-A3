package neuralNets;
import java.util.List;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleFactory1D;

import neuralNets.MnistReader;

public class DataLoader {
	
	
	public DoubleMatrix1D[][] load(String label_file, String image_file) {
		//String LABEL_FILE = "C:/Users/Zac/SCHOOL/TERM 4/Term_4_Java_Work/NeuroNets/Neuralnet_TestData/t10k-labels.idx1-ubyte";
		//String IMAGE_FILE = "C:/Users/Zac/SCHOOL/TERM 4/Term_4_Java_Work/NeuroNets/Neuralnet_TestData/t10k-images.idx3-ubyte";
		
		//String LABEL_FILE = "C:/Users/Zac/SCHOOL/TERM 4/Term_4_Java_Work/NeuroNets/Neuralnet_TestData/t10k-labels.idx1-ubyte";
		//String IMAGE_FILE = "C:/Users/Zac/SCHOOL/TERM 4/Term_4_Java_Work/NeuroNets/Neuralnet_TestData/t10k-images.idx3-ubyte";
		
		int[] labels = MnistReader.getLabels(label_file);
		List<int[][]> images = MnistReader.getImages(image_file);
		
		
		if(labels.length != images.size()) {
			System.out.println("Error:  Labels and Images sizes are NOT equal.");
			
		}
		if(28 != images.get(0).length) {
			System.out.println("Error:  Labels and Images sizes are NOT equal.");
			
		}
		if(28 != images.get(0)[0].length) {
			System.out.println("Error:  Labels and Images sizes are NOT equal.");
			
		}
		
		DoubleMatrix1D[] trainingInputs = new DoubleMatrix1D[images.size()];
		DoubleMatrix1D[] trainingResults = new DoubleMatrix1D[labels.length];
				
		for(int i = 0; i < images.size(); i++) {
			int[][] temp = images.get(i);  //dereference our List of 2D int arrays.
			double[]tempDouble = new double[784];  //Need a double array for decimals.
			
			for(int y = 0; y < 28; y++) {
				//System.out.print("[ ");
				for(int x = 0; x < 28; x++) {
					//Cast 2D int array to 2D double array. Divide by 256.
					assert y < temp[0].length : "y out of bounds : " + y + "temp[0].length : " + temp[0].length;
					assert x < temp.length : "x out of bounds" ;
					tempDouble[y * 28 + x] = ((double)temp[y][x]) / 256.0;
				}
			}
			trainingInputs[i] = DoubleFactory1D.dense.make(tempDouble);
			trainingResults[i] = DoubleFactory1D.dense.make(10);
			trainingResults[i].set(labels[i], 1);
		}
		return new DoubleMatrix1D[][] {trainingInputs, trainingResults };
	}
	
	
	
	public static void printf(String format, Object... args) {
		System.out.printf(format, args);
	}
	
	

}
