package neuralNets;
import java.util.List;

import neuralNets.MnistReader;

public class DataLoader {
	
	
	public void test() {
		String LABEL_FILE = "C:/Users/Zac/SCHOOL/TERM 4/Term_4_Java_Work/NeuroNets/Neuralnet_TestData/t10k-labels.idx1-ubyte";
		String IMAGE_FILE = "C:/Users/Zac/SCHOOL/TERM 4/Term_4_Java_Work/NeuroNets/Neuralnet_TestData/t10k-images.idx3-ubyte";
		
		int[] labels = MnistReader.getLabels(LABEL_FILE);
		List<int[][]> images = MnistReader.getImages(IMAGE_FILE);
		
		
		if(labels.length != images.size()) {
			System.out.println("Error:  Labels and Images sizes are NOT equal.");
			
		}
		if(28 != images.get(0).length) {
			System.out.println("Error:  Labels and Images sizes are NOT equal.");
			
		}
		if(28 != images.get(0)[0].length) {
			System.out.println("Error:  Labels and Images sizes are NOT equal.");
			
		}
		
		double doubleMatrix1D[] = new double[784];
		for(int i = 0; i < 784; i++) {
			int[][] temp = images.get(i);  //dereference our List of 2D int arrays.
			double[][]tempDouble = new double[temp.length][temp[0].length];  //Need a double array for decimals.
			
			for(int x = 0; x < 28; x++) {
				System.out.print("[ ");
				for(int y = 0; y < 28; y++) {
					//Cast 2D int array to 2D double array. Divide by 256.
					tempDouble[x][y] = (double)temp[x][y];  
					doubleMatrix1D[i] = tempDouble[x][y]/256;  //Convert to 1D array of doubles.
					System.out.print(tempDouble[x][y]/256 + ", ");
				}
				System.out.println("]\n\n");
			}
		}
		
	}
	
	
	
	public static void printf(String format, Object... args) {
		System.out.printf(format, args);
	}
	
	

}
