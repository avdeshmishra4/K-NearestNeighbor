import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class KNN {

	float bed_max = 0f;
	float bath_max = 0f;
	float area_max = 0f;

	public static void main(String[] args) {

		KNN knn = new KNN();

		knn.generateModelForGivenKNN(1);
		knn.generateTestForKNNModel(1);
		
		knn.generateModelForGivenKNN(3);
		knn.generateTestForKNNModel(3);
		
		knn.generateModelForGivenKNN(5);
		knn.generateTestForKNNModel(5);
		
		knn.generateModelForGivenKNN(7);
		knn.generateTestForKNNModel(7);
		
		knn.generateModelForGivenKNN(9);
		knn.generateTestForKNNModel(9);

	}

	public void generateModelForGivenKNN(int k) {
		try {
			List<List<Float>> featureList = loadTrainingData();
			List<Float> prediction = new ArrayList<>();
			Map<String, Float> distances = new HashMap<>();
			Map<String, Float> distanceSorted = new HashMap<>();
			boolean ASC = true;

			File currentDir = new File(new File(".").getAbsolutePath());
			String curr_dir_path = currentDir.getCanonicalPath();
			String output = curr_dir_path + "/train_model_" + k +"-NN.csv";
			File op_file = new File(output);
			BufferedWriter model_writer = BufferReaderAndWriter.getWriter(op_file);
			model_writer.write("actual_price, predicted_price");
			model_writer.newLine();
			model_writer.flush();
			

			for (int i = 0; i < featureList.get(0).size(); i++) {
				
								
				for (int j = 0; j < featureList.get(0).size(); j++) {

					float pairDistance = (float) java.lang.Math
							.sqrt(java.lang.Math.pow(
									(featureList.get(0).get(i) - featureList
											.get(0).get(j)), 2)
									+ java.lang.Math
											.pow((featureList.get(1).get(i) - featureList
													.get(1).get(j)), 2)
									+ java.lang.Math
											.pow((featureList.get(2).get(i) - featureList
													.get(2).get(j)), 2));

					String id = i + "-" + j;
					distances.put(id, pairDistance);
					// distances.add(pairDistance);

				}

				distanceSorted = sortByComparator(distances, ASC); // sorts the
																	// list in
																	// ascending
																	// order
				int value_of_k = k;
				float predicted_value = 0f;
				String[] requiredKeys = new String[k];
				
				for(Map.Entry<String, Float> entry : distanceSorted.entrySet()){
					
					if(value_of_k > 0){
						
						requiredKeys[value_of_k-1] = entry.getKey();
								
						value_of_k--;
						
						
					}else{
						
						break;
					}
					
				}
				
				value_of_k = k;
				
				while (value_of_k > 0 ) {
					value_of_k--;
					
					String key = requiredKeys[value_of_k];
					String[] token = key.split("-");
					int min_distance_id = Integer.parseInt(token[1]);
					predicted_value += featureList.get(3).get(min_distance_id);	// here min_distance_id-1 because

				}

				predicted_value = predicted_value / k;
				prediction.add(predicted_value);
				distances.clear();
				distanceSorted.clear();

			}
			
			float MAE = 0f;
			float MSE = 0f;
			float RMSE = 0f;
			for(int m = 0; m < prediction.size(); m++){
				
				model_writer.write(Float.toString(featureList.get(3).get(m)));
				model_writer.write(",");
				model_writer.write(Float.toString(prediction.get(m)));
				model_writer.write(",");
				model_writer.write(Float.toString(java.lang.Math.abs((featureList.get(3).get(m))-(prediction.get(m)))));
				model_writer.write(",");
				model_writer.write(Double.toString(java.lang.Math.pow(java.lang.Math.abs((featureList.get(3).get(m))-(prediction.get(m))), 2)));
				model_writer.write(",");
				MAE += java.lang.Math.abs((featureList.get(3).get(m))-(prediction.get(m))); 
				MSE += java.lang.Math.pow(java.lang.Math.abs((featureList.get(3).get(m))-(prediction.get(m))), 2);
				model_writer.newLine();
				model_writer.flush();
				
				
			}
			
			
			
			model_writer.write("MAE,"+Float.toString(MAE/prediction.size()));
			model_writer.write(",");
			model_writer.write("MSE,"+Float.toString(MSE/prediction.size()));
			model_writer.write(",");
			RMSE = (float) java.lang.Math.sqrt(MSE/prediction.size());
			model_writer.write("RMSE,"+Float.toString(RMSE));
						
			model_writer.close();
			prediction.clear();
			
			

		} catch (Exception ex) {

			ex.printStackTrace();

		}

	}
	
	public void generateTestForKNNModel(int k){
		try{
		List<List<Float>> featureListTest = loadTestData();
		List<List<Float>> featureListTrain = loadTrainingData();
		List<Float> prediction = new ArrayList<>();
		Map<String, Float> distances = new HashMap<>();
		Map<String, Float> distanceSorted = new HashMap<>();
		boolean ASC = true;

		File currentDir = new File(new File(".").getAbsolutePath());
		String curr_dir_path = currentDir.getCanonicalPath();
		String output = curr_dir_path + "/test_model_" + k +"-NN.csv";
		File op_file = new File(output);
		BufferedWriter model_writer = BufferReaderAndWriter.getWriter(op_file);
		model_writer.write("actual_price, predicted_price");
		model_writer.newLine();
		model_writer.flush();
		

		for (int i = 0; i < featureListTest.get(0).size(); i++) {
			
						
			for (int j = 0; j < featureListTrain.get(0).size(); j++) {

				float pairDistance = (float) java.lang.Math
						.sqrt(java.lang.Math.pow(
								(featureListTest.get(0).get(i) - featureListTrain
										.get(0).get(j)), 2)
								+ java.lang.Math
										.pow((featureListTest.get(1).get(i) - featureListTrain
												.get(1).get(j)), 2)
								+ java.lang.Math
										.pow((featureListTest.get(2).get(i) - featureListTrain
												.get(2).get(j)), 2));

				String id = i + "-" + j;
				distances.put(id, pairDistance);
				// distances.add(pairDistance);

			}

			distanceSorted = sortByComparator(distances, ASC); // sorts the
																// list in
																// ascending
																// order
			int value_of_k = k;
			float predicted_value = 0f;
			String[] requiredKeys = new String[k];
			
			for(Map.Entry<String, Float> entry : distanceSorted.entrySet()){
				
				if(value_of_k > 0){
					
					requiredKeys[value_of_k-1] = entry.getKey();
							
					value_of_k--;
					
					
				}else{
					
					break;
				}
				
			}
			
			value_of_k = k;
			
			while (value_of_k > 0 ) {
				value_of_k--;
				
				String key = requiredKeys[value_of_k];
				String[] token = key.split("-");
				int min_distance_id = Integer.parseInt(token[1]);
				predicted_value += featureListTrain.get(3).get(min_distance_id);	// here min_distance_id-1 because

			}

			predicted_value = predicted_value / k;
			prediction.add(predicted_value);
			distances.clear();
			distanceSorted.clear();

		}
		
		float MAE = 0f;
		float MSE = 0f;
		float RMSE = 0f;
		for(int m = 0; m < prediction.size(); m++){
			
			model_writer.write(Float.toString(featureListTest.get(3).get(m)));
			model_writer.write(",");
			model_writer.write(Float.toString(prediction.get(m)));
			model_writer.write(",");
			model_writer.write(Float.toString(java.lang.Math.abs((featureListTest.get(3).get(m))-(prediction.get(m)))));
			model_writer.write(",");
			model_writer.write(Double.toString(java.lang.Math.pow(java.lang.Math.abs((featureListTest.get(3).get(m))-(prediction.get(m))), 2)));
			model_writer.write(",");
			
			MAE += java.lang.Math.abs((featureListTest.get(3).get(m))-(prediction.get(m))); 
			MSE += java.lang.Math.pow(java.lang.Math.abs((featureListTest.get(3).get(m))-(prediction.get(m))), 2);
			
			model_writer.newLine();
			model_writer.flush();
			
			
		}
		
		
		
		model_writer.write("MAE,"+Float.toString(MAE/prediction.size()));
		model_writer.write(",");
		model_writer.write("MSE,"+Float.toString(MSE/prediction.size()));
		model_writer.write(",");
		RMSE = (float) java.lang.Math.sqrt(MSE/prediction.size());
		model_writer.write("RMSE,"+Float.toString(RMSE));
					
		model_writer.close();
		prediction.clear();
		
		

	} catch (Exception ex) {

		ex.printStackTrace();

	}

		
		
}
	
	public List<List<Float>> loadTestData(){
		
		List<List<Float>> allfeatures_with_required_scaling = new ArrayList<List<Float>>();
		try {
			File currentDir = new File(new File(".").getAbsolutePath());
			String curr_dir_path = currentDir.getCanonicalPath();
			String input = curr_dir_path + "/Test.txt";
			File ip_file = new File(input);

			List<Float> bed = new ArrayList<Float>();
			List<Float> bath = new ArrayList<Float>();
			List<Float> areaSquare = new ArrayList<Float>();
			List<Float> price = new ArrayList<Float>();

			BufferedReader ip_br = BufferReaderAndWriter.getReader(ip_file);

			String test_line;
			while ((test_line = ip_br.readLine()) != null) {

				String[] lineArray = test_line.split("\\t+");

				bed.add(Float.parseFloat(lineArray[0]));
				bath.add(Float.parseFloat(lineArray[1]));
				areaSquare.add(Float.parseFloat(lineArray[2]));
				price.add(Float.parseFloat(lineArray[3]));

			}
			
			for (int j = 0; j < bed.size(); j++) {

				bed.set(j, bed.get(j) / bed_max);
				bath.set(j, bath.get(j) / bath_max);
				areaSquare.set(j, areaSquare.get(j) / area_max);

			}

			allfeatures_with_required_scaling.add(bed);
			allfeatures_with_required_scaling.add(bath);
			allfeatures_with_required_scaling.add(areaSquare);
			allfeatures_with_required_scaling.add(price);
			
			
		}catch(Exception ex){
			ex.printStackTrace();
			
			
		}
		
		return allfeatures_with_required_scaling;
		
		
	}
	
	

	public List<List<Float>> loadTrainingData() {

		List<List<Float>> allfeatures_with_required_scaling = new ArrayList<List<Float>>();
		try {
			File currentDir = new File(new File(".").getAbsolutePath());
			String curr_dir_path = currentDir.getCanonicalPath();
			String input = curr_dir_path + "/Training.txt";
			File ip_file = new File(input);

			List<Float> bed = new ArrayList<Float>();
			List<Float> bath = new ArrayList<Float>();
			List<Float> areaSquare = new ArrayList<Float>();
			List<Float> price = new ArrayList<Float>();

			BufferedReader ip_br = BufferReaderAndWriter.getReader(ip_file);

			String train_line;
			while ((train_line = ip_br.readLine()) != null) {

				String[] lineArray = train_line.split("\\t+");

				bed.add(Float.parseFloat(lineArray[0]));
				bath.add(Float.parseFloat(lineArray[1]));
				areaSquare.add(Float.parseFloat(lineArray[2]));
				price.add(Float.parseFloat(lineArray[3]));

			}

			float max_bed = bed.get(0);
			float max_bath = bath.get(0);
			float max_areaSquare = areaSquare.get(0);

			for (int i = 0; i < bed.size(); i++) {

				if (bed.get(i) > max_bed) {

					max_bed = bed.get(i);

				}

				if (bath.get(i) > max_bath) {

					max_bath = bath.get(i);

				}

				if (areaSquare.get(i) > max_areaSquare) {

					max_areaSquare = areaSquare.get(i);

				}

			}

			for (int j = 0; j < bed.size(); j++) {

				bed.set(j, bed.get(j) / max_bed);
				bath.set(j, bath.get(j) / max_bath);
				areaSquare.set(j, areaSquare.get(j) / max_areaSquare);

			}

			allfeatures_with_required_scaling.add(bed);
			allfeatures_with_required_scaling.add(bath);
			allfeatures_with_required_scaling.add(areaSquare);
			allfeatures_with_required_scaling.add(price);

			bed_max = max_bed;
			bath_max = max_bath;
			area_max = max_areaSquare;

		} catch (Exception e) {

			e.printStackTrace();

		}

		return allfeatures_with_required_scaling;

	}

	private Map<String, Float> sortByComparator(Map<String, Float> unsortMap,
			final boolean order) {

		List<Entry<String, Float>> list = new LinkedList<Entry<String, Float>>(
				unsortMap.entrySet());

		// Sorting the list based on values
		Collections.sort(list, new Comparator<Entry<String, Float>>() {
			public int compare(Entry<String, Float> o1, Entry<String, Float> o2) {
				if (order) {
					return o1.getValue().compareTo(o2.getValue());
				} else {
					return o2.getValue().compareTo(o1.getValue());

				}
			}
		});

		// Maintaining insertion order with the help of LinkedList
		Map<String, Float> sortedMap = new LinkedHashMap<String, Float>();
		for (Entry<String, Float> entry : list) {
			sortedMap.put(entry.getKey(), entry.getValue());
		}

		return sortedMap;
	}

}
