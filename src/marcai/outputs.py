import pandas as pd

def predictions_to_csv(model, test_dataset, test_file_path, id_col_name, prediction_col_name, output_filename):
  
  predictions = model.predict_classes(test_dataset, verbose=0)
  test_ids = pd.read_csv(test_file_path)
  submissions=pd.DataFrame({id_col_name: test_ids[id_col_name], 
                            prediction_col_name: list(predictions.flatten())})
  submissions.to_csv(output_filename, index=False, header=True)

if __name__ == '__main__':
  predictions_to_csv(None, None, 'G:/code/kaggle-titanic/data/test.csv', 'PassengerId', None, None)
  
  