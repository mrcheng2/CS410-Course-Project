from process_data import DataProcessor
# In your_script.py or in an interactive session
data_processor = DataProcessor()
processed_data = data_processor.process_data(dump_processed=True)

print(processed_data.head())