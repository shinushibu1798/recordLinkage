import sys
import json
import os
from src.data.DatasetGenerator import generateDataset
from src.FeatureExtraction.graphCreation import *

def main(targets):

	if "gen-data" in targets:
		configPath = './config/datasetGenConfig.json'
		with open(configPath) as f:
			config = json.load(f)
		f.close()
		generateDataset(config)

	if "create-small-graphs" in targets:
		configPath = './config/train_config.json'
		with open(configPath) as f:
			config = json.load(f)
		f.close()

		build_abt_buy_graph(config)
		print("Finished Abt Buy Dataset")
		build_dblp_acm_dataset(config)
		print("Finished dblp_acm Dataset")
		build_dblp_scholar_dataset(config)
		print("Finished dblp_scholar Dataset")
		build_amazon_google_dataset(config)
		print("Finished amazon_google Dataset")

	if "test" in targets:
		configPath = './config/test_config.json'
		with open(configPath) as f:
			config = json.load(f)
		f.close()

		build_abt_buy_graph(config)

	pass

if __name__ == '__main__':
	targets = sys.argv[1:]
	main(targets)