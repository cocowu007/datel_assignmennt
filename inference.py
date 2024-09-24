# Important notes:
# In order to run the notebook and script successfully in a conda environment,
# you need to install necessary dependencies:
# torch numpy matplotlib import_ipynb

# Command to run this script:
# python inference.py --model path_of_model --dataset path_of_data
# For example, if you put the model and validation data in the same file as notebook and script, 
# and name the model like the one below, then: 
# python inference.py --model unet_model.pth --dataset dataset.pickle

import import_ipynb
import argparse
from dateltest import run_inference  # Import the run_inference function defined in the notebook

def main():
    parser = argparse.ArgumentParser(description="Run inference on the validation set")
    parser.add_argument('--model', type=str, help="Path to the model file", required=True)
    parser.add_argument('--dataset', type=str, help="Path to the dataset file", required=True)
    
    args = parser.parse_args()
    run_inference(args.model, args.dataset)  # Calls the model path and dataset paths passed from the command

if __name__ == "__main__":
    main()
