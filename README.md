# AneurysmSegmentation

AneurysmSegmentation is a deep learning project for the segmentation of Aneurysms using python.

## Installation

1. Clone the repository:
git clone https://github.com/larshaupt/AneurysmSegmentation.git

Copy code

2. Install the required libraries from environment.yml:
conda env create -f environment.yml

Copy code

## Usage

1. Train the segmentation model:
python train_segmentation.py --data-path [path_to_data] --output-path [path_to_output]

Copy code
Note: The data should be in .h5 format

2. Network Inference:

from network_inference import evaluate
evaluate(model_path, data_path)
This class can be used to evaluate the saved models and get prediction masks.

Copy code
Please let me know if there is anything else you would like me to add or change.
