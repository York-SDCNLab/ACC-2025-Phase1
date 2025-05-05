# ACC-2025-Phase1
Phase 1 Submission of the ACC self-driving car competition

You can view the performance of our solution [here](TODO)

# Installation
pip install -r requirements.txt

To run our perception pipeline, you will need to download the model weights [here](TODO)

# Running the Solution
Once all dependencies are installed, run ```Setup_Real_Scenario.py``` in one terminal and ```python scripts/angle_diff_demo.py``` 

The default configuration in ```scripts/angle_diff_demo.py``` can be modified here. Specify a list of tuples that indicate pickup/dropoff regions that the taxi service needs to cover: https://github.com/York-SDCNLab/ACC-2025-Phase1/blob/db52d265567f654dfad2a77326aa426f827e97be/scripts/angle_diff_demo.py#L149-L153 
