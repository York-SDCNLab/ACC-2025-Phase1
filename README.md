# ACC-2025-Phase1
Phase 1 Submission of the ACC self-driving car competition

You can view the performance of our solution [here](https://www.youtube.com/watch?v=-Kq4Oqgqnl8&ab_channel=SDCNLABYorkU)

# Installation
pip install -r requirements.txt

To run our perception pipeline, you will need to download the model weights [here](https://yuoffice-my.sharepoint.com/:u:/g/personal/hunterls_yorku_ca/EQpNHLK2sRlMh60c1uHSd0IB7_I-RnMmKLUDGat5fBDVKA?email=studentcompetition%40Quanser.com&e=8xn8dn) \
Only studentcompetition@Quanser.com has been granted access to these model weights, please let us know if you cannot download them.

# Running the Solution
Once all dependencies are installed, run ```Setup_Real_Scenario.py``` in one terminal and ```python scripts/angle_diff_demo.py``` 

The default configuration in ```scripts/angle_diff_demo.py``` can be modified here. Specify a list of tuples that indicate pickup/dropoff regions that the taxi service needs to cover: https://github.com/York-SDCNLab/ACC-2025-Phase1/blob/db52d265567f654dfad2a77326aa426f827e97be/scripts/angle_diff_demo.py#L149-L153 

# Additional Information
To indicate normal driving modes, we set the LEDs to green. When the car is stopping at a red light or stop sign, we set the LEDs to red. When the car is in a pickup or dropoff region, we set the LEDs to blue.

Our planner stops the car for 1 second at stop signs, and 3 seconds when picking up or dropping off customers.
