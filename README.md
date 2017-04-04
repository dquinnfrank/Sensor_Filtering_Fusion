# Sensor Filtering and Fusion

## Project 1: Kalman Filter

This program estimates the position of a robot using GPS, IMU, and future state prediction. It takes a file with the relevant data and will produce a graph showing the raw data and the Kalman filter output. The covariances of the data can be manipulated to observe the impact on the Kalman filter output. 

## Project 2: Consensus Filter

This program simulates a network of sensors to estimate a signal by sharing readings without a central server. Each sensor node in the network gets a somewhat low quality reading of the target signal that it shares with its neighbors. By iteratively fusing neighbor readings, the network will arrive at an estimate that is much more accurate than the reading of any one node alone. Multiple fusion schemes and dynamic target signals can be simulated. The program will output an image of the network and the tracked signal.
