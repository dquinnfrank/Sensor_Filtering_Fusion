# Project 2: Consensus Filter
# David Frank

# For getting commandline arguments
import argparse

# For numerical operations
import numpy as np

# For getting distances
from scipy.spatial import distance

# For plotting results
import matplotlib.pyplot as plt

# Working 2D space right now
R_space = 2

# Target reading is 1D right now
T_space = 1

# A single node in the sensor network
# R_space is the number of dimensions being worked in
class Node:

	# Implementations of fusing sensor readings

	# Max Degree
	def max_degree(self):

		# Go through each neighbor reading and accumulate the reading
		# Each neighbor reading has 1 / total_nodes as its weight
		acc_reading = np.zeros(T_space)
		for neighbor_name in self.neighbor_readings:

			# Get the value of the neighbor reading
			neighbor_value = self.neighbor_readings[neighbor_name]

			# Set the weight for the reading
			weight = 1.0 / self.network.total_nodes()

			# Add the reading based on the weight
			acc_reading += weight * neighbor_value

		# Set the weight of this node so that the weights all sum to 1
		self_weight = 1 - self.get_degree() / float(self.network.total_nodes())

		# Add the weight of this node
		acc_reading += self_weight * self.get_sensor_reading()

		return acc_reading

	# The methods for fusing neighbor readings
	consensus_methods = {
				"Weight Design 1" : 1,
				"Weight Design 2" : 2,
				"Max Degree" : max_degree,
				"Metropolis" : 4
				}
	
	# Class members
	#
	# neighbors : list
	# Holds the names of all neighboring nodes
	#
	# stable_reading : np.array : shape(T_space)
	# The reading from the last update, needed to avoid race conditions
	#
	# unstable_reading : np.array : shape(T_space)
	# The reading after running the consensus for this node, while other nodes are still updating

	# Constructor
	#
	# environment : Environment()
	# Object that the node will use to find information about the environment
	#
	# network : Network()
	# This node is a member of the network
	# Used to find information about the network
	#
	# name : string
	# A unique name for this node
	#
	# reading_noise : float
	# the noise of the reading as defined by cv in the project description
	#
	# range_s : float
	# optimal sensing range
	#
	# consensus_method : string
	# The name of the method to use to fuse readings from neighbors
	def __init__(self, environment, network, name, reading_noise=.01, range_s=1.6, consensus_method = "Max Degree"):

		# Save the environment and network
		self.environment = environment
		self.network = network

		# Save the name of this node
		self.name = name

		# Save the reading parameters
		self.reading_noise = reading_noise
		self.range_s = range_s

		# This holds the neighbors of this node
		self.neighbors = []

		# Holds the readings of every neighbor
		self.neighbor_readings = {}

		# The last reading that is valid, to avoid race conditions among node reports
		# Use the intial estimate of the target without any fusion
		self.stable_reading = np.random.normal(scale=2, size=(T_space,)) + self.environment.get_target_reading_at(0)

		# The value storing the consensus while the other nodes are updating
		self.unstable_reading = None

		# Set which consensus method to use
		self.fuse_readings = self.consensus_methods[consensus_method]

	# Sets the unstable reading for this node, and returns it
	# Output is based on the consensus method choosen for the node
	def reading(self):

		# Update the neighbor readings
		self.acquire_neighbor_readings()

		# Use the set consensus method to get the reading
		self.unstable_reading = self.fuse_readings(self)

		# Return the reading
		return self.unstable_reading

	# Once the network is done updating, the unstable reading becomes the new stable reading
	def stabilize(self):

		self.stable_reading = self.unstable_reading

	# Gets the postion
	#
	# return : numpy array : shape (R_space,)
	def get_position(self):

		return self.environment.get_node_position(self.name)

	# Gets the sensor reading of this node
	#
	# return : numpy array : shape (R_space,)
	def get_sensor_reading(self):

		# Get the deviation for the noise term
		sigma = (np.linalg.norm(self.get_position() - self.network.get_network_average_position()) ** 2 + self.reading_noise) / (self.range_s ** 2)

		# Get the noise
		noise = np.random.normal(scale=sigma, size=(T_space,))

		# Get the measurement from the environment and add the noise
		measurement = self.environment.get_target_reading() + noise

		return measurement

	# Acquires the neighbors of this node
	def acquire_neighbors(self):

		self.neighbors = self.environment.get_node_neighbors(self.name)

	# Gets the degree of this node
	# Does not update before checking
	def get_degree(self):

		return len(self.neighbors)

	# Gets the readings from every neighbor
	def acquire_neighbor_readings(self):

		# Clear current readings
		self.neighbor_readings = {}

		# Update the neighbors
		self.acquire_neighbors()

		# Go through each neighbor and get the reading
		for node in self.neighbors:

			self.neighbor_readings[node] = self.network.get_node_reading(node)

# The environment that the nodes are in
# Nodes interact with this class to find their positions and sensor readings
class Environment:

	# Class members
	#
	# nodes : dictionary : (node_name : position)
	# Holds information about nodes
	# position : numpy array : shape (R_space,)
	#
	# target_function : function(numeric time_step)
	# Gives the true position of the target as a function of time
	#
	# time_step : numeric
	# The current time to evalutate

	# Constructor
	#
	# target_position : function(numeric time_step)
	# Needs to give the position of the target being tracked as a function of time
	#
	# target_reading : function(numeric time_step)
	# Needs to give the reading of the target being tracked as a function of time
	#
	# communication_radius : float
	# Determines how far a node can communicate
	# Defaults to None, means that the node can communicate with every other node in network
	#
	# max_neighbors : int
	# The maximum number of neighbors to accept, nodes closer have priority
	# Defaults to None, means that every node in the communication_radius will be a neighbor
	def __init__(self, target_position, target_reading, communication_radius = None, max_neighbors = None):

		# Create the node dictionary
		self.nodes = {}

		# Save the target position function
		self.target_position = target_position

		# Save the reading position function
		self.target_reading = target_reading

		# Set the node communication parameters
		self.communication_radius = communication_radius
		self.max_neighbors = max_neighbors

		# Set the initial time to 0
		self.time_step = 0

	# Moves the environment ahead by the set amount
	def advance(self, increment = 1):

		self.time_step += increment

	# Adds a node to the nework at the specified postion
	#
	# node_name : string
	# The unique name of the node
	#
	# node_position : numpy array : shape (R_space,)
	# The location of the node in the environment
	def add_node(self, node_name, node_position):

		self.nodes[node_name] = node_position

	# Returns the position of a node
	#
	# node_name : string
	# The name of the node to get the postion of
	#
	# return : numpy array : shape (R_space)
	# The position of the specified node
	def get_node_position(self, node_name):

		return self.nodes[node_name]

	# Returns the true position of the target
	#
	# return : numpy array : shape (R_space,)
	def get_target_position(self):

		return self.get_target_position_at(self.time_step)

	# Returns the true reading of the target
	#
	# return : numpy array : shape (T_space,)
	def get_target_reading(self):

		return self.get_target_reading_at(self.time_step)

	# Evaluates the position function at the specified time step
	#
	# return : numpy array : shape (R_space,)
	def get_target_position_at(self, time_step):

		return self.target_position(time_step)

	# Evaluates the reading function at the specified time step
	#
	# return : numpy array : shape (T_space,)
	def get_target_reading_at(self, time_step):

		return self.target_reading(time_step)

	# Returns the neighbors of the node
	#
	# Returns : list[strings]
	# The nodes that are the neighbors of the specified node
	def get_node_neighbors(self, node_name):

		# Get the location of the interest node
		node_position = self.nodes[node_name]

		# Get the distances to every other node in the network
		names = []
		distances = []
		for name in self.nodes:

			# Get the position
			position = self.nodes[name]

			# Ignore if the node is the interest node
			if node_name != name:

				# Get the distance between the nodes
				dist = distance.euclidean(node_position, position)

				# If the communication_radius is None, add the node
				if self.communication_radius is None:

					# Add the node to the name list
					names.append(name)

					# Add the distance
					distances.append(dist)

				# If communication_radius is sent and distance is within range, add the node to the lists
				elif dist <= self.communication_radius:

					# Add the node to the name list
					names.append(name)

					# Add the distance
					distances.append(dist)

		# Sort the name list by the distances
		closest = [n for (d,n) in sorted(zip(distances, names))]

		# If max_neighbors is sent, remove all items after the max
		if self.max_neighbors is not None:

			closest = closest[:self.max_neighbors]

		return closest

# A network of sensor nodes
class Network:

	# Class members
	# nodes : dictionary : (node_name, node_object)
	# The node objects
	#
	# edges : list : "first_name-second_name" sorted lexigraphically
	# Represents an undirected graph of neighbors

	# Constructor
	#
	def __init__(self):

		# Create the node dictionary
		self.nodes = {}

		# Holds the edges in the network
		self.edges = []

	# Runs the network to get the reading of the nodes
	# TODO: Return mean and variance of the network
	def get_all_readings(self):

		# Update the readings of all nodes
		for node_name in self.nodes:

			# Updates the value in unstable_reading
			self.nodes[node_name].reading()

		# Once all nodes have updated, they can be stabilized
		for node_name in self.nodes:

			self.nodes[node_name].stabilize()

	# Adds a node to the nework
	#
	# node_name : string
	# The unique name of the node
	def add_node(self, node_name, node_object):

		# Add the node
		self.nodes[node_name] = node_object

	# Generator that returns the names of all nodes in the network
	#
	# yield : string
	def node_names(self):

		for node_name in self.nodes.keys():

			yield node_name

	# Gets average position of all nodes in the network
	#
	# return : numpy array : shape (R_space,)
	def get_network_average_position(self):

		# The total number of nodes in the network
		num_nodes = self.total_nodes()

		# Get the location of all nodes
		all_nodes = np.empty((num_nodes, R_space))
		for index, item in enumerate(self.nodes.values()):

			all_nodes[index] = item.get_position()

		# Get the sum of all of the positions along space dim and divide by the number of nodes
		average_position = np.sum(all_nodes, axis=0) / num_nodes

		return average_position

	# Gets the stable reading of a node
	def get_node_reading(self, node_name):

		return self.nodes[node_name].stable_reading

	# Gets the degree of a node
	def get_node_degree(self, node_name):

		return self.nodes[node_name].get_degree()

	# Updates the neighbors of every node
	def update_neighbors(self):

		# The nodes are all stored in the values
		for node in self.nodes.values():

			# Update
			node.acquire_neighbors()

	# Makes an undirected graph from the nodes and their neighbors
	#
	# Returns a list of all edges
	def make_graph(self):

		# Update the neighbors in the graph
		self.update_neighbors()

		# Go through each node and get their neighbors
		self.edges = []
		for node_name in self.nodes:

			# Get the neighbors
			node_neighbors = self.nodes[node_name].neighbors

			# Go through neighbors
			for neighbor_name in node_neighbors:

				# Make the edge key
				edge_key = "-".join(sorted([node_name, neighbor_name]))

				# Add it to the edge list if it is not already present
				if edge_key not in self.edges:

					self.edges.append(edge_key)

		return self.edges

	# Checks to make sure that the network is connected
	#
	# update : bool
	# If true, neighbors will be updated before checking
	#
	# Return Bool
	def check_connected(self, update=True):

		# Update if needed
		if update:

			self.update_neighbors()

		# Go through each node checking that each degree is greater than 0
		for node in self.nodes:

			# Only one node needs to be disconnnected to fail
			if len(self.nodes[node].neighbors) < 1:

				return False

		return True

	# Gives the total number of nodes in the network
	def total_nodes(self):

		return len(self.nodes)

	# Gets the nodes with the lowest and highest number of neighbors
	#
	# return : string, string
	# names of the lowest, highest
	#def get_interest_nodes(self):

		# Go through each node in the network to find the min and max degrees
		#for name in self.node_names():

			

# Runs the entire consensus filter and visualizes results
class Simulate:

	# Class members
	#
	# environment : Environment()
	# Object that handles the environment where the nodes are
	#
	# network : Network()
	# This node is a member of the network
	# Used to find information about the network

	# Constructor
	def __init__(self):

		# Create the environment
		# Use a constant position and reading for the target
		self.environment = Environment(lambda t : np.array([50.0, 50.0]), lambda t : np.array([50.0]), communication_radius = 50.0)

		# Make a network
		self.make_network()

	# Makes a network, ensuring that it is connected
	#
	# retry_max : int
	# The maximum number of times to create a new network if they keep ending up disconnected
	# Throws exception after too many failures
	def make_network(self, retry_max = 10):

		# Redo if the network is disconnected
		retry_count = 0
		network_connected = False
		while retry_count < retry_max and not network_connected:

			# Create the network
			self.network = Network()

			# Randomly generate 10 nodes with positions around the target
			random_nodes = np.random.rand(10, R_space) * 100

			# Place the nodes into the environment and network
			for node_index, node_position in enumerate(random_nodes):

				# Name is just the index as a string
				node_name = str(node_index)

				# Create the node, use default values
				new_node = Node(self.environment, self.network, node_name)

				# Add to environment
				self.environment.add_node(node_name, node_position)

				# Add to the network
				self.network.add_node(node_name, new_node)

			# Get the connected flag
			network_connected = self.network.check_connected()

			# Increment counter
			retry_count += 1

		# Throw exception
		if not network_connected:

			raise RuntimeError

	# Runs the network and environment for the specified number of iterations
	# TODO: make convergence a condition
	def run(self, iterations = 1000):

		for time_step in range(iterations):

			# Have the network update all of the readings
			self.network.get_all_readings()

			# Advance the environment
			self.environment.advance()

	# Shows information about the network through matplotlib
	# Shows positions of nodes and location of the object being sensed
	def visualize(self):

		# Figure 1 will be the locations of nodes and the tracked object
		plt.figure(1)

		# Set the title
		plt.title("Node and Target Positions")

		# Set the x and y axis names
		plt.xlabel("X location")
		plt.ylabel("Y location")

		# Plot the neighbors of each node
		edges = self.network.make_graph()

		# Make a line for each neighbor
		for edge in edges:

			# Unpack the node names
			first_node, second_node = edge.split("-")

			# Get the coordinates of each node
			first_coordinates = self.environment.get_node_position(first_node)
			second_coordinates = self.environment.get_node_position(second_node)

			# Make a line
			plt.plot([first_coordinates[0], second_coordinates[0]], [first_coordinates[1], second_coordinates[1]], 'bs-', markersize=0.0)

		# Add the locations of every node in the graph
		# Uses the true positions in the environment
		node_positions_x = []
		node_positions_y = []
		for node_name in self.network.node_names():

			# Get the position of the node
			node_position = self.environment.get_node_position(node_name)

			# Add it to the list
			node_positions_x.append(node_position[0])
			node_positions_y.append(node_position[1])

		# Plot points
		plt.plot(node_positions_x, node_positions_y, 'ro', label="Nodes")

		# Plot the target location
		# Use the starting position, target is stationary for now
		target_position = self.environment.get_target_position()
		plt.plot(target_position[0], target_position[1], 'g*', markersize=20.0, label="Target")

		# Set the legend
		plt.legend(loc="best")

		print "Displaying positions of nodes and the target"
		print "Exit window to continue"

		# Show the graph
		plt.show()

if __name__ == "__main__":

	sim = Simulate()

	sim.run()

	sim.visualize()
