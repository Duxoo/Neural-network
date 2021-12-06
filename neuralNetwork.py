import numpy as np
import scipy.special as ss

class neuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # amount of nodes in input, hidden and output layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        # matrix of weight coef
        self.wih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        # sigmoida
        self.activation_function = lambda x: ss.expit(x)
        

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # input signals for hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # output signals for hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # input signals for output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # output signals for output layer
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                   np.transpose(hidden_outputs))
        self.wih += self.learning_rate * np.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)),
                                                   np.transpose(inputs))
        
        

    def query(self, inputs_list):
        # reorganise input list to 2n array
        inputs = np.array(inputs_list, ndmin=2).T
        # input signals for hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # output signals for hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # input signals for output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # output signals for output layer
        return self.activation_function(final_inputs)
