from random import *
from math import *
import numpy as np
from graphics import *

frame_number = 0


def log_output(log_type, val):
    if log_type == 'Weights' or log_type == 'OutputError' or log_type == 'WeightsError' or log_type == 'full':
        # if log_type == 'WeightsErrorFIX':
        print(val)


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


class NITInput:

    def __init__(self, weight="random"):
        if weight == "random":
            # weight = (random() - 0.5) * 2
            weight = random()

        self.weight = weight
        self.value = 0

    def set_weight(self, weight):
        self.weight = weight
        return True

    def get_weight(self):
        return self.weight

    def set_value(self, input_val):
        self.value = input_val

    def get_value(self):
        return self.value

    def get_sum(self):
        return self.value * self.weight


class NITNeuron:

    def __init__(self, input_count, learning_rate=1):

        self.learning_rate = learning_rate
        self.inputs = []

        for i in range(input_count):
            self.inputs.append(NITInput())

        # Create the bias input
        self.inputs.append(NITInput())
        self.inputs[input_count].set_value(1)
        self.input_count = input_count + 1

    def print_inputs(self):
        for i in range(self.input_count):
            log_output("full", "Input " + str(i)
                       + ":  Weight = " + str(self.inputs[i].get_weight())
                       + "  Value = " + str(self.inputs[i].get_value()))

        log_output("full", "  **    Sum = " + str(self.get_sum()))
        log_output("full", "  ** Output = " + str(self.get_output()))

    def draw_inputs(self, win, x_offset, y_offset):

        global frame_number

        for i in range(self.input_count):
            x = x_offset + frame_number
            y = y_offset + (50 * i) - (self.inputs[i].get_weight() * 50)

            data_point = Point(x, y)
            data_point.setOutline(color_rgb(255, 255, 255))
            data_point.draw(win)

    def set_input_value(self, index, value):
        self.inputs[index].set_value(value)

    def set_inputs(self, values):
        # Don't allow anyone to set the bias input
        for i in range(self.input_count - 1):
            self.inputs[i].set_value(values[i])

    def train(self, error):

        input_errors = list()
        weight_sum = self.get_sum_weights()

        if weight_sum == 0:
            weight_sum = 0.000000001

        if error != 0:
            for i in range(self.input_count):
                old_weight = self.inputs[i].get_weight()
                unmitigated_weight_delta = (error ** 2) * (old_weight / weight_sum)
                weight_delta = unmitigated_weight_delta * self.learning_rate
                if error < 0:
                    weight_delta = 0 - weight_delta
                new_weight = old_weight + weight_delta
                self.inputs[i].set_weight(new_weight)

                log_output("WeightsError", "Error: " + str(error))
                log_output("Weights", "Input " + str(i) + " -  Weights:  Old: " + str(old_weight)
                           + "  Delta: " + str(weight_delta)
                           + "    New: " + str(new_weight))

                input_errors.append(unmitigated_weight_delta)

        # Return the weight delta for back propagation:  See page 78 of book
        return input_errors

    def get_sum(self):
        input_sum = 0

        for i in range(self.input_count):
            input_sum = input_sum + self.inputs[i].get_sum()

        return input_sum

    def get_sum_weights(self):
        weight_sum = 0

        for i in range(self.input_count):
            weight_sum = weight_sum + self.inputs[i].get_weight()

        return weight_sum

    def get_output(self):
        input_sum = self.get_sum()
        output = self.activation(input_sum)

        return output

    @staticmethod
    def activation(num):
        return sigmoid(num)


class NITLayer:

    def __init__(self, input_count, neuron_count, learning_rate=1):

        self.neuron_count = neuron_count
        self.input_count = input_count
        self.neurons = []

        for i in range(neuron_count):
            self.neurons.append(NITNeuron(self.input_count, learning_rate))

    def set_inputs(self, input_values):
        for i in range(self.neuron_count):
            self.neurons[i].set_inputs(input_values)

    def get_outputs(self):
        output_array = []

        for i in range(self.neuron_count):
            output_array.append(self.neurons[i].get_output())

        return output_array

    def get_neuron_count(self):
        return self.neuron_count

    def train(self, error_table):

        # error_table is a table with the errors from each neuron
        neuron_errors = []

        for i in range(self.neuron_count):

            # Grab the errors from each neuron
            if i == 0:
                neuron_errors = self.neurons[self.neuron_count - 1].train(error_table[i])
            else:
                neuron_errors.__add__(self.neurons[self.neuron_count - 1].train(error_table[i]))

        #  Return layer weight delta for back propagation
        return neuron_errors

    def print_neurons(self):
        for i in range(self.neuron_count):
            log_output("full", "********* Neuron " + str(i))
            self.neurons[i].print_inputs()

    def draw_layer(self, win, x_offset):

        y_offset = 300

        for i in range(self.neuron_count):
            self.neurons[i].draw_inputs(win, x_offset, y_offset)
            y_offset += 150


class NITStructure:

    def __init__(self, input_count, neuron_counts, learning_rate):

        # neuron_counts is an array of integers with each member being a count of neurons in this layer
        # (+1 for the output layer)
        self.layer_count = len(neuron_counts)
        self.layers = []

        for i in range(self.layer_count):

            # If it's the first row, use the input count, for all subsequent rows, use the previous neuron count
            if i == 0:
                self.layers.append(NITLayer(input_count, neuron_counts[i], learning_rate))
            else:
                self.layers.append(NITLayer(neuron_counts[i - 1], neuron_counts[i], learning_rate))

        #  Add the output neuron / layer
        self.layers.append(NITLayer(neuron_counts[self.layer_count - 1], 1, learning_rate))
        self.layer_count = self.layer_count + 1

    def set_inputs(self, input_values):
        self.layers[0].set_inputs(input_values)

    def get_output(self):
        for i in range(self.layer_count):
            this_result = self.layers[i].get_outputs()

            #  Send it on to the next layer until we reach the last layer, then return the result
            if i < self.layer_count - 1:
                self.layers[i + 1].set_inputs(this_result)
            else:
                return self.activation(this_result[0])   # todo:  Change back to  this_result[0] ???

    def train(self, error_table):

        for i in range(self.layer_count):
            log_output("Weights", "\n ROW " + str((self.layer_count - 1) - i) + ".............")
            self.layers[(self.layer_count - 1) - i].print_neurons()
            #  We use the current error table to calculate the error table for the previous layer
            error_table = self.layers[(self.layer_count - 1) - i].train(error_table)

    @staticmethod
    def activation(num):

        return sigmoid(num)

    def draw_structure(self, win):

        x_offset = 20

        for i in range(self.layer_count):
            self.layers[i].draw_layer(win, x_offset)
            x_offset += 550

    def set_frame_number(self, num):
        global frame_number
        frame_number = num
