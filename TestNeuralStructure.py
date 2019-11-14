from NeuralIT import *
from graphics import *
from math import *

inputs = 2
layers = [2]
learning_rate = 0.1

testNIT = NITStructure(inputs, layers, learning_rate)

# desired output is number set should have 2 values more than 100 and 2 values less than 100
# Get a known data set
testValues = []

testSetSize = 1000
iterateOverSet = 10  # We can iterate over the same data multiple times

for x in range(testSetSize):
    testValues.append({'values': [], 'target': False})

    for y in range(inputs):
        testValues[x]['values'].append(random())
        if testValues[x]['values'][y] >= 0.5:
            testValues[x]['values'][y] = 1
        else:
            testValues[x]['values'][y] = 0

    if testValues[x]['values'][0] == 1 or testValues[x]['values'][1] == 1:
        testValues[x]['target'] = 1
    else:
        testValues[x]['target'] = 0


last_ten = []

win = GraphWin("Output", 1100, 800)
win.setBackground(color_rgb(0, 0, 0))
frame_number = 0

error_line = Line(Point(10, 500), Point(1000, 500))
error_line.setOutline(color_rgb(128, 128, 128))
error_line.draw(win)

positive_error_log = []
negative_error_log = []


testValues = [{'values': [0.1, 1], 'target': 0},
              {'values': [1, 0.1], 'target': 1}]

for index in range(len(testValues) * iterateOverSet):

    idx = index % testValues.__len__()

    testNIT.set_inputs(testValues[idx]['values'])
    result = testNIT.get_output()

    log_output("Weights", "\n\n\n\n\nInputs:  (" + str(testValues[idx]['values'][0])
               + ", " + str(testValues[idx]['values'][1]) + ") "
               + "  Output: " + str(result))

    error = testValues[idx]['target'] - result

    output_error = list()
    output_error.append(error)

    if last_ten.__len__() == 10:
        last_ten.pop()

    if fabs(output_error[0]) < 0.5:
        correct = 1
    else:
        correct = 0

    # Always train, even on correct guesses
    testNIT.train(output_error)

    last_ten.insert(0, correct)

    last_ten_sum = 0

    for i in range(last_ten.__len__()):
        last_ten_sum = last_ten_sum + last_ten[i]

    log_output("OutputError", "                                                                                       >>>> This Error: " + str(output_error[0]) + "    Correct In Last Ten: " + str(last_ten_sum))

    if idx == 0:
        frame_number += 1
        testNIT.set_frame_number(frame_number)
        testNIT.draw_structure(win)

        data_point = Point(10 + frame_number, 500 - (output_error[0] * 200))
        data_point.setOutline(color_rgb(255, 0, 0))
        data_point.draw(win)

    log_output("OutputError", "\n ************************************************************************************"
                              "\n ************************************************************************************"
                              "\n ************************************************************************************"
                              ""
                              ""
                              ""
                              ""
                              ""
                              "")

win.getMouse()
win.close



