from NeuralIT import *

test_input_count = 2
test_neuron_count = 20

# testLayer = NITLayer(test_input_count, test_neuron_count, 0.001)
# testOutput = NITNeuron(test_neuron_count, 0.001)

# desired output is number set should have 2 values more than 100 and 2 values less than 100
# Get a known data set
testValues = []

for x in range(1000):
    testValues.append({'values': [], 'desired_result': False})

    # above_hundred = 0
    # below_hundred = 0
    #
    # for y in range(test_input_count):
    #     testValues[x]['values'].append(randrange(0, 200))
    #
    #     if testValues[x]['values'][y] > 100:
    #         above_hundred = above_hundred + 1
    #     if testValues[x]['values'][y] < 100:
    #         below_hundred = below_hundred + 1
    #
    # if above_hundred == 2 and below_hundred == 2:
    #     testValues[x]['desired_result'] = 1
    # else:
    #     testValues[x]['desired_result'] = -1

    for y in range(test_input_count):
        testValues[x]['values'].append(randrange(0, 100))

    if testValues[x]['values'][0] > testValues[x]['values'][1]:
        testValues[x]['desired_result'] = 1
    else:
        testValues[x]['desired_result'] = -1


last_ten = []

for idx in range(testValues.__len__()):

    testLayer.set_inputs(testValues[idx]['values'])
    result = testLayer.get_outputs()

    testOutput.set_inputs(result)
    final_result = testOutput.get_output()

    #delete me
    #testLayer.print_neurons()

    this_error = testValues[idx]['desired_result'] - final_result
    testLayer.train(this_error)
    testOutput.train(this_error)

    if last_ten.__len__() == 10:
        last_ten.pop()

    if this_error != 0:
        correct = 0
    else:
        correct = 1

    last_ten.insert(0, correct)

    last_ten_sum = 0

    for i in range(last_ten.__len__()):
        last_ten_sum = last_ten_sum + last_ten[i]

    print("This Error: " + str(this_error) + "    Correct In Set: " + str(last_ten_sum))






