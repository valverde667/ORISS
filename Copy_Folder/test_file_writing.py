
import numpy as np




test_file = open("test_file.txt", "w")

x = np.linspace(0, 100, 101)

for number in x:
    if number%5 == 0:
        test_file.write('{}'.format(number) + "\n")
    else:
        test_file.write('{},'.format(number))

test_file.close()
