import PSO
import sys
import os
import json

sys.path.append(os.path.abspath('../tools'))
import data_prep

pso = PSO.PSO()

# user set up
number, contrast = 4,5
opt_method = "Gen"

test_size = 200
is_binary_mask = False
## use saved json file
with open(f"Gen_result/4vs5_Sun Jul 21 16:00:08 2024.json", "r") as f:
    data = json.load(f)
    is_binary_mask = False
guess = data["result"]

"""----------------------------------------------------"""

print("Mask Analysed:")
print(guess)
print(len(guess))
print("\n")

# keep the order of number and contrast
if number > contrast:
    contrast, number = number, contrast

# Get data
_, data1 = data_prep.get_MNIST(test_size, number)
_, data2 = data_prep.get_MNIST(test_size, contrast)

_, fp_fn = pso.test([number,contrast], [data1, data2], guess, is_binary_mask=is_binary_mask)

# use if no_mask_result.json file is not ready
_, data1 = data_prep.get_MNIST(test_size, number)
_, data2 = data_prep.get_MNIST(test_size, contrast)
_, fp_fn_no_mask = pso.test([number,contrast], [data1, data2], [], is_binary_mask=True)
print(f"This is no mask result(newly calculated): {fp_fn_no_mask}")


with open("no_mask_result.json", "r") as f:
    no_mask = json.load(f)
no_mask = no_mask["result"][contrast][number]

print(f"This is with mask result: {fp_fn}")
print(f"This is no mask result: {no_mask}")

# Deciding the best fp_fn/Saving the result
if no_mask > fp_fn:
    print(f"The optimisation result is better by {no_mask - fp_fn}")
    filename = f"BestMask/{number}vs{contrast}_best.json"
    result = {"opt_method":opt_method, "fp_fn":fp_fn, "mask":guess}

    if os.path.isfile(filename):
        print("Previous optimisation result exists")
        with open(filename, "r") as f:
            prev = json.load(f)
            print(f"Previous optimistion result: {prev['fp_fn']}")
        if prev["fp_fn"] > result["fp_fn"]:
            print("Discarding previous optimisation")
            print(f"Saving the optimisation result in {filename}")
            with open(filename, "w") as f:
                json.dump(result, f)
            print("Saved")
        else:
            print("Previous optimisation result is better \nNot saving the result")
    else:
        print(f"Saving the optimisation result in {filename}")
        with open(filename, "w") as f:
            json.dump(result, f)
        print("Saved")
else:
    print("No mask result is better")
    print("No saved file")
