import random

sample_ratio = 0.3
count = 0
total = 1000
for i in range(total):
    num = random.uniform(0,1)
    if num < sample_ratio:
        count += 1
print(f"The sample_ratio is:{sample_ratio*100}%, but the real value is :{count/total*100}%")