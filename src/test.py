import torch
str_test = '[1,2,3]'
print(str_test[1:-1])
list_test = list(map(float,str_test[1:-1].split(',')))
print(list_test)