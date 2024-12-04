import os

path = 'train/'
ids = []

for one in os.listdir(path):
    one = int(one.split('.')[0])
    ids.append(one)

print(ids)