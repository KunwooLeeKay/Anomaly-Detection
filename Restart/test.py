import pickle


with open('find_optimal.pickle', 'rb') as pickle_data:
    data = pickle.load(pickle_data)


print(data)

with open('IQR_find_optimal.pickle', 'rb') as pickle_data:
    data = pickle.load(pickle_data)

print(data)