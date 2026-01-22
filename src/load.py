import pickle
# import numpy as np
import matplotlib.pyplot as plt

og = "data/cifar-10-batches-py-downloaded/data_batch_1"
my = "data/cifar10/distilled_data_batch_10000"

with open(my, "rb") as f:
    batch = pickle.load(f, encoding="bytes")

print(batch.keys())

data = batch[b'data']        # numpy array, shape (N, 3072)
# labels = batch[b'labels']    # list of ints
# filenames = batch[b'filenames']
batch_label = batch[b'batch_label']

print(data.shape)            # (10000, 3072)
print(type(data))
# print(labels[:10])
# print(filenames[:3])

# img = data[0]
# img = img.reshape(3, 32, 32).transpose(1, 2, 0)

# plt.imshow(img)
# plt.axis("off")
# plt.show()
