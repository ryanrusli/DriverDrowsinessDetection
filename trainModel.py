from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import LSTMModel
from data import Dataset
import time
import os
import os.path
import numpy as np
import sys
import pandas as pd
import csv


seq= 26
hyper = 26
df = pd.read_csv("./data/data_file.csv", header = None)
test_no = df[df.iloc[:,0] == 'testing'].shape[0]
train_no = df[df.iloc[:,0] == 'training'].shape[0]

checkpointer = ModelCheckpoint(filepath=os.path.join('data', 'checkpoints', 'lstm' + '-' + 'features' + '.{epoch:03d}-{val_loss:.3f}.hdf5'),verbose=1,save_best_only=True)

early_stopper = EarlyStopping(patience=5)

data = Dataset(
        seq_length=seq,
        class_limit=2,
    )

x_train, y_train = data.get_all_sequences_in_memory('training', hyper, seq)
x_test, y_test = data.get_all_sequences_in_memory('testing', hyper, seq)

lstmModel = LSTMModel(len(data.classes),'lstm',data.seq_length, None)

x_train=np.ravel(x_train)
x_train=x_train.reshape((52, seq,-1))

print("##################################################")

lstmModel.model.fit(x_train,y_train,
        batch_size=4,
        validation_data=(x_train, y_train),
        verbose=1,
        callbacks=[early_stopper],
		epochs=10)

with open(os.path.join('data', 'data_file.csv'), 'r') as video_file:
	reader = csv.reader(video_file)
	data = list(reader)

filenames = []
for videos in data:
	if(videos[0] == 'training'):
		i = 1
		while i <= int(hyper/seq):
			cnt = i*seq
			filenames.append(videos[2] + '-' + str(seq) + '-' + 'features' + str(cnt)+'.npy')
			i+=1

k = 0
print("\n##################################################\n")
print("Detection with TRAINING Data\n")
print("##################################################\n")
predictions = lstmModel.model.predict(x_train)
loss, accuracy = lstmModel.model.evaluate(x_train, y_train)

for j in predictions:
	if j[0]>j[1]:
		print(filenames[k], "- Driver is alert with the confidence of",(j[0]*100),"%\n")
	else:
		print(filenames[k],"- Driver is drowsy with the confidence of",(j[1]*100),"%\n")
	k+=1

print("\n##################################################\n")
print("Detection with TESTING Data\n")
print("##################################################\n")
x_test=np.ravel(x_test)
x_test=x_test.reshape((8,hyper,-1))
predictions = lstmModel.model.predict(x_test)
loss, accuracy = lstmModel.model.evaluate(x_test, y_test)
with open(os.path.join('data', 'data_file.csv'), 'r') as video_file:
	reader = csv.reader(video_file)
	data = list(reader)

filenames = []
for videos in data:
	if(videos[0] == 'testing'):
		i = 1
		while i <= int(hyper/seq):
			cnt = i*seq
			filenames.append(videos[2] + '-' + str(seq) + '-' + 'features' + str(cnt)+'.npy')
			i+=1
k = 0

for j in predictions:
	if j[0]>j[1]:
		print(filenames[k],"- Driver is alert with the confidence of",(j[0]*100),"%\n")
	else:
		print(filenames[k],"- Driver is drowsy with the confidence of",(j[1]*100),"%\n")
	k+=1
			