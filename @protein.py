import pandas as pd
import numpy as np
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
____________________________________________________________________________________________________________________________________________________________________________________________

#import following dependices

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Flatten,Conv1D, GlobalMaxPooling1D,  Dropout
from tensorflow.keras import Sequential
import progressbar
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

#Mixed precision is the use of both 16-bit and 32-bit floating-point types in a model during training to make it run faster and use less memory.

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras import backend as K
K.clear_session()

#import train dataset

train_protein=pd.read_csv("/kaggle/input/cafa-5-protein-function-prediction/Train/train_terms.tsv",sep="\t")
print(train_protein.shape,"\n")
print(train_protein.head(5),"\n")



print(train_protein["term"].iloc[:5])

#import train_ids dataset 
train_ids=np.load("/kaggle/input/t5embeds/train_ids.npy")
print(train_ids[:5])
print(train_ids.shape,"\n")

#import train_embeddings

print("------>train_embeddings---->")
train_embeddings=np.load("/kaggle/input/t5embeds/train_embeds.npy")
print(train_embeddings[:3],"\n")
print(train_embeddings.shape)



train_embeddings.shape[1]
train_df=pd.DataFrame(train_embeddings,columns=["Column"+str(i) for i in range(1,train_embeddings.shape[1]+1)])

train_df.head(3)


df_duplicated = train_df[train_df.loc[:, train_df.columns != 'ID'].duplicated(keep=False)]
print(df_duplicated.shape[0],"\n")

num_labels=1500
labels=train_protein["term"].value_counts().index[:num_labels].tolist()
print(f"number of labels ->{len(labels)}")

#pie chart
pie_df = train_protein['aspect'].value_counts()
palette_color = sns.color_palette('bright')
plt.pie(pie_df.values, labels=np.array(pie_df.index), colors=palette_color, autopct='%.0f%%')
plt.show()


# Fetch the train_terms data for the relevant labels only
train_updated = train_protein.loc[train_protein['term'].isin(labels)]

bar = progressbar.ProgressBar(maxval=num_labels, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
# Create an empty dataframe of required size for storing the labels,
# i.e, train_size x num_of_labels (142246 x 1500)
train_size = train_ids.shape[0] # len(X)
train_labels = np.zeros((train_size ,num_labels))

# Convert from numpy to pandas

#series for better handling
series_train_protein_ids = pd.Series(train_ids)

# Loop through each label
for i in range(num_labels):
    # For each label, fetch the corresponding train_terms data
    n_train_terms = train_updated[train_updated['term'] ==  labels[i]]
    
    # Fetch all the unique EntryId aka proteins related to the current label(GO term ID)
    label_related_proteins = n_train_terms['EntryID'].unique()
    
    # In the series_train_protein_ids pandas series, if a protein is related
    # to the current label, then mark it as 1, else 0.
    # Replace the ith column of train_Y with with that pandas series.
    train_labels[:,i] =  series_train_protein_ids.isin(label_related_proteins).astype(float)
    
    # Progress bar percentage increase
    bar.update(i+1)

# Notify the end of progress bar 
bar.finish()

# Convert train_Y numpy into pandas dataframe
labels_df = pd.DataFrame(data = train_labels, columns = labels)
print(labels_df.shape)

#enable the gpu
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # Limit GPU memory to 4GB
    except RuntimeError as e:
        print(e)

#model checkpoints

checkpoint_path = 'best_model_checkpoint.h5'

# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#concat train_df and labels_df into single dataframe for feeding to cnn-model
# Combine the protein embeddings with the labels dataframe
train_data = pd.concat([train_df, labels_df], axis=1)

# Split the data into input features (X) and target labels (Y)
X = train_data.iloc[:, :-num_labels].values  # Input features (protein embeddings)

Y = train_data.iloc[:, -num_labels:].values #Target variables

#distributed training for easy training 
"""
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    mixed_precision.set_global_policy('mixed_float16')
    #cnn model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(train_embeddings.shape[1], 1)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='sigmoid'))
    model.compile( tf.keras.optimizers.Adam(learning_rate=1e-2),loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    """


"""# Reshape the input features to match the expected input shape of the CNN model
X = X.reshape(X.shape[0], X.shape[1], 1)
print(X.shape)
"""

"""history=model.fit(X, Y, batch_size=16, epochs=10, validation_split=0.2,callbacks=[checkpoint])"""

#above is distributed training CNN model based try with that also

#visualize plots

history = pd.DataFrame(history.history)
history.loc[:, ['loss']].plot(title="Cross-entropy")
history.loc[:, ['accuracy']].plot(title="Accuracy")

#Sequential api

#strategy = tf.distribute.MirroredStrategy()

#with strategy.scope():
model=tf.keras.Sequential([
    keras.layers.BatchNormalization(input_shape=[train_df.shape[1]]),
    keras.layers.Dense(512,activation="relu"),
    keras.layers.Dense(513,activation="relu"),
    keras.layers.Dense(512,activation="relu"),
        
    keras.layers.Dense(num_labels,activation="sigmoid")
])

#compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(train_df,labels_df,epochs=10,batch_size=5123,verbose=1)

plt.subplot(212)
plt.plot(history.history["loss"],label="training_loss")
plt.plot(history.history["accuracy"],label="accuracy")
plt.legend()
plt.title("loss and accuracy")
plt.show()


#for out of stack of memory to clean that use below code
from tensorflow.keras import backend as K
K.clear_session()

