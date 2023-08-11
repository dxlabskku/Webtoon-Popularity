import pandas as pd
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from glob import glob
from tqdm import tqdm
import random
from sentence_transformers import SentenceTransformer, util
import keras
from keras.models import Model
import keras.backend as K
from keras.layers import Input, Reshape, Dense, Embedding, Dropout, LSTM, Lambda, Concatenate, \
    Multiply, RepeatVector, Permute, Flatten, Activation, MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

tf.__version__
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


#data upload
title=[]
img_list = []
label_list = []
df = pd.DataFrame(columns=['title', 'story'])

#naver_launched_one
naver_launched_one=pd.read_csv('./text_file/naver_launched1.csv')

for img_title in tqdm(naver_launched_one['title']):
    title.append(img_title)
    img_path= './thumbnail_file/naver_launched_one_thumbnail/'
    if img_path+img_title+'.jpg' in glob(img_path+'/*'):
        image = np.array(Image.open(img_path+img_title+'.jpg').convert("RGB").resize((224, 224))) 
        img_list.append(image)
        row1 = {'title': img_title, 'story': naver_launched_one[naver_launched_one['title']==img_title]['story'].item()}
        df = pd.concat([df, pd.DataFrame(row1, index=[0])], ignore_index=True)
        label_list.append(1)
    else:
        print(img_title)
        
        

#naver_launched_two
naver_launched_two=pd.read_csv('./text_file/naver_launched2.csv')


for img_title in tqdm(naver_launched_two['title']):
    title.append(img_title)
    
    img_path= './thumbnail_file/naver_launched_two_thumbnail/'
    if img_path+img_title+'.jpg' in glob(img_path+'/*'):
        image = np.array(Image.open(img_path+img_title+'.jpg').convert("RGB").resize((224, 224))) 
        img_list.append(image)
        row1 = {'title': img_title, 'story': naver_launched_two[naver_launched_two['title']==img_title]['story'].item()}
        df = pd.concat([df, pd.DataFrame(row1, index=[0])], ignore_index=True)
        label_list.append(1)
    else:
        print(img_title)

#kakao_launched_one
kakao_launched_one=pd.read_csv('./text_file/kakao_launched1.csv')



for img_title in tqdm(kakao_launched_one['title']):
    title.append(img_title)
    
    img_path= './thumbnail_file/kakao_launched_one_thumbnail/'
    if img_path+img_title+'.jpg' in glob(img_path+'/*'):
        image = np.array(Image.open(img_path+img_title+'.jpg').convert("RGB").resize((224, 224))) 
        img_list.append(image)
        row1 = {'title': img_title, 'story': kakao_launched_one[kakao_launched_one['title']==img_title]['story'].item()}
        df = pd.concat([df, pd.DataFrame(row1, index=[0])], ignore_index=True)
        label_list.append(1)
    else:
        print(img_title)

#kakao_launched_two
kakao_launched_two=pd.read_csv('./text_file/kakao_launched_two.csv')


for img_title in tqdm(kakao_launched_two['title']):
    title.append(img_title)
    
    img_path= './thumbnail_file/kakao_launched_two_thumbnail/'
    if img_path+img_title+'.jpg' in glob(img_path+'/*'):
        image = np.array(Image.open(img_path+img_title+'.jpg').convert("RGB").resize((224, 224))) 
        img_list.append(image)
        row1 = {'title': img_title, 'story': kakao_launched_two[kakao_launched_two['title']==img_title]['story'].item()}
        df = pd.concat([df, pd.DataFrame(row1, index=[0])], ignore_index=True)
        label_list.append(1)
    else:
        print(img_title)

# naver_chalenge

naver_challenge_one=pd.read_csv('./text_file/naver_challenge.csv')



for img_title in tqdm(naver_challenge_one['title']):
    
    
    img_path= './thumbnail_file/naver_challenge_one_thumbnail/'
    if img_path+str(img_title)+'.jpg' in glob(img_path+'/*'):
        title.append(img_title)
        image = np.array(Image.open(img_path+img_title+'.jpg').convert("RGB").resize((224, 224))) 
        img_list.append(image)
        row1 = {'title': img_title, 'story': naver_challenge_one[naver_challenge_one['title']==img_title]['story'].item()}
        df = pd.concat([df, pd.DataFrame(row1, index=[0])], ignore_index=True)
        label_list.append(0)
        
# kakao_challenge_one
kakao_challenge_one=pd.read_csv('./text_file/kakao_challenge1.csv')

for img_title in tqdm(kakao_challenge_one['title']):
    title.append(img_title)
    
    img_path= './thumbnail_file/kakao_challenge_one_thumbnail/'
    if img_path+img_title+'.jpg' in glob(img_path+'/*'):
        image = np.array(Image.open(img_path+img_title+'.jpg').convert("RGB").resize((224, 224))) 
        img_list.append(image)
        row1 = {'title': img_title, 'story': kakao_challenge_one[kakao_challenge_one['title']==img_title]['story'].item()}
        df = pd.concat([df, pd.DataFrame(row1, index=[0])], ignore_index=True)
        label_list.append(0)
    
    
# kakao_challenge_two

kakao_challenge_two=pd.read_csv('./text_file/kakao_challenge2.csv')

for img_title in tqdm(kakao_challenge_two['title']):
    title.append(img_title)
    
    img_path= './thumbnail_file/kakao_challenge_two_thumbnail/'
    if img_path+img_title+'.jpg' in glob(img_path+'/*'):
        image = np.array(Image.open(img_path+img_title+'.jpg').convert("RGB").resize((224, 224))) 
        img_list.append(image)
        row1 = {'title': img_title, 'story': kakao_challenge_two[kakao_challenge_two['title']==img_title]['story'].item()}
        df = pd.concat([df, pd.DataFrame(row1, index=[0])], ignore_index=True)
        label_list.append(0)


# shape
label_list = np.array(label_list)
img_list = np.array(img_list)
img_list=img_list.astype(np.float16)
img_list.shape, label_list.shape


# ko-sentence-transformers
embedder = SentenceTransformer("jhgan/ko-sbert-sts")

sentence=df['story']
corpus_embeddings =embedder.encode(sentence, convert_to_tensor=True)
corpus_embeddings  = corpus_embeddings .cpu().numpy()
corpus_embeddings = tf.convert_to_tensor(corpus_embeddings, dtype=tf.float16)

# gather

gather = list(zip(list(label_list), list(img_list), list(corpus_embeddings)))
random.shuffle(gather)
#gather [i][0]: label , [i][1]: image [i][2]: text
len(corpus_embeddings),len(label_list)


# split train/valid/test
text_train = []
text_valid = []
text_test = []
image_train = []
image_valid = []
image_test = []
y_train = []
y_valid = []
y_test = []


for i in range(len(gather)):
    if i<=len(gather)*0.8: #train
        y_train.append(gather[i][0])
        image_train.append(gather[i][1])
        text_train.append(gather[i][2])
        
    elif len(gather)*0.8<i<len(gather)-(len(gather)*0.1): # valid
        y_valid.append(gather[i][0])
        image_valid.append(gather[i][1])
        text_valid.append(gather[i][2])
        
    else: #test
        y_test.append(gather[i][0])
        image_test.append(gather[i][1])
        text_test.append(gather[i][2])

image_train=np.array(image_train)
image_valid=np.array(image_valid)
image_test=np.array(image_test)
y_train=np.array(y_train)
y_valid=np.array(y_valid)
y_test=np.array(y_test)
text_train=np.array(text_train)
text_valid=np.array(text_valid)
text_test=np.array(text_test)





# Images Model

inp2 = tf.keras.layers.Input([224,224,3]) 
conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='valid')(inp2)
conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='valid')(conv2)
conv2 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))(conv2)
conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='valid')(conv2)
conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='valid')(conv2)
conv2 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))(conv2)
conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='valid')(conv2)
conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='valid')(conv2)
conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='valid')(conv2)
conv2 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))(conv2)
conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='valid')(conv2)
conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='valid')(conv2)
conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='valid')(conv2)
conv2 = tf.keras.layers.Flatten()(conv2)
conv2 = tf.keras.layers.Dense(2000, activation='relu')(conv2)

out = tf.keras.layers.Dense(128)(conv2)

CNN_model = tf.keras.Model(inp2, out)
CNN_model.summary()


# text_model
inputs_text = tf.keras.layers.Input(shape=(768,))

# transform to (None, 768, 1)
reshaped_input = tf.keras.layers.Reshape((768, 1))(inputs_text)
lstm = tf.keras.layers.LSTM(128, name='LSTM')(reshaped_input)


model_lstm = tf.keras.Model(inputs=inputs_text, outputs=lstm)
model_lstm.summary()


#proposed model
inputs_text = tf.keras.layers.Input(shape=(768,))
inp2 = tf.keras.layers.Input([224,224,3])

image_side = CNN_model(inp2)
text_side = model_lstm(inputs_text)

# Concatenate features from images and texts
merged= tf.keras.layers.Concatenate()([image_side, text_side])

# early fusion
merged = tf.keras.layers.Dense(256, activation = 'relu', name='Dense_256')(merged)
merged = tf.keras.layers.Dropout(0.2)(merged)
merged = tf.keras.layers.Dense(128, activation = 'relu', name='Dense_128')(merged)
merged = tf.keras.layers.Dropout(0.2)(merged)
merged = tf.keras.layers.Dense(64, activation = 'relu', name='Dense_64')(merged)
merged = tf.keras.layers.Dropout(0.2)(merged)
output = tf.keras.layers.Dense(1, activation='sigmoid', name = "class")(merged)

     
earlyfusion_model = tf.keras.Model([inputs_text,inp2], output)

earlyfusion_model.summary()


# lion optimizer
# ==============================================================================
"""TF1 implementation of the Lion optimizer."""
from typing import Optional, Union, Callable

import tensorflow.compat.v1 as tf
from tensorflow.python.ops import resource_variable_ops

VType = Union[Callable, float, tf.Tensor]


class Lion(tf.compat.v1.train.Optimizer):
  """Optimizer that implements the discovered algorithm in automl-hero."""

  def __init__(self,
               learning_rate: VType = 0.0001,
               beta1: VType = 0.9,
               beta2: VType = 0.99,
               wd: Optional[VType] = 0.0,
               use_locking=False,
               name="Lion"):
    r"""Construct a new Lion optimizer.
    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta1: A float value or a constant float tensor. The rate to combine
        the gradient and the moment estimate.
      beta2: A float value or a constant float tensor. The exponential decay
        rate for the moment estimate.
      wd: Optional[A float value or a constant float tensor].
        The decoupled weight decay.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
    """
    super(Lion, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._wd = None if isinstance(wd, float) and wd < 0 else wd

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._wd_t = None

  def _create_slots(self, var_list):
    # Create slots for the moment.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    wd = self._call_if_callable(self._wd)

    self._lr_t = tf.convert_to_tensor(lr, name="learning_rate")
    self._beta1_t = tf.convert_to_tensor(beta1, name="beta1")
    self._beta2_t = tf.convert_to_tensor(beta2, name="beta2")
    if wd is not None:
      self._wd_t = tf.convert_to_tensor(wd, name="weight_decay")

  def _apply_dense_shared(self, grad, var):
    m = self.get_slot(var, "m")

    lr_t = tf.cast(self._lr_t, dtype=var.dtype)
    beta1_t = tf.cast(self._beta1_t, dtype=var.dtype)
    beta2_t = tf.cast(self._beta2_t, dtype=var.dtype)
    if self._wd_t is None:
      weight_decay_t = None
    else:
      weight_decay_t = tf.cast(self._wd_t, dtype=var.dtype)

    updates_grad = tf.sign(m * beta1_t + grad * (1. - beta1_t))
    if weight_decay_t is not None:
      updates_grad = updates_grad + var * weight_decay_t

    var_update = tf.assign_sub(
        var, lr_t * updates_grad, use_locking=self._use_locking)
    with tf.control_dependencies([var_update]):
      m_update = tf.assign(m, m * beta2_t + grad * (1. - beta2_t))
    return tf.group(*[var_update, m_update])

  def _apply_dense(self, grad, var):
    return self._apply_dense_shared(grad, var)

  def _resource_apply_dense(self, grad, var):
    return self._apply_dense_shared(grad, var)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    m = self.get_slot(var, "m")
    lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
    wd_t = tf.cast(self._wd_t, var.dtype.base_dtype)

    m_update = tf.assign(m, m * beta1_t, use_locking=self._use_locking)
    with tf.control_dependencies([m_update]):
      m_update = scatter_add(m, indices, grad * (1. - beta1_t))
      with tf.control_dependencies([m_update]):
        var_update = tf.assign_sub(
            var,
            lr_t * (tf.sign(m) + var * wd_t),
            use_locking=self._use_locking)
        with tf.control_dependencies([var_update]):
          m_update = scatter_add(m, indices, grad * (beta1_t - 1.))
          with tf.control_dependencies([m_update]):
            m_update = tf.assign(
                m, m * beta2_t / beta1_t, use_locking=self._use_locking)
            with tf.control_dependencies([m_update]):
              m_update = scatter_add(m, indices, grad * (1. - beta2_t))
    return tf.group(*[var_update, m_update])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
        grad.values,
        var,
        grad.indices,
        lambda x, i, v: tf.scatter_add(
            x,
            i,
            v,
            use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    with tf.control_dependencies(
        [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(grad, var, indices,
                                     self._resource_scatter_add)

# ==============================================================================

lion=Lion()
lion_optimizer = Lion(learning_rate=0.005, wd=0.01)

# Compile the model
earlyfusion_model.compile(optimizer=lion, loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)
history = earlyfusion_model.fit([text_train, image_train], y_train,epochs=20,callbacks=early_stopping ,batch_size=64, validation_data=([text_valid, image_valid], y_valid))

# Get the training and validation loss values from the history object
train_loss_list = history.history['loss']
valid_loss_list = history.history['val_loss']
train_acc_list = history.history['accuracy']
valid_acc_list = history.history['val_accuracy']

# Print the loss values for each epoch
for epoch, (train_loss, valid_loss) in enumerate(zip(train_loss_list, valid_loss_list)):
    print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}")
    
# evaluation
y_pred = earlyfusion_model.predict([text_test,image_test])
y_pred_labels = np.round(y_pred).flatten()

print(classification_report(y_test, y_pred_labels, digits=5))



# draw plot
plt.plot(train_loss_list, label = 'train')
plt.plot(valid_loss_list, label = 'valid')
plt.legend()
plt.show()

plt.plot(train_acc_list, label = 'train')
plt.plot(valid_acc_list, label = 'valid')
plt.legend()
plt.show()


