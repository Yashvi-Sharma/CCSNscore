import tensorflow as tf
from tensorflow import keras
#from keras.utils.np_utils import to_categorical
#from tensorflow.keras.layers import BatchNormalization
from keras import regularizers, optimizers
#import kerastuner as kt
#from kerastuner import HyperModel

from tensorflow.keras.layers import (
	Conv1D, Conv2D,
	Dense,
	Dropout,
	Flatten,
	MaxPooling1D,MaxPooling2D,LSTM,Bidirectional,Masking,GRU
)
import numpy as np 

class Model:
	def __init__(self,inp_shapes, num_classes, channels, mc):
		self.inp_shape_spec, self.inp_shape_spec_dmdt, self.inp_shape_lcr, self.inp_shape_lcg, self.inp_shape_dmdtr, self.inp_shape_dmdtg = inp_shapes
		self.num_classes = num_classes
		self.channels = channels
		self.mc = mc
		self.inp_shape_spec_dmdt = (11,11,1)


	def get_dropout(self,input_tensor,p=0.5):
		if self.mc:
			return Dropout(p)(input_tensor, training=True)
		else:
			return Dropout(p)(input_tensor)

	def create_model_spec(self):
		inputs_spec = keras.layers.Input(shape=self.inp_shape_spec)
		x = Masking(mask_value=0.)(inputs_spec)
		x = Bidirectional(LSTM(24,return_sequences=True))(x)
		x = self.get_dropout(x,p=0.5)
		# x = Bidirectional(LSTM(24,return_sequences=True))(x)
		# x = self.get_dropout(x,p=0.1)
		x = Bidirectional(LSTM(24))(x)
		x = self.get_dropout(x,p=0.1)
		x = Dense(64,activation='relu')(x)
		x = self.get_dropout(x,p=0.1)
		x = Dense(8,activation='relu')(x)
		# x = self.get_dropout(x,p=0.3)
		self.model_spec = keras.Model(inputs_spec,x)

	def create_model_lcr(self):
		inputs_lcr = keras.layers.Input(shape=self.inp_shape_lcr)
		y = Masking(mask_value=0.)(inputs_lcr)
		y = LSTM(24,return_sequences=True)(y)
		y = self.get_dropout(y,p=0.5)
		y = LSTM(24,return_sequences=True)(y)
		y = self.get_dropout(y,p=0.5)
		y = LSTM(24)(y)
		y = self.get_dropout(y,p=0.5)
		y = Dense(8,activation='relu')(y)
		self.model_lcr = keras.Model(inputs_lcr,y)

	def create_model_lcg(self):
		inputs_lcg = keras.layers.Input(shape=self.inp_shape_lcg)
		z = Masking(mask_value=0.)(inputs_lcg)
		z = LSTM(24,return_sequences=True)(z)
		z = self.get_dropout(z,p=0.5)
		z = LSTM(24,return_sequences=True)(z)
		z = self.get_dropout(z,p=0.5)
		z = LSTM(24)(z)
		z = self.get_dropout(z,p=0.5)
		z = Dense(8,activation='relu')(z)
		self.model_lcg = keras.Model(inputs_lcg,z)

	def create_model_lcr_dmdt(self):
		inputs_dmdtr = keras.layers.Input(shape=self.inp_shape_dmdtr)
		y = Conv2D(filters=16,kernel_size=(3,3),activation='relu',
				   kernel_initializer='random_uniform',strides=1)(inputs_dmdtr)
		y = self.get_dropout(y,p=0.5)
		y = Conv2D(filters=8, kernel_size=(3,3), activation='relu')(y)#, kernel_regularizer=regularizers.l2(0.05))(y)
		y = self.get_dropout(y,p=0.1)
		y = MaxPooling2D(pool_size=(2,2))(y)
		y = Flatten()(y)
		y = Dense(64,activation='relu')(y)
		y = self.get_dropout(y,p=0.1)
		y = Dense(8,activation='relu')(y)
		self.model_dmdtr = keras.Model(inputs_dmdtr,y)

	def create_model_lcg_dmdt(self):
		inputs_dmdtg = keras.layers.Input(shape=self.inp_shape_dmdtg)
		z = Conv2D(filters=16,kernel_size=(3,3),activation='relu',
				   kernel_initializer='random_uniform',strides=1)(inputs_dmdtg)
		z = self.get_dropout(z,p=0.5)
		z = Conv2D(filters=8, kernel_size=(3,3), activation='relu')(z)#, kernel_regularizer=regularizers.l2(0.05))(z)
		z = self.get_dropout(z,p=0.1)
		z = MaxPooling2D(pool_size=(2,2))(z)
		z = Flatten()(z)
		z = Dense(64,activation='relu')(z)
		z = self.get_dropout(z,p=0.1)
		z = Dense(8,activation='relu')(z)
		self.model_dmdtg = keras.Model(inputs_dmdtg,z)

	def main_model(self):
		if('spec' in self.channels):
			self.create_model_spec()
		else:
			self.model_spec = None

		if('lcr' in self.channels):
			self.create_model_lcr()
		else:
			self.model_lcr = None

		if('lcg' in self.channels):
			self.create_model_lcg()
		else:
			self.model_lcg = None

		if('lcrdmdt' in self.channels):
			self.create_model_lcr_dmdt()
		else:
			self.model_dmdtr = None

		if('lcgdmdt' in self.channels):
			self.create_model_lcg_dmdt()
		else:
			self.model_dmdtg = None

		modelarr = np.array([self.model_spec, self.model_lcr, self.model_lcg,
							 self.model_dmdtr, self.model_dmdtg])
		modelarr = np.array([m for m in modelarr if m is not None])
		modeloutputs = [m.output for m in modelarr if m is not None]
		modelinputs = [m.input for m in modelarr if m is not None]
		if(len(modeloutputs)>1):
			combined = keras.layers.concatenate(modeloutputs)
		else:
			combined = modeloutputs[0]
		w = Dense(self.num_classes, activation="sigmoid")(combined)
		self.model = keras.Model(inputs=modelinputs, outputs=w)

	def model_for_tuner(self,hp):
		hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])

		inputs_spec = keras.layers.Input(shape=self.inp_shape_spec)
		x = Masking(mask_value=0.)(inputs_spec)
		x = Bidirectional(LSTM(hp.Int('lstm_spec0',min_value=16,max_value=24, step=4),return_sequences=True))(x)
		x = Dropout(hp.Float('dropout_spec0', min_value=0.1, max_value=0.7, step=0.3))(x)
		x = Bidirectional(LSTM(hp.Int('lstm_spec1',min_value=4,max_value=16, step=4)))(x)
		x = Dropout(hp.Float('dropout_spec1', min_value=0.1, max_value=0.7, step=0.3))(x)
		x = Dense(hp.Int('units_spec0', min_value=32, max_value=64, step=16),activation='relu')(x)
		x = Dropout(hp.Float('dropout_spec2', min_value=0.1, max_value=0.7, step=0.3))(x)
		x = Dense(hp.Int('units_spec1', min_value=8, max_value=24, step=8),activation='relu')(x)
		model_spec = keras.Model(inputs_spec,x)

		inputs_lcr = keras.layers.Input(shape=self.inp_shape_lcr)
		y = Masking(mask_value=0.)(inputs_lcr)
		y = LSTM(hp.Int('lstm_lcr0',min_value=4,max_value=24, step=4),return_sequences=True)(y)
		y = Dropout(hp.Float('dropout_lcr0', min_value=0.1, max_value=0.9, step=0.2))(y)
		y = LSTM(hp.Int('lstm_lcr1',min_value=4,max_value=24, step=4),return_sequences=True)(y)
		y = Dropout(hp.Float('dropout_lcr1', min_value=0.1, max_value=0.9, step=0.2))(y)
		y = LSTM(hp.Int('lstm_lcr2',min_value=4,max_value=24, step=4))(y)
		# y = Dense(hp.Int('units_lcr0', min_value=32, max_value=512, step=32),activation='relu')(y)
		y = Dropout(hp.Float('dropout_lcr2', min_value=0.1, max_value=0.9, step=0.2))(y)
		y = Dense(hp.Int('units_lcr1', min_value=8, max_value=64, step=8),activation='relu')(y)
		model_lcr = keras.Model(inputs_lcr,y)

		inputs_lcg = keras.layers.Input(shape=self.inp_shape_lcg)
		z = Masking(mask_value=0.)(inputs_lcg)
		z = LSTM(hp.Int('lstm_lcg0',min_value=4,max_value=24, step=4),return_sequences=True)(z)
		z = Dropout(hp.Float('dropout_lcg0', min_value=0.1, max_value=0.9, step=0.2))(z)
		z = LSTM(hp.Int('lstm_lcg1',min_value=4,max_value=24, step=4),return_sequences=True)(z)
		z = Dropout(hp.Float('dropout_lcg1', min_value=0.1, max_value=0.9, step=0.2))(z)
		z = LSTM(hp.Int('lstm_lcg2',min_value=4,max_value=24, step=4))(z)
		# z = Dense(hp.Int('units_lcg0', min_value=32, max_value=512, step=32),activation='relu')(z)
		z = Dropout(hp.Float('dropout_lcg2', min_value=0.1, max_value=0.9, step=0.2))(z)
		z = Dense(hp.Int('units_lcg1', min_value=8, max_value=64, step=8),activation='relu')(z)
		model_lcg = keras.Model(inputs_lcg,z)


		inputs_dmdtr = keras.layers.Input(shape=self.inp_shape_dmdtr)
		c = Conv2D(filters=hp.Choice('filt_dmdtr0', values=[16,32,64]),kernel_size=3,activation='relu',
				   kernel_initializer='random_uniform',strides=1)(inputs_dmdtr)
		c = Dropout(hp.Float('dropout_dmdtr0', min_value=0.1, max_value=0.9, step=0.2))(c)
		c = Conv2D(filters=hp.Choice('filt_dmdtr1', values=[8,16,32]), kernel_size=3, activation='relu')(c)
		c = Dropout(hp.Float('dropout_dmdtr1', min_value=0.1, max_value=0.9, step=0.2))(c)
		c = MaxPooling2D(pool_size=2)(c)
		c = Flatten()(c)
		c = Dense(hp.Int('units_dmdtr0', min_value=8, max_value=64, step=8),activation='relu')(c)
		c = Dropout(hp.Float('dropout_dmdtr2', min_value=0.1, max_value=0.9, step=0.2))(c)
		c = Dense(hp.Int('units_dmdtr1', min_value=8, max_value=64, step=8),activation='relu')(c)
		model_dmdtr = keras.Model(inputs_dmdtr,c)

		inputs_dmdtg = keras.layers.Input(shape=self.inp_shape_dmdtg)
		b = Conv2D(filters=hp.Choice('filt_dmdtg0', values=[16,32,64]),kernel_size=3,activation='relu',
				   kernel_initializer='random_uniform',strides=1)(inputs_dmdtg)
		b = Dropout(hp.Float('dropout_dmdtg0', min_value=0.1, max_value=0.9, step=0.2))(b)
		b = Conv2D(filters=hp.Choice('filt_dmdtg1', values=[8,16,32]), kernel_size=3, activation='relu')(b)
		b = Dropout(hp.Float('dropout_dmdtg1', min_value=0.1, max_value=0.9, step=0.2))(b)
		b = MaxPooling2D(pool_size=2)(b)
		b = Flatten()(b)
		b = Dense(hp.Int('units_dmdtg0', min_value=8, max_value=64, step=8),activation='relu')(b)
		b = Dropout(hp.Float('dropout_dmdtg2', min_value=0.1, max_value=0.9, step=0.2))(b)
		b = Dense(hp.Int('units_dmdtg1', min_value=8, max_value=64, step=8),activation='relu')(b)
		model_dmdtg = keras.Model(inputs_dmdtg,b)

		modelarr = []
		if('spec' in self.channels):
			modelarr.append(model_spec)

		if('lcr' in self.channels):
			modelarr.append(model_lcr)

		if('lcg' in self.channels):
			modelarr.append(model_lcg)

		if('lcrdmdt' in self.channels):
			modelarr.append(model_dmdtr)

		if('lcgdmdt' in self.channels):
			modelarr.append(model_dmdtg)

		modelarr = np.array(modelarr)
		modeloutputs = [m.output for m in modelarr]
		modelinputs = [m.input for m in modelarr]
		if(len(modeloutputs)>1):
			combined = keras.layers.concatenate(modeloutputs)
		else:
			combined = modeloutputs[0]
		w = Dense(self.num_classes, activation="sigmoid")(combined)
		model = keras.Model(inputs=modelinputs, outputs=w)
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
					  loss=keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
		return model

def create_model(inp_shapes, num_classes, channels, mc=False, tuning=False):
	m = Model(inp_shapes, num_classes, channels, mc)
	if not tuning:
		m.main_model()
		return m.model 
	else:
		return m.model_for_tuner
