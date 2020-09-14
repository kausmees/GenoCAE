import tensorflow as tf

class ResidualBlock1(tf.keras.layers.Layer):
	'''
	Define a residual block with pre-activations with batch normalization.

	:param filters:
	:param kernel_size:
	:return: list of layers.py added
	'''

	def __init__(self, filters, kernel_size, name="res_block1"):
		super(ResidualBlock1, self).__init__()
		self.cname=name
		layers = []

		layers.append(tf.keras.layers.BatchNormalization())
		print("--- batch normalization")

		layers.append(tf.keras.layers.Activation('elu'))
		print("--- elu")

		layers.append(tf.keras.layers.Conv1D(filters, kernel_size, activation=None, padding='same'))
		print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))


		layers.append(tf.keras.layers.BatchNormalization())
		print("--- batch normalization")

		layers.append(tf.keras.layers.Activation('elu'))
		print("--- elu")

		layers.append(tf.keras.layers.Conv1D(filters, kernel_size, activation=None, padding='same'))
		print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

		self.layers = layers


	def call(self, input_data):
		'''
		Call a residual block.

		:param residual_block: list of layers.py in the block
		:return: output tensor

		'''

		# print("--- adding {0} ".format(type(self.layers[0])))
		x = self.layers[0](input_data)

		for layer in self.layers[1:]:
			# print("--- adding {0} ".format(type(layer)))
			x = layer(x)


		# print("--- performing addition ")
		x = tf.keras.layers.Add()([x, input_data])


		return x



class ResidualBlock2(tf.keras.layers.Layer):
	'''
	Define a residual block with conv act bn.


	:param filters:
	:param kernel_size:
	:return: list of layers.py added
	'''

	def __init__(self, filters, kernel_size, name="res_block1"):
		super(ResidualBlock2, self).__init__()
		self.cname=name
		layers = []

		layers.append(tf.keras.layers.Conv1D(filters, kernel_size, activation="elu", padding='same'))
		print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

		layers.append(tf.keras.layers.BatchNormalization())
		print("--- batch normalization")


		layers.append(tf.keras.layers.Conv1D(filters, kernel_size, activation="elu", padding='same'))
		print("--- conv1d  filters: {0} kernel_size: {1}".format(filters, kernel_size))

		layers.append(tf.keras.layers.BatchNormalization())
		print("--- batch normalization")

		self.layers = layers


	def call(self, input_data):
		'''
		Call a residual block.

		:param residual_block: list of layers.py in the block
		:return: output tensor

		'''

		# print("--- adding {0} ".format(type(self.layers[0])))
		x = self.layers[0](input_data)

		for layer in self.layers[1:]:
			# print("--- adding {0} ".format(type(layer)))
			x = layer(x)


		# print("--- performing addition ")
		x = tf.keras.layers.Add()([x, input_data])


		return x
