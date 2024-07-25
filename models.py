import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU

class UNet(tf.keras.Model):
    
    def __init__(self, in_channels=None, out_channels=None, num_filters_per_layer=64, activation=None, dropout=0.5, leaky_relu_alpha=0.1, name="unet", **kwargs):
        super(UNet,self).__init__(name=name, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters_per_layer = num_filters_per_layer
        self.activation = activation
        self.dropout_rate = dropout
        self.leaky_relu_alpha = leaky_relu_alpha
        
        self.encoder_conv1 = Conv2D(num_filters_per_layer, kernel_size=3, padding='same')
        self.encoder_conv2 = Conv2D(num_filters_per_layer*2, kernel_size=3, padding='same')
        self.encoder_conv3 = Conv2D(num_filters_per_layer*4, kernel_size=3, padding='same')
        self.encoder_conv4 = Conv2D(num_filters_per_layer*8, kernel_size=3, padding='same')
        
        self.pool = MaxPooling2D(pool_size=(2, 2))

        self.decoder_conv1 = Conv2DTranspose(num_filters_per_layer * 8, kernel_size=2, strides=2, padding='same')
        self.decoder_conv2 = Conv2DTranspose(num_filters_per_layer * 4, kernel_size=2, strides=2, padding='same')
        self.decoder_conv3 = Conv2DTranspose(num_filters_per_layer * 2, kernel_size=2, strides=2, padding='same')
        self.decoder_conv4 = Conv2DTranspose(out_channels, kernel_size=2, strides=2, padding='same')
        self.final_conv = Conv2D(out_channels, kernel_size=1, padding='same', activation='sigmoid')
        self.dropout = Dropout(dropout)

        self.batchnorm_encoder1 = BatchNormalization(axis=-1)
        self.batchnorm_encoder2 = BatchNormalization(axis=-1)
        self.batchnorm_encoder3 = BatchNormalization(axis=-1)
        self.batchnorm_encoder4 = BatchNormalization(axis=-1)  

        self.batchnorm_decoder1 = BatchNormalization(axis=-1)
        self.batchnorm_decoder2 = BatchNormalization(axis=-1)
        self.batchnorm_decoder3 = BatchNormalization(axis=-1)
        self.batchnorm_decoder4 = BatchNormalization(axis=-1)     
        self.leaky_relu = LeakyReLU(alpha=leaky_relu_alpha)
    
    
    def call(self, x):
        x1 = self.encoder_conv1(x)
        x1 = self.batchnorm_encoder1(x1)
        x1 = self.leaky_relu(x1)
        x1 = self.dropout(x1)
        p1 = self.pool(x1)

        x2 = self.encoder_conv2(p1)
        x2 = self.batchnorm_encoder2(x2)
        x2 = self.leaky_relu(x2)
        x2 = self.dropout(x2)
        p2 = self.pool(x2)

        x3 = self.encoder_conv3(p2)
        x3 = self.batchnorm_encoder3(x3)
        x3 = self.leaky_relu(x3)
        x3 = self.dropout(x3)
        p3 = self.pool(x3)

        x4 = self.encoder_conv4(p3)
        x4 = self.batchnorm_encoder4(x4)
        x4 = self.leaky_relu(x4)
        x4 = self.dropout(x4)
        p4 = self.pool(x4)

        u1 = self.decoder_conv1(p4)
        u1 = tf.concat([u1, x4], axis=-1)
        u1 = self.batchnorm_decoder1(u1)
        u1 = self.leaky_relu(u1)
        u1 = self.dropout(u1)

        u2 = self.decoder_conv2(u1)
        u2 = tf.concat([u2, x3], axis=-1)
        u2 = self.batchnorm_decoder2(u2)
        u2 = self.leaky_relu(u2)
        u2 = self.dropout(u2)

        u3 = self.decoder_conv3(u2)
        u3 = tf.concat([u3, x2], axis=-1)
        u3 = self.batchnorm_decoder3(u3)
        u3 = self.leaky_relu(u3)
        u3 = self.dropout(u3)

        u4 = self.decoder_conv4(u3)
        u4 = tf.concat([u4, x1], axis=-1)
        u4 = self.batchnorm_decoder4(u4)
        u4 = self.leaky_relu(u4)
        u4 = self.dropout(u4)
        
        output = self.final_conv(u4)

        return output
            
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


    def get_config(self):
        config = super(UNet, self).get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'num_filters_per_layer': self.num_filters_per_layer,
            'activation': self.activation,
            'dropout': self.dropout_rate,
            'leaky_relu_alpha': self.leaky_relu_alpha,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)