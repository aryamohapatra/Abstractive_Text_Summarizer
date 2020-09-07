import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.LSTM = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    #output, state = self.gru(x, initial_state = hidden)
    output, state_h, state_c = self.LSTM(x, initial_state = hidden)
    #return output, state
    return output, state_h,state_c

  def initialize_hidden_state(self):
    return (tf.zeros([self.batch_sz, self.enc_units]),
                tf.zeros([self.batch_sz, self.enc_units]))
    #return tf.zeros((self.batch_sz, self.enc_units))
  
  
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    #print('We are here')
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    #print('let us calculate score')
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    #print('we are out of here')
    return context_vector, tf.squeeze(attention_weights,-1)
  
class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(units)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, decoder_hidden,encoder_output):
        # Dot score: h_t (dot) Wa (dot) h_s
        # encoder_output shape: (batch_size, max_len, rnn_size)
        # decoder_output shape: (batch_size, 1, rnn_size)
        # score will have shape: (batch_size, 1, max_len)
        hidden_with_time_axis = tf.expand_dims(decoder_hidden, 1)
        #score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)

        score = self.V(tf.nn.tanh(self.W1(encoder_output+hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        #alignment = tf.nn.softmax(score, axis=1)
        #print('shape of score L',score.shape)
    
        #print('shape of attention_weights L',alignment.shape)
        #context = tf.matmul(alignment, encoder_output)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        #print('--Out of this')
        return context_vector, tf.squeeze(attention_weights,-1)

class Decoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.attention = LuongAttention(dec_units)
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.LSTM = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)
    #self.wc = tf.keras.layers.Dense(rnn_size, activation='tanh')
  def call(self, x, hidden, enc_output,context_vector_):
      #print('we are inside decoder')
      x = self.embedding(x)
      lstm_out, state_h,state_c = self.LSTM(x,initial_state=hidden)
      #print('We got out put')
      context_vector, attn = self.attention(state_h, enc_output)

      x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
      #print('ok')
      #output, state_h,state_c = self.LSTM(x)
      #print('ok fine')
      lstm_out = tf.reshape(lstm_out, (-1, lstm_out.shape[2]))
      #lstm_out = self.wc(lstm_out)
      out = self.fc(lstm_out)
      #print('Out of this')
      return x, out, state_h,context_vector, attn

class Pointer(tf.keras.layers.Layer):
  
  def __init__(self):
    super(Pointer, self).__init__()
    self.w_s_reduce = tf.keras.layers.Dense(1)
    self.w_i_reduce = tf.keras.layers.Dense(1)
    self.w_c_reduce = tf.keras.layers.Dense(1)
    
  def call(self, context_vector, state, dec_inp):
    return tf.nn.sigmoid(self.w_s_reduce(state)+self.w_c_reduce(context_vector)+self.w_i_reduce(dec_inp))
    