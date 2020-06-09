## Table of Contents
* [Natural Language Processing (NLP)](#natural-language-processing-nlp)
* [Classical vs DNLP](#classical-vs-dnlp)
* [Bag-of-Words Models](#bag-of-words-models)
* [Seq2Seq](#seq2seq)
* [Greedy Decoding](#greedy-decoding)
* [Beam Search Decoding](#beam-search-decoding)
* [Attention Mechanisms](#attention-mechanisms)

## Natural Language Processing (NLP)
Natural Language Processing consists of teaching machines to understand what is said in spoken and written words. For example: when you dictate something into your iPhone/Android device, that is then converted to text – this is done by using an NLP algorithm.

NLP can be used in:
* Speech Transcription
* Neural Machine Translation (NMT)
* Chatbots
* Q&A
* Text Summarization
* Image Captioning
* Video Captioning

![NLP Diagram](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/nlp/nlp-diagram.png)

## Classical vs DNLP
Classic model examples:
* If / Else Rules (old Chatbot method)
   * Someone asks a question, we give them a response
   * All answers are predetermined
* Audio frequency components analysis (Speech Recognition)
   * Done by looking at sound waves via the frequency & time
   * We compare these to prerecorded words
* Bag-of-words model (Classification)
   * For example: you have a list of comments, some are positive and some are negative
   * All of these words are put into a bag. The model calculates how often the word great (for example) comes up with a 1 
     and pool with a 0 (1 being positive, 0 being negative)

Deep Natural Language Processing example:

A convolutional neural network for text recognition (Classification)
* Words are processed through embedding of words into a matrix
* These words are the passed through the same process as how images are processed within a convolutional neural network

![Deep Natural Language Processing Example](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/nlp/dnlp-example.png)

Seq2Seq model example:

Seq2Seq (many applications)

![Seq2Seq Example](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/nlp/seq2seq-example.png)

## Bag-of-Words Models
Bag-of-Words is a vector of 0s for each word within a sentence. Every word has a position within this vector and when it is within a chosen sentence, +1 is added to that position within the vector. There are a total of 3 vectors that are placeholders for additional elements, for example:
* SOS - Start of sentence
* EOS - End of sentence
* Special Words - Every other word that is not within the vector of our chosen words

![Bag-of-Words Vector](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/nlp/bag-of-words-vector.png)

In the image example, we want to come up with a reply to the 'Checking if you are back to Oz'. The response to this needs to be a yes or a no. Here is an image with some example training data:

![Bag-of-Words Training Data](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/nlp/bow-training-data.png)

Using this data, we would convert the sentences into vectors. These vectors would be the same size and consist of the same words/format. We use the training data on a model, a Logistic Regression is a common model used for this. This would then provide us with information on the training data. Then a new question would be supplied as test data which then provides us with a yes or no response.

Alternatively, we could use a neural network to identify the answer to this question, this would be deep natural language processing. Some issues that consist with the Bag-of-Words model:
* Fixed-sized input
* Doesn't take word order into account - sentences rely on the word order to understand the context
* Fixed-sized output

## Seq2Seq
Seq2Seq models use recurrent neural networks. 

![Seq2Seq Examples](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/nlp/seq2seq-examples.png)

Each box is a whole layer of neurons. Using our example from the bag-of-words model, we are going to create a many to many model.

![Many to Many Vector](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/nlp/many-to-many-vector.png)

Within our 20,000 elements long vector, we take the position of each word and create a new vector of those positions. If the words in our sentence are not within our vector, we mark the position of it as 0. SOS will always be 1 and EOS will always be a 2. Seq2Seq consists of 2 recurrent neural networks, one is called the encoder and one is called the decoder.

The values are fed into our recurrent neural network and when the sentence reaches the end, it starts providing a predicted output. For the total number of values in our original vector, we would get that total number as possible predictions (in this case it would be 20k per word).

![DNLP Layout](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/nlp/dnlp-layout.png)

The words within the vector are assigned probabilities based on the sentence to determine which words fit the reply best. The words with the highest probability are output. Using an LSTM network, we feed the outputted value back into the network so that it can continue on with creating the sentence. The EOS cell stores all information from the encoded sentence and then decodes a reply to that sentence.

A more complex version would look similar to the below:

![DNLP Complex Layout](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/nlp/dnlp-complex-layout.png)

Through the process of backpropagation and stochastic gradient descent, each part of the recurrent neural network has certain weights and attributes defined to it. These attributes and weights are grouped so that the process is reiterated correctly, to ensure that the network is performing as intended and to save computation power.

![DNLP Response](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/nlp/dnlp-response.png)

## Greedy Decoding
Greedy Decoding means that the highest probability word is only looked at and is chosen to be the reply word & next word in the reply sentence.

## Beam Search Decoding
Beam Search Decoding is when you define a number of words to look at to determine which word is best for the reply to the sentence. For example: the 5 highest probability words. Each word we choose for the reply of the sentence will at the same number of highest probability words to complete the sentence. 

In order to pick the winner, we choose the highest probability of the joint beam. Making it the best fit for our reply.

![Beam Search Responses](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/nlp/beam-responses.png)

It is possible to truncate a beam, limiting the amount of words a beam/sentence can extend to.

## Attention Mechanisms
Attention Mechanisms are when the decoder is able to access any of the information from the encoder, instead of just relying on the EOS layer. The decoder is trained to assign unique weights to each word of the encoder, the higher the weight the more relevance it has to the next word in the sentence. 

The weights within the encoder are then calculated together to make a weighted sum called the context vector. This is then fed into the new layer of the decoder to help provide a new word as the output.

![Attention Mechanism](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/nlp/attention-mechanism.png)

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/natural_language_processing/chatbot.py) for an example of a chatbot using tensorflow.

```python
############### PART 2 - BUILDING THE SEQ2SEQ MODEL ###############
            

# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, lr, keep_prob

# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1]) # batch_size being the number of lines
    preprocessed_targets = tf.concat([left_side, right_side], 1) # 1 = horizontal; 2 = vertical
    return preprocessed_targets

# Creating the Encoder RNN Layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob) # Deactivates neurons during training
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    # Outputs - encoder_output, encoder_state; we only need the encoder_state
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, 
                                                       cell_bw=encoder_cell,
                                                       sequence_length=sequence_length,
                                                       inputs=rnn_inputs,
                                                       dtype=tf.float32)
    return encoder_state

# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option='bahdanau', num_units=decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name='attn_dec_train')
    # Outputs - decoder_output, decoder_final_state, decoder_final_context_state; we only need the decoder_output
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                  training_decoder_function,
                                                                  decoder_embedded_input,
                                                                  sequence_length,
                                                                  scope=decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option='bahdanau', num_units=decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name='attn_dec_inf')
    # Outputs - test_predictions, decoder_final_state, decoder_final_context_state; we only need the test_predictions
    test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                    test_decoder_function,
                                                                    scope=decoding_scope)
    return test_predictions

# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob) # Deactivates neurons during training
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x, num_words, None, scope=decoding_scope, weights_initializer=weights, biases_initializer=biases)
        training_predictions = decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, word2int['<SOS>'], word2int['<EOS>'], sequence_length - 1, num_words, decoding_scope, output_function, keep_prob, batch_size)
        return training_predictions, test_predictions

# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, answers_num_words + 1, encoder_embedding_size, initializer=tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, questions_num_words, sequence_length, rnn_size, num_layers, questionswords2int, keep_prob, batch_size)
    return training_predictions, test_predictions
```