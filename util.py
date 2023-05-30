from __future__ import print_function
import keras 
from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
from keras.models import load_model
from keras.callbacks import TensorBoard
import numpy as np
import _pickle as pickle
import wandb
from wandb.keras import WandbCallback
import random
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import numpy as np

batch_size = 128  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 1024#256  # Latent dimensionality of the encoding space.
num_samples =  90000 #145437  # Number of samples to train on.
# Path to the data txt file on disk.
#data_path = './cat-eng/cat.txt' # to replace by the actual dataset name
# el dataset en catala nomes te 1336 linies
data_path = './spa-eng/spa.txt' #139705
encoder_path='encoder_modelPredTranslation.h5'
decoder_path='decoder_modelPredTranslation.h5'
validation_split = 0.1
#LOG_PATH='/home/alumne/projecte/xnap-project-ed_group_07/log' #quan estem en remot
LOG_PATH='./log' #quan estem en local

validation_split = 0.1
learning_rate = 0.02
name = "spanish acc grafica"

# # DEALING WITH BLEU METRIC FUNCTION
# from nltk.translate.bleu_score import sentence_bleu

# def BLEU(y_true, y_pred):
#     # Convert y_true and y_pred to integer tensors
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)

#     # Calculate the maximum n-gram order
#     max_order = 4

#     # Initialize variables to store the n-gram counts and the brevity penalty
#     ngram_counts = [0] * max_order
#     brevity_penalty = 0

#     # Get the batch size
#     batch_size = tf.shape(y_true)[0]

#     # Iterate over the batch dimension
#     for i in range(batch_size):
#         # Calculate the length of the reference and hypothesis sentences
#         ref_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true[i], 0), tf.float32))
#         hyp_length = tf.reduce_sum(tf.cast(tf.not_equal(y_pred[i], 0), tf.float32))

#         # Update the brevity penalty
#         brevity_penalty += tf.math.log(tf.minimum(tf.cast(1.0, tf.float32), tf.cast(hyp_length / ref_length, tf.float32)))

#         # Iterate over the n-gram orders
#         for n in range(1, max_order + 1):
#             # Calculate the n-grams for the reference and hypothesis sentences
#             ref_ngrams = [tuple(y_true[i, j:j+n].numpy()) for j in range(ref_length - n + 1)]
#             hyp_ngrams = [tuple(y_pred[i, j:j+n].numpy()) for j in range(hyp_length - n + 1)]

#             # Count the number of matching n-grams
#             matches = sum(1 for ng in hyp_ngrams if ng in ref_ngrams)

#             # Update the n-gram counts
#             ngram_counts[n-1] += matches / len(hyp_ngrams)

#     # Calculate the final BLEU score
#     bleu_score = tf.exp(tf.cast(brevity_penalty / batch_size, tf.float32) + tf.reduce_sum([tf.math.log(count) for count in ngram_counts]) / max_order)

#     return bleu_score

#API

#GRÀFIQUES
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Translation",
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "LSTM",
    "dataset": data_path,
    "epochs": epochs,
    },
    name = name
)



def prepareData(data_path, batch_inici, batch_final):

    input_characters,target_characters,input_texts,target_texts=extractChar(data_path, batch_inici, batch_final)
    #es bidireccional per tant li passem com a 4 argument True si volem que faci la traduccio al reves
    

    # with open('./spa-eng/train/source/source.txt', 'w') as f_source:
    #     with open('./spa-eng/train/target/target.txt', 'w') as f_target:
    #         for  input, target in zip(input_texts, target_texts):
    #             f_source.write(input + '\n')
    #             f_target.write(target + '\n')


    # path_loader = './spa-eng/train'

    # dataloader = tf.keras.preprocessing.text_dataset_from_directory(
    #     path_loader,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     seed=None,
    #     validation_split=None,
    #     subset=None,
    # )
    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length =encodingChar(input_characters,target_characters,input_texts,target_texts)
    #encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length =encodingChar(dataloader, input_characters, target_characters, input_texts,target_texts)
    
    # executar si no s'ha fet el dataloader encara
    # create_data_loader(encoder_input_data, decoder_input_data, decoder_target_data, batch_size)
    encoder_dataset = np.load('ENCODED.npy')
    encoder_dataset = tf.data.Dataset.from_tensor_slices(encoder_dataset)

    decoder_input_dataset = np.load('INPUT.npy')
    decoder_input_dataset = tf.data.Dataset.from_tensor_slices(decoder_input_dataset)

    decoder_target_dataset = np.load('TARGET.npy')
    decoder_target_dataset = tf.data.Dataset.from_tensor_slices(decoder_target_dataset)

    
    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length, encoder_dataset, decoder_input_dataset, decoder_target_dataset

def create_data_loader(encoder_input_data, decoder_input_data, decoder_target_data, batch_size):
    # Create TensorFlow datasets from the encoded data arrays
    encoder_dataset = tf.data.Dataset.from_tensor_slices(encoder_input_data)
    encoder_dataset = np.array(list(encoder_dataset.as_numpy_iterator()))
    np.save('ENCODED.npy', encoder_dataset)

    decoder_input_dataset = tf.data.Dataset.from_tensor_slices(decoder_input_data)
    decoder_input_dataset = np.array(list(decoder_input_dataset.as_numpy_iterator()))
    np.save('INPUT.npy', decoder_input_dataset)

    decoder_target_dataset = tf.data.Dataset.from_tensor_slices(decoder_target_data)
    decoder_target_dataset = np.array(list(decoder_target_dataset.as_numpy_iterator()))
    np.save('TARGET.npy', decoder_target_dataset)
    
    # Combine the datasets into a single dataset
    
    #dataset_input = tf.data.Dataset.zip((encoder_dataset, decoder_input_dataset))
    #dataset_target = tf.data.Dataset.zip((decoder_target_dataset))

    # Shuffle and batch the dataset
    #dataset = dataset.shuffle(len(encoder_input_data))
    
    #dataset_input = dataset_input.batch(batch_size)
    #dataset_target = dataset_target.batch(batch_size)


    # decoder_input_dataset.save('DECODEDINPUT.h5')
    # decoder_target_dataset.save('DECODEDTARGET.h5')

    #return dataset_input, dataset_target
    # return decoder_input_dataset, decoder_target_dataset 


def extractChar(data_path, batch_inici, batch_final, exchangeLanguage=False):
    # We extract the data (Sentence1 \t Sentence 2) from the anki text file
    input_texts = [] 
    target_texts = []
    input_characters = set()
    target_characters = set()
    lines = open(data_path).read().split('\n')

    if (exchangeLanguage==False):

        for line in lines[: min(num_samples, len(lines) - 1)]: #change
        #batch_final=min(batch_final, len(lines) - 1)
        #for line in lines[batch_inici: batch_final]:
            input_text, target_text, _ = line.split('\t')
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))

    else:
        for line in lines[: min(num_samples, len(lines) - 1)]:
        
        #batch_final=min(batch_final, len(lines) - 1)
        #for line in lines[batch_inici: batch_final]:

            target_text , input_text, _ = line.split('\t')
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))

    return input_characters,target_characters,input_texts,target_texts
    
    
def encodingChar(input_characters,target_characters,input_texts,target_texts):
# We encode the dataset in a format that can be used by our Seq2Seq model (hot encoding).
# Important: this project can be used for different language that do not have the same number of letter in their alphabet.
# Important2: the decoder_target_data is ahead of decoder_input_data by one timestep (decoder = LSTM cell).
# 1. We get the number of letter in language 1 and 2 (num_encoder_tokens/num_decoder_tokens)
# 2. We create a dictonary for both language
# 3. We store their encoding and return them and their respective dictonary
    
    #if num_encoder_tokens < 
    num_encoder_tokens = len(input_characters) #numero de lletres diferents llengua entrada
    num_decoder_tokens = len(target_characters) #numero de lletres diferents llengua sortida
    max_encoder_seq_length = max([len(txt) for txt in input_texts]) #max len d'una linia entrada
    max_decoder_seq_length = max([len(txt) for txt in target_texts]) #max len d'una linia sortida
    print('Number of num_encoder_tokens:', num_encoder_tokens)
    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)]) # {"a": 0, "b": 1, "?": 2}
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length
    
# def encodingChar(dataset, input_characters, target_characters, input_texts,target_texts):
#     # Extract input and target texts from the dataset
#     input_texts = []
#     target_texts = []
#     for inputs, targets in dataset:
#         input_texts.extend(inputs)
#         target_texts.extend(targets)
    
#     # Get the number of unique characters in the input and target languages
#     num_encoder_tokens = len(input_characters)
#     num_decoder_tokens = len(target_characters)
    
#     # Determine the maximum sequence lengths for inputs and outputs
#     max_encoder_seq_length = max([len(txt) for txt in input_texts])
#     max_decoder_seq_length = max([len(txt) for txt in target_texts])
    
#     print('Number of num_encoder_tokens:', num_encoder_tokens)
#     print('Number of samples:', len(input_texts))
#     print('Number of unique input tokens:', num_encoder_tokens)
#     print('Number of unique output tokens:', num_decoder_tokens)
#     print('Max sequence length for inputs:', max_encoder_seq_length)
#     print('Max sequence length for outputs:', max_decoder_seq_length)
    
#     # Create dictionaries for encoding characters
#     input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
#     target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    
#     # Initialize the input, target, and decoder target data arrays
#     encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
#     decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
#     decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

#     for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
#         for t, char in enumerate(input_text):
#             encoder_input_data[i, t, input_token_index[char]] = 1.
#         for t, char in enumerate(target_text):
#             decoder_input_data[i, t, target_token_index[char]] = 1.
#             if t > 0:
#                 decoder_target_data[i, t - 1, target_token_index[char]] = 1.

#     return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length



def modelTranslation2(num_encoder_tokens,num_decoder_tokens):
# We crete the model 1 encoder(gru) + 1 decode (gru) + 1 Dense layer + softmax

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = GRU(latent_dim, return_state=True)
    encoder_outputs, state_h = encoder(encoder_inputs)
    encoder_states = state_h

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_gru = GRU(latent_dim, return_sequences=True)
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    return model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_gru,decoder_dense
    
def modelTranslation(num_encoder_tokens,num_decoder_tokens):
# We crete the model 1 encoder(lstm) + 1 decode (LSTM) + 1 Dense layer + softmax
    
    encoder_inputs = Input(shape=(None, num_encoder_tokens)) 
    # input tensor to the encoder. It has a shape of (None, num_encoder_tokens), where None represents the variable-length 
    # sequence and num_encoder_tokens is the number of tokens in the input lenguage.
    encoder = LSTM(latent_dim, return_state=True) 
    # latent_dim: Latent dimensionality of the encoding space.
    # encoder LSTM layer is created with latent_dim units
    encoder_outputs, state_h, state_c = encoder(encoder_inputs) 
    # encoder_outputs: output sequence from the encoder LSTM layer
    # state_h and state_c: final hidden state and cell state of the encoder.
    encoder_states = [state_h, state_c]
    # will be used as the initial state for the decoder
    
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                            initial_state=encoder_states)

    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense

#def trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data, dataloader_encoded_input, dataloader_encoded_target):
def trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data, encoder_dataset, decoder_input_dataset, decoder_target_dataset):
# We load tensorboad
# We train the model
    LOG_PATH="./log"
        
    tbCallBack = TensorBoard(log_dir=LOG_PATH, histogram_freq=0, write_graph=True, write_images=True)
    # Run training
    #CANVIAR 'rmsprop' per adam model.compile(optimizer="Adam", loss="mse", metrics=["mae"]) mse no te sentit
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)
    #optimizer = 'rmsprop'

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[BLEU])
    #categorical_crossentropy:  loss between the true classes and predicted classes. The labels are given in an one_hot format.
      
    # model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
    #             batch_size=batch_size,
    #             epochs=epochs,
    #             validation_split=0.01,
    #             callbacks = [tbCallBack])


    train_dataset = tf.data.Dataset.zip((encoder_dataset, decoder_input_dataset))
    train_dataset = tf.data.Dataset.zip((train_dataset,  decoder_target_dataset))
    train_dataset = train_dataset.batch(batch_size)

    validation_dataset = train_dataset.take(int(validation_split * len(train_dataset)))
    train_dataset = train_dataset.skip(int(validation_split * len(train_dataset)))

    #history = model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=validation_dataset, callbacks=[tbCallBack])
    history = model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=validation_dataset, callbacks=[WandbCallback()])
    
    #Retrieve loss and accuracy from the history object    
    loss = history.history['loss']
    accuracy = history.history['accuracy']
    #loss, accuracy = model.evaluate(validation_dataset, callbacks=[tbCallBack])

    # log metrics to wandb
    wandb.log({"accuracy": accuracy, "loss": loss})
    

def generateInferenceModel(encoder_inputs, encoder_states,input_token_index,target_token_index,decoder_lstm,decoder_inputs,decoder_dense):
# Once the model is trained, we connect the encoder/decoder and we create a new model
# Finally we save everything
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())
    encoder_model.save(encoder_path)
    decoder_model.save(decoder_path)
    return encoder_model,decoder_model,reverse_target_char_index

def loadEncoderDecoderModel():
# We load the encoder model and the decoder model and their respective weights
    encoder_model= load_model(encoder_path)
    decoder_model= load_model(decoder_path)
    return encoder_model,decoder_model

def decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index):\
# We run the model and predict the translated sentence

    # We encode the input
    states_value = encoder_model.predict(input_seq)

    
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    # We predict the output letter by letter 
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # We translate the token in hamain language
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # We check if it is the end of the string
        if (sampled_char == '\n' or
           len(decoded_sentence) > 500):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence

def encodingSentenceToPredict(sentence,input_token_index,max_encoder_seq_length,num_encoder_tokens):
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    for t, char in enumerate(sentence):
        encoder_input_data[0, t, input_token_index[char]] = 1.
    return encoder_input_data

def saveChar2encoding(filename,input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index):
    f = open(filename, "wb")
    pickle.dump(input_token_index, f)
    pickle.dump(max_encoder_seq_length, f)
    pickle.dump(num_encoder_tokens, f)
    pickle.dump(reverse_target_char_index, f)
    
    pickle.dump(num_decoder_tokens, f)
    
    pickle.dump(target_token_index, f)
    f.close()
    

def getChar2encoding(filename):
    f = open(filename, "rb")
    input_token_index = pickle.load(f)
    max_encoder_seq_length = pickle.load(f)
    num_encoder_tokens = pickle.load(f)
    reverse_target_char_index = pickle.load(f)
    num_decoder_tokens = pickle.load(f)
    target_token_index = pickle.load(f)
    f.close()
    return input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index

