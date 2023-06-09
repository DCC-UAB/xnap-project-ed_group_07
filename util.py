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

batch_size = 128  # Batch size for training.

epochs = 25  # Number of epochs to train for.

latent_dim = 256 #1024 
 # Latent dimensionality of the encoding space.
 
num_samples =  90000 # Number of samples to train on.

# Path to the data txt file on disk.
# './cat-eng/cat.txt' el dataset en catala nomes te 1336 linies
data_path = './spa-eng/spa.txt' #139705 lines
encoder_path='encoder_modelPredTranslation.h5'
decoder_path='decoder_modelPredTranslation.h5'
validation_split = 0.2
LOG_PATH='./log' 
learning_rate = 0.0001

name = "Execution"
opt = 'rmsprop' #'adam'

# start a new wandb run to track this script
config_defaults = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "dataset": data_path,
    "epochs": epochs,
    "latent_dim": latent_dim,
    "cell_type": 'LSTM', #'GRU'
    "optimizer":  opt,
    "lstm_layers": 1, 
    'dropouts': 0
    }

wandb.init(
    # set the wandb project where this run will be logged
    project="Translation",
    # track hyperparameters and run metadata
    config=config_defaults,
    name = name,
    allow_val_change=True
)

def prepareData(data_path):

    input_characters,target_characters,input_texts,target_texts=extractChar(data_path)

    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length =encodingChar(input_characters,target_characters,input_texts,target_texts)
        
    encoder_dataset, decoder_input_dataset, decoder_target_dataset  = create_data_loader(encoder_input_data, decoder_input_data, decoder_target_data)
    
    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length, encoder_dataset, decoder_input_dataset, decoder_target_dataset

def create_data_loader(encoder_input_data, decoder_input_data, decoder_target_data):
    # Create TensorFlow datasets from the encoded data arrays
    encoder_dataset = tf.data.Dataset.from_tensor_slices(encoder_input_data)
    decoder_input_dataset = tf.data.Dataset.from_tensor_slices(decoder_input_data)
    decoder_target_dataset = tf.data.Dataset.from_tensor_slices(decoder_target_data)
        
    return encoder_dataset, decoder_input_dataset, decoder_target_dataset 

def extractChar(data_path, exchangeLanguage=False):
    # We extract the data (Sentence1 \t Sentence 2) from the anki text file
    input_texts = [] 
    target_texts = []
    input_characters = set()
    target_characters = set()
    lines = open(data_path).read().split('\n')

    if (exchangeLanguage==False):

        for line in lines[: min(num_samples, len(lines) - 1)]: 
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

    
def modelTranslation(num_encoder_tokens,num_decoder_tokens):
# We crete the model 1 encoder(lstm)/1 encoder(gru) + 1 decode (LSTM)/1 decode (gru) + 1 Dense layer + softmax

    if wandb.config.cell_type == 'LSTM':
    
        encoder_inputs = Input(shape=(None, num_encoder_tokens)) 
        # input tensor to the encoder. It has a shape of (None, num_encoder_tokens), where None represents the variable-length 
        # sequence and num_encoder_tokens is the number of tokens in the input lenguage.
        encoder = LSTM(wandb.config.latent_dim, return_state=True, dropout=wandb.config.dropouts)
        # latent_dim: Latent dimensionality of the encoding space.
        # encoder LSTM layer is created with latent_dim units
        encoder_outputs, state_h, state_c = encoder(encoder_inputs) 
        # encoder_outputs: output sequence from the encoder LSTM layer
        # state_h and state_c: final hidden state and cell state of the encoder.
        encoder_states = [state_h, state_c]
        # will be used as the initial state for the decoder
        
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_lstm = LSTM(wandb.config.latent_dim, return_sequences=True, return_state=True, dropout=wandb.config.dropouts)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                                initial_state=encoder_states)

        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        

        return model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense

    
    elif wandb.config.cell_type =='GRU':
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = GRU(wandb.config.latent_dim, return_state=True)
        encoder_outputs, state_h = encoder(encoder_inputs)
        encoder_states = state_h

        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_gru = GRU(wandb.config.latent_dim, return_sequences=True)
        decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
        return model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_gru,decoder_dense

def trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data, encoder_dataset, decoder_input_dataset, decoder_target_dataset):
# We load tensorboad
# We train the model
    LOG_PATH="./output/log"
        
    tbCallBack = TensorBoard(log_dir=LOG_PATH, histogram_freq=0, write_graph=True, write_images=True)
    
    # Run training

    model.compile(optimizer=wandb.config.optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    #categorical_crossentropy:  loss between the true classes and predicted classes. The labels are given in an one_hot format.

    train_dataset = tf.data.Dataset.zip((encoder_dataset, decoder_input_dataset))
    train_dataset = tf.data.Dataset.zip((train_dataset,  decoder_target_dataset))
    train_dataset = train_dataset.batch(wandb.config.batch_size)

    validation_dataset = train_dataset.take(int(validation_split * len(train_dataset)))
    train_dataset = train_dataset.skip(int(validation_split * len(train_dataset)))

    model.fit(train_dataset, batch_size=wandb.config.batch_size, epochs=wandb.config.epochs, validation_data=validation_dataset, callbacks=[WandbCallback()])
    
    # model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
    #             batch_size=batch_size,
    #             epochs=epochs,
    #             validation_split=0.01,
    #             callbacks = [tbCallBack])
    
    
    # Evaluate    
    #loss, accuracy = model.evaluate(validation_dataset, callbacks=[WandbCallback()])
    loss, acc = model.evaluate(validation_dataset)
    wandb.log({'evaluate/accuracy': acc})
    

def generateInferenceModel(encoder_inputs, encoder_states,input_token_index,target_token_index,decoder_lstm,decoder_inputs,decoder_dense):
# Once the model is trained, we connect the encoder/decoder and we create a new model
# Finally we save everything
    if wandb.config.cell_type == 'LSTM':
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(wandb.config.latent_dim,))
        decoder_state_input_c = Input(shape=(wandb.config.latent_dim,))
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
    
    elif wandb.config.cell_type == 'GRU':
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(wandb.config.latent_dim,))
        #decoder_state_input_c = Input(shape=(wandb.config.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h]
        #decoder_outputs, state_h = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_outputs, state_h = GRU(wandb.config.latent_dim, return_sequences=True, return_state=True)(decoder_inputs, initial_state=decoder_states_inputs[0])
        decoder_states = [state_h]
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

def decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index):
# We run the model and predict the translated sentence
    if wandb.config.cell_type == 'LSTM':
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
            
    elif wandb.config.cell_type == 'GRU':
    
        # We encode the input
        states_value = encoder_model.predict(input_seq)

        
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        
        target_seq[0, 0, target_token_index['\t']] = 1.

        stop_condition = False
        decoded_sentence = ''
        # We predict the output letter by letter 
        while not stop_condition:
            output_tokens, states_value = decoder_model.predict(
                [target_seq] + [states_value])

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






