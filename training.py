from util import *

batch_inici = 0
batch_final = num_samples
data_path = './spa-eng/spa.txt' #139705
lines = open(data_path).read().split('\n')
batches = ((len(lines)-1)//num_samples)+1 #afegim un batch més per als que sobren
print('bat', batches)

for i in range (batches):
    print('ENTRAAAAAAAA')
    print('goo',batch_inici,batch_final)
    print('GOO')
    print('num_sam',num_samples)
    #load the data and format  them for being processed
    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length=prepareData(data_path, batch_inici, batch_final)

    # we build the model
    model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense=modelTranslation(num_encoder_tokens,num_decoder_tokens)

    # we train it
    trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data)

    # we build the final model for the inference (slightly different) and we save it
    encoder_model,decoder_model,reverse_target_char_index=generateInferenceModel(encoder_inputs, encoder_states,input_token_index,target_token_index,decoder_lstm,decoder_inputs,decoder_dense)

    batch_inici += num_samples
    batch_final += num_samples

# we save the object to convert the sequence to encoding  and encoding to sequence
# our model is made for being used with different langages that do not have the same number of letters and the same alphabet
saveChar2encoding("./output/char2encoding.pkl",input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index)

