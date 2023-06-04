from keras.models import load_model
from util import *
filename="./output/char2encoding.pkl"
sentence="I love deep learning"

input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index= getChar2encoding(filename)

encoder_input_data=encodingSentenceToPredict(sentence,input_token_index,max_encoder_seq_length,num_encoder_tokens) 
encoder_model= load_model('encoder_modelPredTranslation.h5')
decoder_model= load_model('decoder_modelPredTranslation.h5')

input_seq = encoder_input_data

decoded_sentence=decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index)
print('-')
print('Input sentence:', sentence)
print('Decoded sentence:', decoded_sentence)


# BLEU SCORE
def get_blue_score(filename):
    with open(str(filename), 'r') as f3:
        lines = f.readlines()
    for line in lines:
        element = line.strip().split('\t')
        y_true.append(element[0])
        y_pred.append(element[1])

    for y_true, y_pred in zip(y_true, y_pred):
        score += int(sentence_bleu(y_true, y_pred))
    return score
