from nltk.translate.bleu_score import sentence_bleu
from keras.models import load_model
from util import *
import random
filename="./output/char2encoding.pkl"

with open('spa-eng/spa.txt') as f:
    lines=f.readlines()
random.shuffle(lines)

test= lines[:30]
with open('DECODED.txt', 'w') as f2:
    for i in test:
        sentence=str(i).split('\t')[0]
        print(sentence)

        #num_encoder_tokens 91 77
        #saveChar2encoding("char2encoding.pkl",input_token_index,16,71,reverse_target_char_index,num_decoder_tokens,target_token_index)
        input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index= getChar2encoding(filename)

        encoder_input_data=encodingSentenceToPredict(sentence,input_token_index,max_encoder_seq_length,num_encoder_tokens) #91
        encoder_model= load_model('encoder_modelPredTranslation.h5')
        decoder_model= load_model('decoder_modelPredTranslation.h5')

        input_seq = encoder_input_data

        decoded_sentence=decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index)
        # f2.write('-')
        # f2.write(str(sentence))
        # f2.write(str(decoded_sentence))
        # f2.write("\n")

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