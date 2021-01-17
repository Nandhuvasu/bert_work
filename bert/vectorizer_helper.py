import numpy as np

# pool embeds - (1,768)
def reduce_mean(vector):
    sentence_vector = np.mean(vector[0], axis=0)
    print(sentence_vector)
    print(sentence_vector.shape)
    return sentence_vector