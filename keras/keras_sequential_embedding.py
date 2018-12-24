import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

# Define documents
docs = ['Well done!', 'Good work', 'Great effort', 'nice work', 'Excellent!',
        'Weak', 'Poor effort!', 'not good', 'poor work', 'Could have done better.']

# Define class labels
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

own_embedding_vocab_size = 10
encoded_docs_oe = [one_hot(d, own_embedding_vocab_size) for d in docs]
print(encoded_docs_oe)
maxlen = 5
padded_docs_oe = pad_sequences(encoded_docs_oe, maxlen=maxlen, padding='post')
print(padded_docs_oe)

model = Sequential()
model.add(Embedding(input_dim=own_embedding_vocab_size, output_dim=32, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
model.fit(padded_docs_oe, labels, epochs=50, verbose=0)
loss, accuracy = model.evaluate(padded_docs_oe, labels, verbose=0)
print('accuracy: %0.3f' % accuracy)

testdocs = 'nice effort'
encoded_testdocs = [one_hot(testdocs, own_embedding_vocab_size)]
print(encoded_testdocs)
padded_testdocs = pad_sequences(encoded_testdocs, maxlen=maxlen, padding='post')
print(padded_testdocs)
result = model.predict(padded_testdocs)
print('%s = ' % testdocs, result)
