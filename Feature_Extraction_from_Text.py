import pandas as pd

with open('E:\\Download\\Resource\\One.txt') as mytest:
    word_ones=mytest.read().lower().split()
    uni_word_one=set(word_ones)

# print(uni_word_one)

with open('E:\\Download\\Resource\\Two.txt') as test:
    word_two=test.read().lower().split()
    uni_word_two=set(word_two)

# print(uni_word_two)

all_uni_words=set()
all_uni_words.update(uni_word_one)
# print(all_uni_words)

all_uni_words.update(uni_word_two)
# print(all_uni_words)

full_vocab=dict()
i=0

for word in all_uni_words:
    full_vocab[word]=i
    i=i+1

# print(full_vocab)

one_freq=[0]*len(full_vocab)
two_freq=[0]*len(full_vocab)
all_words=['']*len(full_vocab)


with open('E:\\Download\\Resource\\One.txt') as f:
    one_text=f.read().lower().split()
# print(one_text)

for word in one_text:
    word_index=full_vocab[word]
    one_freq[word_index] +=1
print(one_freq)

with open('E:\\Download\\Resource\\One.txt') as f:
    two_text=f.read().lower().split()

for word in two_text:
    word_index=full_vocab[word]
    two_freq[word_index] +=1
print(two_freq)

for word in full_vocab:
    word_index=full_vocab[word]
    all_words[word_index]=word
print(all_words)

bow=pd.DataFrame(data=[one_freq,two_freq],columns=all_words)
print(bow)