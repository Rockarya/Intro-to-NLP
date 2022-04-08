import re
import string
import random
import math

total_tokens = 0
distinct_tokens = set()

# the value returned in random.uniform should be small enough, but preplexity increases as we decrease the probability 

def tokenize(file_name):
    # reading the tweet corpus file
    file = open(file_name,'r')
    # reading lines
    lines = file.readlines()

    clean_lines = []
    for i in range(len(lines)):
        text = lines[i]
        # substituting mentions
        text = re.sub(r"@\S*\s", ' MENTION ',text)
        # substituting URLs
        text = re.sub(r'[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%.\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%\+.~#?&//=]*)', ' URL ',text)
        # substituting hashtags
        text = re.sub(r"#\S*\s", ' HASHTAG ',text)

        # removing extra punctuations
        letter_list = []
        punctuation = string.punctuation
        for i in range(len(text)):
            if text[i] in punctuation and i-1>=0 and text[i]==text[i-1]:
                do_nothing = 1
            else:
                # here we are seperating the punctuation marks and words as different words
                if text[i] in punctuation:
                    letter_list.append(' ')   #apostrophe will automatically get handles with it
                    letter_list.append(text[i])
                else:
                    letter_list.append(text[i])


        text = "".join(letter_list)

        # lower case the text
        text = text.lower()

        text = text.split()
        clean_lines.append(text)

    return clean_lines

def write_cleaned_output(file_name):
    file = open(file_name,'w')
    for i in range(len(clean_lines)):
        text = ''
        for j in range(len(clean_lines[i])):
            text += clean_lines[i][j]
            text += ' '
        text += '\n'
        file.write(text)

# used 4-gram to calculate the counts
def calculate_count(clean_lines):
    global total_tokens

    # for one word, don't make tuple
    count = {}
    count['<s>'] = 0
    count[('<s>','<s>')] = 0
    count[('<s>','<s>','<s>')] = 0

    continuation_count = {}
    continuation_count['<s>'] = set()
    continuation_count['<s>'].add('<s>')
    continuation_count[('<s>','<s>')] = set()
    continuation_count[('<s>','<s>')].add('<s>')

    for i in range(len(clean_lines)):
        token_list = []
        for _ in range(3):
            token_list.append('<s>')

        l = len(clean_lines[i])
        total_tokens += l

        for j in range(l):
            token_list.append(clean_lines[i][j])

        for _ in range(3):
            token_list.append('</s>')

        count['<s>'] += 3
        count[('<s>','<s>')] += 2
        count[('<s>','<s>','<s>')] += 1

        for j in range(3,len(token_list)):
            w = token_list[j]
            w_i1 = token_list[j-1]
            w_i2 = token_list[j-2]
            w_i3 = token_list[j-3]

            if count.get(w) == None:
                count[w] = 0
            if count.get((w_i1,w)) == None:
                count[(w_i1,w)] = 0
            if count.get((w_i2,w_i1,w)) == None:
                count[(w_i2,w_i1,w)] = 0
            if count.get((w_i3,w_i2,w_i1,w)) == None:
                count[(w_i3,w_i2,w_i1,w)] = 0

            count[w] += 1
            count[(w_i1,w)] += 1
            count[(w_i2,w_i1,w)] += 1
            count[(w_i3,w_i2,w_i1,w)] += 1

            if continuation_count.get(w_i1) == None:
                continuation_count[w_i1] = set()
            if continuation_count.get((w_i2,w_i1)) == None:
                continuation_count[(w_i2,w_i1)] = set()
            if continuation_count.get((w_i3,w_i2,w_i1)) == None:
                continuation_count[(w_i3,w_i2,w_i1)] = set()

            continuation_count[w_i1].add(w)
            continuation_count[(w_i2,w_i1)].add(w)
            continuation_count[(w_i3,w_i2,w_i1)].add(w)

    return count, continuation_count


# here u should use value of n to be less than or equal to 4
def Kneyser_Ney(i,n,sentence):
    d = 0.75
    if n == 1:
        word = sentence[i]
        if word not in distinct_tokens:
            word = 'UNK'

        if count.get(word) != None:
            return count[word]/total_tokens
        else:
            return random.uniform(1e-2,1e-3)

    else:
        numerator_list = []
        denominator_list = []
        for j in range(i-n+1,i+1):
            word = sentence[j]
            if word not in distinct_tokens:
                word = 'UNK'

            numerator_list.append(word)

            # Wi won't be added
            if j != i:
                denominator_list.append(word)

        numerator_tuple = tuple(numerator_list)

        # numerator tuple will always have atleast 2 elements, because n is atleast 2 in else waala case
        if len(denominator_list) == 1:
            denominator_tuple = denominator_list[0]
        else:
            denominator_tuple = tuple(denominator_list)

        if count.get(denominator_tuple) != None:
            if count.get(numerator_tuple) == None:
                f1 = 0
            else:
                f1 = max(count[numerator_tuple]-d,0)/count[denominator_tuple]
            cont_cnt = len(continuation_count[denominator_tuple])
            f2 = (d*cont_cnt*Kneyser_Ney(i,n-1,sentence))/count[denominator_tuple]
            return (f1 + f2)
        else:
            return random.uniform(1e-2,1e-3)


def Witten_Bell(i,n,sentence):
    if n == 1:
        word = sentence[i]
        if word not in distinct_tokens:
            word = 'UNK'
        if count.get(word) != None:
            return count[word]/total_tokens
        else:
            return random.uniform(1e-2,1e-3)

    else:
        numerator_list = []
        denominator_list = []
        for j in range(i-n+1,i+1):
            word = sentence[j]
            if word not in distinct_tokens:
                word = 'UNK'

            numerator_list.append(word)

            # Wi won't be added
            if j != i:
                denominator_list.append(word)

        numerator_tuple = tuple(numerator_list)

        # numerator tuple will always have atleast 2 elements, because n is atleast 2 in else waala case
        if len(denominator_list) == 1:
            denominator_tuple = denominator_list[0]
        else:
            denominator_tuple = tuple(denominator_list)

        if count.get(denominator_tuple) != None:
            cont_cnt = len(continuation_count[denominator_tuple])
            numerator = cont_cnt*Witten_Bell(i,n-1,sentence)
            if count.get(numerator_tuple) != None:
                numerator += count[numerator_tuple]
            denominator = count[denominator_tuple] + cont_cnt
            return numerator/denominator
        else:
            return random.uniform(1e-2,1e-3)


def Perplexity(line,n,type):
    sentence = []
    for i in range(n-1):
        sentence.append('<s>')

    for i in range(len(line)):
        sentence.append(line[i])
    
    for i in range(n-1):
        sentence.append('</s>')

    # use logarithm of probabilties as stated in the book
    logarithm_sum = 0.0
    if type == 'k':
        for i in range(n-1,len(sentence)):
            logarithm_sum += math.log(Kneyser_Ney(i,n,sentence))
    else:
        for i in range(n-1,len(sentence)):
            logarithm_sum += math.log(Witten_Bell(i,n,sentence))

    # returning perplexity
    N = len(sentence)
    return math.exp(-logarithm_sum/N)



def write_LM_file(file_name,data,type,n):
    avg = 0.0
    N = len(data)
    perplexity_score = []
    for i in range(N):
        perplexity = Perplexity(data[i],n,type)
        perplexity_score.append(perplexity)
        avg += perplexity
        print(i)
        

    avg = avg/N
    print('Average is: ',avg)

    file = open(file_name,'w')
    file.write(str(avg))
    file.write('\n')

    for i in range(N):
        text = ''
        for j in range(len(data[i])):
            text += data[i][j]
            text += ' '

        text += '   '
        text += str(perplexity_score[i])
        text += '\n'
        file.write(text)


def fxn(clean_lines):
    initial_count = {}
    for i in range(len(clean_lines)):
        for w in clean_lines[i]:
            if initial_count.get(w) == None:
                initial_count[w] = 0
            initial_count[w] += 1

    return initial_count

def insert_UNK(clean_lines,initial_count):
    global distinct_tokens
    new_lines = []
    for i in range(len(clean_lines)):
        line = []
        for w in clean_lines[i]:
            if initial_count[w] < 2:
                line.append('UNK')
            else:
                line.append(w)
                distinct_tokens.add(w)

        new_lines.append(line)

    return new_lines



if __name__ == '__main__':
    # file_name = 'general-tweets.txt'
    file_name = 'europarl-corpus.txt'
    # file_name = 'medical-corpus.txt'
    clean_lines = tokenize(file_name)

    # storing the initial count. which we will use for assigning the UNK to the low frequency tokens
    initial_count = fxn(clean_lines)
    clean_lines = insert_UNK(clean_lines,initial_count)

    # writing in tokenize file -> tweets file
    # write_cleaned_output('2019101056_tokenize.txt')

    # extracting 1000 lines from clean_lines and keeping them for test data
    l = len(clean_lines)
    test_indexes = random.sample(range(0,l), 1000)
    train_data = []
    test_data = []
    for i in range(0,l):
        if i in test_indexes:
            test_data.append(clean_lines[i])
        else:
            train_data.append(clean_lines[i])

    # 4-gram
    count, continuation_count = calculate_count(train_data)

    # change these train-test values and corpus to get desired perplexity data
    write_LM_file('2019101056_LM4_test-perplexity.txt',test_data,'k',4)