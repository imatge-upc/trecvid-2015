from math import ceil

def takespread(sequence, num):
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(ceil(i * length / num))]



if __name__ == "__main__":

    sequence = ['a','b','c','d','e','f','g','h','i']
    new_sequence = []
    num = 5

    new_sequence.extend(takespread(sequence,num))

    print new_sequence