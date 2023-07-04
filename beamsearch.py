
def beam_search_decoder(data, k):
    sequences = [[[], 1.0]]

    for row in data:
        candidates = []
        for i in range(len(row)):
            for sequence in sequences:
                seq, score = sequence[0], sequence[1]
                candidates.append([seq + [i], row[i] * score])
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        sequences = sorted_candidates[:k]
    
    return sequences

if __name__ == '__main__':

    data = [[0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.5, 0.3, 0.2]]
    
    result = beam_search_decoder(data, 3)
    print("****use beam search decoder****")
    for seq in result:
        print(seq)
