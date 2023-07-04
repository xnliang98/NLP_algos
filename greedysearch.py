
def greedy_search_decoder(data):
    max_probs = [max(d) for d in data]
    indexes = [d.index(p) for d, p in zip(data, max_probs)]
    return indexes

if __name__ == '__main__':

    data = [[0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.5, 0.3, 0.2]]
    
    result = greedy_search_decoder(data)
    print("****use greedy search decoder****")
    print(result)
