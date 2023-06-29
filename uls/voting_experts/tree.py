import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait


class Stats:
    def __init__(self):
        self.freq_mean = 0
        self.freq_std = 0
        self.entropy_mean = 0
        self.entropy_std = 0

class Tree:
    def __init__(self, level):
        self.nodes = {}
        self.frequency = 0
        self.entropy = 0
        self.level = level

    def add_node(self, letter):
        if letter in self.nodes.keys():
            self.nodes[letter].frequency += 1
        else:
            self.nodes[letter] = Tree(self.level+1)
            self.nodes[letter].frequency += 1
        return self.nodes[letter]

    def add_node_frequency(self, letter, frequency):
        if letter in self.nodes:
                self.nodes[letter].frequency += frequency
        else:
            self.nodes[letter] = Tree(self.level+1)
            self.nodes[letter].frequency = frequency
        return self.nodes[letter]
    

def print_tree(tree, output, parent=''):
    for letter, node in tree.nodes.items():
        output.write(f'{node.frequency}, {parent+letter}')
        print_tree(node, parent+letter)

def build_tree(data, depth):
    root = Tree(0)
    update_tree(root, data, depth)
    return root

def calculate_entropy(root, data, depth):
    log_entropy = 0
    for i in range(len(data)):
        last = root.nodes[data[i]]
        log_entropy += np.log(last.entropy)
        for j in range(depth - 2):
            last = last.nodes[data[i + j + 1]]
            log_entropy += np.log(last.entropy)
    return log_entropy/len(data)

def update_tree(root, data, depth):
    for i in range(len(data)):
        last = root.add_node(data[i])
        for j in range(depth - 2):
            if i + j + 1 < len(data):
                last = last.add_node(data[i + j + 1])

def add_standard_values(standard, level, type, values):
    if level in standard:
        if type in standard[level]:
            standard[level][type].extend(values)
        else:
            standard[level][type] = values
    else:
        standard[level] = {}
        standard[level][type] = values
    return standard

def calculate_experts_features(node, standard):
    for _, child in node.nodes.items():
        calculate_experts_features(child, standard)
    frequencies = [child.frequency for child in node.nodes.values()]
    entropies = [child.entropy for child in node.nodes.values()]

    standard = add_standard_values(standard, node.level+1, "freq", frequencies)
    standard = add_standard_values(standard, node.level+1, "entropy", entropies)

    sum_of_occurrences = sum([child.frequency for child in node.nodes.values()])
    for child in node.nodes.values():
        px = child.frequency / sum_of_occurrences
        node.entropy -= px * np.log2(px)

    return standard

def get_stats(standard):
    stats = {}
    for level, values in standard.items():
        stats[level] = Stats()
        if (values["freq"] == [] and values["entropy"] == []):
            continue
        stats[level].freq_mean = np.mean(np.array(values["freq"]))
        stats[level].freq_std = np.std(np.array(values["freq"]))
        stats[level].entropy_mean = np.mean(np.array(values["entropy"]))
        stats[level].entropy_std = np.std(np.array((values["entropy"])))
    return stats

def standardized_frequency(node, stats):
    if stats[node.level].freq_std == 0:
        return node.frequency
    return (node.frequency - stats[node.level].freq_mean)/stats[node.level].freq_std

def standardized_entropy(node, stats):
    if stats[node.level].entropy_std == 0:
        return node.entropy

    return (node.entropy - stats[node.level].entropy_mean)/stats[node.level].entropy_std

def standardize(root, stats, threads=55):
    if root is None:
        return
    
    queue = []
    for node in root.nodes.values():
        queue.append(node)
 

    # Dividing tree into {threads} parts to process all of them independently
    index = 0
    divided_queues = []
    quotient, remainder = len(queue) // threads, len(queue) % threads
    for _ in range(threads):
        sublist_size = quotient + 1 if remainder > 0 else quotient
        divided_queues.append(queue[index:index + sublist_size])
        index += sublist_size
        remainder -= 1

    # Standardize in parallel
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(inner_standardize, q, stats) for q in divided_queues]
        wait(futures)

def inner_standardize(queue, stats):
    while queue:
        queue[0].frequency = standardized_frequency(queue[0], stats)
        queue[0].entropy = standardized_entropy(queue[0], stats)
        node = queue.pop(0)
        queue.extend(node.nodes.values())

def find_node(seq, ngram_tree):
    last = ngram_tree
    for letter in seq:
        last = last.nodes[letter]
    return last

def find_child(child, ngram_tree):
    if child in ngram_tree.nodes:
        return ngram_tree.nodes[child]
    else:
        return None 

def tree_from_ngram(root, ngram_filename):
    lasts=[]
    with open(ngram_filename, "r") as ngram_file:
        for ngram_entry in ngram_file:
            ngram_parts = ngram_entry.split('\t')
            ngram = ngram_parts[0].split(' ')
            counts = int(ngram_parts[1])
            if len(ngram)==1:
                lasts= [root.add_node_frequency(ngram[0], counts)]
            else:
                if len(ngram) > len(lasts):
                    lasts.append(lasts[-1].add_node_frequency(ngram[-1], counts))
                else:
                    lasts = lasts[:len(ngram)-1]
                    lasts.append(lasts[-1].add_node_frequency(ngram[-1], counts))
    return root
    