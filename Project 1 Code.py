import sys
import numpy as np

def load_graph(file_path):
    link_structure = {}
    with open(file_path, 'r') as f:
        for line in f:
            node, links = line.strip().split(':')
            link_structure[int(node)] = list(map(int, links.split(','))) if links else []
    return link_structure

def initialize_ranks(node_count):
    return {node: 1.0 / node_count for node in range(node_count)}

def calculate_pagerank(graph, damping_factor, max_iterations=100, tolerance=1.0e-6):
    total_nodes = len(graph)
    ranks = initialize_ranks(total_nodes)
    
    for _ in range(max_iterations):
        updated_ranks = {node: (1 - damping_factor) / total_nodes for node in graph}
        
        for node, outbound_links in graph.items():
            if outbound_links:
                shared_rank = ranks[node] / len(outbound_links)
                for linked_node in outbound_links:
                    updated_ranks[linked_node] += damping_factor * shared_rank
        
        if np.linalg.norm(np.array(list(updated_ranks.values())) - np.array(list(ranks.values())), ord=1) < tolerance:
            break
        ranks = updated_ranks
    
    return ranks

def store_results(output_file, ranks):
    with open(output_file, 'w') as f:
        for node, rank in sorted(ranks.items()):
            f.write(f"{rank:.10e}\n")

if __name__ == "__main__":
    input_filename = sys.argv[1]
    damping = float(sys.argv[2])
    output_filename = "output.txt"
    
    web_graph = load_graph(input_filename)
    final_ranks = calculate_pagerank(web_graph, damping)
    store_results(output_filename, final_ranks)
