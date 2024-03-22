import argparse
import pickle
import os
import graph_parser
import dgl


def create_inference_graph(gfa_path, reads_path, out_dir, assembler):
    assert os.path.isfile(gfa_path), "GFA not found!"
    assert os.path.isfile(reads_path), "Reads not found!"

    print(f'Starting to parse assembler output')
    graph, pred, succ, reads, edges, read_to_node, node_to_read, _ = graph_parser.only_from_gfa(gfa_path, training=False, reads_path=reads_path, get_similarities=True)
    print(f'Parsed assembler output! Saving files...')

    out_dir = os.path.join(out_dir, assembler)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    processed_dir = f'{out_dir}/processed'
    info_dir = f'{out_dir}/info'
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)
    if not os.path.isdir(info_dir):
        os.mkdir(info_dir)

    processed_path = f'{processed_dir}/0.dgl'
    dgl.save_graphs(processed_path, graph)
    pickle.dump(pred, open(f'{info_dir}/0_pred.pkl', 'wb'))
    pickle.dump(succ, open(f'{info_dir}/0_succ.pkl', 'wb'))
    pickle.dump(reads, open(f'{info_dir}/0_reads.pkl', 'wb'))
    pickle.dump(edges, open(f'{info_dir}/0_edges.pkl', 'wb'))
    pickle.dump(read_to_node, open(f'{info_dir}/0_read_to_node.pkl', 'wb'))
    if node_to_read is not None:
        pickle.dump(node_to_read, open(f'{info_dir}/0_node_to_read.pkl', 'wb'))
    print(f'Processing of graph done!\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gfa', type=str, help='Path to the GFA graph file')
    parser.add_argument('--reads', type=str, help='Path to the FASTA/Q reads file')
    parser.add_argument('--asm', type=str, help='Assembler used')
    parser.add_argument('--out', type=str, help='Output directory')
    args = parser.parse_args()

    gfa = args.gfa
    reads = args.reads
    out = args.out
    asm = args.asm
    
    create_inference_graph(gfa, reads, out, asm)
