import argparse
import os
import subprocess

from create_inference_graphs import create_inference_graph
from inference import inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reads', required=True, type=str, help='Path to the reads')
    # parser.add_argument('--asm', type=str, help='Assembler used')
    parser.add_argument('-o', '--out', type=str, default='.', help='Output directory')
    parser.add_argument('-t', '--threads', type=str, default=1, help='Number of threads to use')
    parser.add_argument('-m', '--model', type=str, default='weights/weights.pt', help='Path to the model')
    args = parser.parse_args()

    reads = args.reads
    out = args.out
    threads = args.threads
    model = args.model
    asm = 'hifiasm'

    # Step 1
    print(f'\nStep 1: Running {asm} on {reads} to generate the graph')
    hifiasm_out = f'{out}/hifiasm/output'
    if not os.path.isdir(hifiasm_out):
        os.makedirs(hifiasm_out)
    subprocess.run(f'./vendor/hifiasm-0.18.8/hifiasm --prt-raw -o {hifiasm_out}/asm -t{threads} -l0 {reads}', shell=True)
    
    # Step 2
    print(f'\nStep 2: Preparing the graph for the inference')
    gfa = f'{hifiasm_out}/asm.bp.raw.r_utg.gfa'
    create_inference_graph(gfa, reads, out, asm)
    
    # Step 3
    print(f'\nStep 3: Using the model {model} to run inference on {reads}')
    inference(data_path=out, assembler=asm, model_path=model, savedir=os.path.join(out, asm))

    asm_dir = f'{out}/{asm}/assembly'
    print(f'\nDone!')
    print(f'Assembly saved in: {asm_dir}/0_assembly.fasta')
    print(f'Thank you for using GNNome!')