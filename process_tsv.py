import os
from pathlib import Path

def parse_fasta(fasta_path):

    chains = {}
    current_chain = None
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_chain = line[1:]
                chains[current_chain] = ''
            elif current_chain:
                chains[current_chain] += line
    return chains

def create_sabdab_summary(base_dir):

    with open(f"{base_dir}/prot_ids.txt") as f:
        pdb_list = [line.strip() for line in f if line.strip()]

    summary_path = f"{base_dir}/summary.tsv"
    
    with open(summary_path, 'w') as fout:

        fout.write("pdb\tHchain\tLchain\tantigen_chain\tantigen_type\n")

        for pdb_name in pdb_list:
            try:
                pdb = pdb_name[:4]

                native_fasta = parse_fasta(f"{base_dir}/fasta.files.native/{pdb_name}.fasta")
                
                chains = list(native_fasta.keys())
                heavy_chain = chains[0]
                light_chain = chains[1]
                antigen_chains = chains[2:]

                if antigen_chains:
                    antigen_chain_str = " | ".join(antigen_chains)
                    antigen_type_str = " | ".join(["protein"] * len(antigen_chains))
                else:
                    antigen_chain_str = "nan"
                    antigen_type_str = "nan"
                
                fout.write(f"{pdb}\t{heavy_chain}\t{light_chain}\t{antigen_chain_str}\t{antigen_type_str}\n")
                
                print(f"Added: {pdb_name}")
                
            except Exception as e:
                print(f"Error processing {pdb_name}: {e}")
    
    print(f"\nSummary file created: {summary_path}")
    print(f"Total entries: {len(pdb_list)}")
    return summary_path

def main():
    base_dir = "all_data/SAb-23-H2-Ab"
    
    summary_path = create_sabdab_summary(base_dir)
    
    print(f"\n{'='*60}")
    print("Summary TSV file created successfully!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Run the original data processing script with:")
    print(f"   python data/download.py \\")
    print(f"     --summary {summary_path} \\")
    print(f"     --fout {base_dir}/antibody_data.json \\")
    print(f"     --type sabdab \\")
    print(f"     --pdb_dir {base_dir}/pdb.files.native \\")
    print(f"     --numbering imgt \\")
    print(f"     --pre_numbered \\")
    print(f"     --n_cpu 8")
    print("\n2. The output will be in antibody_data.json with correct IMGT numbering")

if __name__ == "__main__":
    main()