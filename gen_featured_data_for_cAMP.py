import sys

import pandas as pd

from featured_data_generated import cal_pep_des

if __name__ == '__main__':
    input_tsv, filtered_data_tsv, featured_data_csv = sys.argv[1: 4]
    # input_tsv = "task/GEM_GMBC_ripp_camp.result.tsv"
    df = pd.read_csv(input_tsv, sep='\t')
    df = df[~df["sequence"].str.contains("B|J|X|Z|O|U")] 
    df = df[df["seq"].apply(lambda x: 5 < len(x) < 50)].reset_index(drop=True)
    df.to_csv(filtered_data_tsv, sep='\t', index=False)
    sequence = df["seq"]
    sequence.name = "sequence"
    peptides = sequence.values.copy().tolist()
    print(len(peptides))
    results = None
    types = None
    featured_df = cal_pep_des.cal_pep(peptides, sequence, results, types)
    featured_df.to_csv(featured_data_csv, encoding="utf-8")
