import pandas as pd
import numpy as np

def read_data(Nt_file:str, outfile:str):
    '''
    Get necessary information and save to a file.
    :param Nt_file: Nt sequence without gaps
    :param outfile: output file path and name
    :return: None
    '''
    df_n = pd.read_csv(Nt_file, sep='\t')
    df_n['JUNCTION length'] = df_n['JUNCTION end'] - df_n['JUNCTION start'] + 1
    # sequence with 2 V/J gene annotations: flatten into two rows
    df_n['2 possible V-GENE alleles'] = df_n.loc[:, 'V-GENE and allele'].str.split(", or").str.len() == 2
    df_n['2 possible J-GENE alleles'] = df_n.loc[:, 'J-GENE and allele'].str.split(", or").str.len() == 2
    df_n['V-GENE and allele'] = df_n.loc[:, 'V-GENE and allele'].str.split(", or").values
    df_n['J-GENE and allele'] = df_n.loc[:, 'J-GENE and allele'].str.split(", or").values
    df_n = df_n.explode('V-GENE and allele').reset_index(drop=True)
    df_n = df_n.explode('J-GENE and allele').reset_index(drop=True)

    df_nes = df_n[
        ['Sequence number', 'Sequence ID', 'V-GENE and allele', 'J-GENE and allele', 'V-D-J-REGION', 'JUNCTION',
         'JUNCTION length', '2 possible V-GENE alleles', '2 possible J-GENE alleles']]
    df_nes.to_csv(outfile, sep='\t', index=False)

# def group_seq(df_nes:pd.DataFrame,keys=('V-GENE and allele','J-GENE and allele','JUNCTION length')):
#     seq_groups = df_nes.groupby(list(keys)).groups
#     return seq_groups.keys(),seq_groups.values()


# def retrieve_group(key:tuple,seq_groups:dict,df:pd.DataFrame) -> pd.DataFrame:
#     df_group = pd.DataFrame([])
#     return df_group

def norm_Hamming_dist(seq1,seq2,gaps='ignore'):
    return

def dist_groupwise(df_group,dist:str):
    return

def dist_to_nearest(df,dist='norm_Hamming'):
    return

def det_cutoff():
    return

def check_duplicate(): # check duplicated sequence across groups
    return
