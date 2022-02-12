"""Functions defining preprocessing steps for IgBlast output data."""

import re
import pandas as pd
import argparse

def clean_data(df):
    """
    Remove items if productive != "T".
    """
    n_rows1, _ = df.shape
    print("%d sequences in total." % n_rows1)
    df = df.loc[df["productive"] == "T", :]
    n_rows2, _ = df.shape
    print("Drop %d sequences by filtering Functionality." % (n_rows1 - n_rows2))
    print("%d sequences remain." % n_rows2)
    return df

def read_VJ_genes(allele_notation) -> str:
    """
    Get the V/J fragment annotation regardless of the allele types. If multiple fragment types
    appear, only preserve the first one.
    """
    alleles = re.split(",\s|\s+|\*|,", allele_notation)
    genes = [gene for gene in alleles if "IGH" in gene]
    genes = list(set(genes))
    return genes[0]


def read_data(df_n):
    """
    Get necessary information for analysis, save to a file.
    :param nt_file: IgBlast output data. Should contain the following fields: 
    sequence_id, sequence, productive, complete_vdj, v_call, j_call, junction, junction_length
    :param outfile: output file path and name
    :return: None
    """

    # clean the data
    df_n = clean_data(df_n)
    # getV/J gene annotations
    df_n['V-GENE'] = df_n['v_call'].apply(read_VJ_genes)
    df_n['J-GENE'] = df_n['j_call'].apply(read_VJ_genes)
    # rename
    df_nes = df_n[['sequence_id','sequence', 'V-GENE', 'J-GENE', 'junction', 'junction_length']]
    df_nes = df_nes.rename(columns={'sequence_id':'Sequence ID','sequence':'Sequence',
        'junction':'JUNCTION', 'junction_length':'JUNCTION length'})
    # drop nan
    rows1, _ = df_nes.shape
    df_nes = df_nes.dropna()
    rows2, _ = df_nes.shape
    print("Drop %d rows contain nan." % (rows1 - rows2))
    # sort by V,J,junction length
    df_nes = df_nes.sort_values(by=['V-GENE', 'J-GENE','JUNCTION length'])
    return df_nes


