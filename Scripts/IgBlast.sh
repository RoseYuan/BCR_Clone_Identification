#!/bin/bash

cd "/home/siyuan/software/ncbi-igblast-1.17.1" || exit
# awk '{ print ">"NR"\n"$0 }' sample90_sequence.txt > sample90.fasta


# for real data
sVAR1="76 77 78 79 82 83 86 87 88 89 90 91 92 93 94 95 96 97 98 99"
VAR1=($sVAR1)
count1=${#VAR1[@]}

for j in $(seq 1 $count1); do
  sample=${VAR1[$j-1]}
  fasta="/home/siyuan/thesis/Data/new_data/rerun/datasets/sample${sample}.fasta"
  rst="/home/siyuan/thesis/Data/new_data/rerun/datasets/sample${sample}_IgBlast.txt"

  bin/igblastn -germline_db_V database/IGHV_shortname.fasta -germline_db_J database/IGHJ_shortname.fasta -germline_db_D database/IGHD_shortname.fasta -organism human -query $fasta -auxiliary_data optional_file/human_gl.aux -show_translation -outfmt 19 > $rst
done


# for simulated data
sVAR1="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
VAR1=($sVAR1)
count1=${#VAR1[@]}

for j in $(seq 1 $count1); do
  sample=${VAR1[$j-1]}
  echo $sample
  fasta="/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/datasets/2_${sample}.fasta"
  rst="/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/datasets/2_${sample}_IgBlast.txt"

  bin/igblastn -germline_db_V database/IGHV_shortname.fasta -germline_db_J database/IGHJ_shortname.fasta -germline_db_D database/IGHD_shortname.fasta -organism human -query $fasta -auxiliary_data optional_file/human_gl.aux -show_translation -outfmt 19 > $rst
done

#sVAR1="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
#VAR1=($sVAR1)
#count1=${#VAR1[@]}
#
#for j in $(seq 1 $count1); do
#  sample=${VAR1[$j-1]}
#  echo $sample
#  fasta="/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/datasets/3_${sample}.fasta"
#  rst="/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/datasets/3_${sample}_IgBlast.txt"
#
#  bin/igblastn -germline_db_V database/IGHV_shortname.fasta -germline_db_J database/IGHJ_shortname.fasta -germline_db_D database/IGHD_shortname.fasta -organism human -query $fasta -auxiliary_data optional_file/human_gl.aux -show_translation -outfmt 19 > $rst
#done
#
#sVAR1="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"
#VAR1=($sVAR1)
#count1=${#VAR1[@]}
#
#for j in $(seq 1 $count1); do
#  sample=${VAR1[$j-1]}
#  echo $sample
#  fasta="/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/datasets/4_${sample}.fasta"
#  rst="/home/siyuan/thesis/Data/Afree_paper_data/simulated/rerun/datasets/4_${sample}_IgBlast.txt"
#
#  bin/igblastn -germline_db_V database/IGHV_shortname.fasta -germline_db_J database/IGHJ_shortname.fasta -germline_db_D database/IGHD_shortname.fasta -organism human -query $fasta -auxiliary_data optional_file/human_gl.aux -show_translation -outfmt 19 > $rst
#done

#IgBlast output explanation (-outfmt 19)
#https://docs.airr-community.org/en/stable/datarep/rearrangements.html
