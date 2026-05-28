import numpy as np
import pandas as pd
import argparse
import pysam


class File:

    def get_vcf_file(self, vcf_path, variants):
        # Read the CSV-formatted candidate table instead of a VCF.
        variants_pos = pd.read_csv(vcf_path, sep=",")

        vcf_detail = []

        if variants == 'whole':
            for _, row in variants_pos.iterrows():
                chr_id = str(row["chr"])
                scan_l_pos = int(row["position"])
                scan_r_pos = scan_l_pos + 1000
                vcf_tuples = (chr_id, scan_l_pos, scan_r_pos)
                vcf_detail.append(vcf_tuples)
            return vcf_detail


    def get_sam_file(self,bam_path):
        sam_file = pysam.AlignmentFile(bam_path,"rb")
        if sam_file == None:
            print("bam_file is empty")
            return
        return sam_file



