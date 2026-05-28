import numpy as np
import pandas as pd


class VCF:
	def __init__(self, vcf_file):
		self.vcf_file = vcf_file
		self.sv_50_200 = []
		self.sv_200_700 = []
		self.sv_700_1000 = []
		self.sv_1000 = []
		self.sv_other = []


	def vcf_class(self):
		for vcf in self.vcf_file:
			if(abs(vcf[1]-vcf[2])>=50 and abs(vcf[1]-vcf[2])<200):
				self.sv_50_200.append(vcf)
			elif(abs(vcf[1]-vcf[2])>=200 and abs(vcf[1]-vcf[2])<700):
				self.sv_200_700.append(vcf)
			elif(abs(vcf[1]-vcf[2])>=700 and abs(vcf[1]-vcf[2])<1000):
				self.sv_700_1000.append(vcf)
			elif(abs(vcf[1]-vcf[2])>=1000):
				self.sv_1000.append(vcf)
			else:
				self.sv_other.append(vcf)

