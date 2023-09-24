import os
from glob import glob

x = glob('../../data/' + "refined_patents" + '/raw/g_detail*.tsv')
x.sort(reverse=True)
print(x)
