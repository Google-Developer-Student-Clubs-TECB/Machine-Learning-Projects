import re
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

def removeText(data, field):
    iter=0
    for text in data[field]:
        
        # Using re.search to find the numeric part
        match = re.search(r'[\d.]+', text)
        # Check if a match was found
        if match:
            numeric_part = match.group()
            numeric_float = float(numeric_part)
        else:
            numeric_float=0.0
            
        data[field][iter]=numeric_float
        iter+=1
    
def input_tuning(ram,ssd,ssdCap,os,gpu,osEncoder,gpu_dataset):
    ram=ram.split(' ')
    ram=float(ram[0])
    if(ssd=='Yes'):
        ssd=1.0
    else:
        ssd=0.0
    ssdCap=ssdCap.split(' ')
    if(ssdCap[1]=="GB"):
        ssdCap=float(ssdCap[0])
    else:
        ssdCap=float(ssdCap[0]*1000)
    os=np.array([os])
    os=os.reshape(-1,1)
    os_encoded = osEncoder.transform(os)
    if gpu != 'ABC':
        res=[]
        for i in  gpu_dataset['Model']:
            if(i=="ABC"):
                res.append(0.0)           
            res.append(fuzz.ratio(i, gpu))
        highest=max(res)
        index_of_max = res.index(highest)
        benchmarkScore=gpu_dataset['Benchmark'][index_of_max]
    else:
        benchmarkScore=0.0
    
    
    x=pd.DataFrame({'SSD cap':[ssdCap],'encSSD':[ssd],'OSenc':[os_encoded],'RAM':[ram],'Benchmark': [benchmarkScore]})
    return x
        
    