import pandas as pd
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import func
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score
from fuzzywuzzy import fuzz


    
main_dataset=pd.read_excel('/home/sahil/Code/Python/Projects/Laptop Price Prediction/Laptop Price Prediction/Laptops_data.xlsx')

gpu_dataset=pd.read_csv('/home/sahil/Code/Python/Projects/Laptop Price Prediction/Laptop Price Prediction/GPU_Full_Details.csv')

#Features in both the datasets
print(gpu_dataset.columns)
print(main_dataset.columns)

#pruning the SSD Capacity Field
sc=main_dataset['SSD Capacity']
k=[]
for i in sc:
    if i==0:
        k.append(i)
    else:
        ls=i.split(' ')
        if ls[1]=="TB":
            k.append(int(ls[0])*1000)
        else:
            k.append(int(ls[0]))
            
#pruning the OS Field
possible_OS=['Mac','Windows','Chrome', 'DOS','Prime']

osList=[]
for i in main_dataset['Operating System']:
    stri=i.split(' ')
    for j in possible_OS:
        if j == stri[0]:
            osList.append(j)
            break
    if stri[0]=='macOS':
        osList.append('Mac')


#pruning the GPU field and inserting the benchmark feature in the pruned dataset
bm=[]

for i in main_dataset['Graphic Processor']:
    if i== 'ABC':
        bm.append(0)
        continue
    temp=[]
    for j in gpu_dataset['Model']:
        temp.append(fuzz.token_sort_ratio(i,j))
    highest=max(temp)
    index_of_max = temp.index(highest)
    bm.append(gpu_dataset['Benchmark'][index_of_max])

#to prune the price field and insert it into new dataset    
temp=[]

for price in main_dataset['Price']:
    if ',' in price:
        price = price.replace(',','')  
    if '₹' in price:
        price = price.replace('₹', '')  
    price=float(price)
    temp.append(price)

#to convert the RAM field into float
func.removeText(main_dataset, 'RAM')

ram=main_dataset['RAM']
ssd=main_dataset['SSD']
gpu=main_dataset['Graphic Processor']


df=pd.DataFrame({'SSD':ssd,'SSD Capacity':k,'RAM':ram, 'OS':osList,'GPU':gpu, 'GPU Benchmark': bm,'Price':temp})
df.to_csv('Updated.csv', index=False)



ssd = np.array(df['SSD']).reshape(-1, 1)
os=np.array(df['OS']).reshape(-1, 1)


ssd_encoder = OrdinalEncoder(categories=[["No","Yes"]])
ssd_encoder.fit(ssd)
ssd_encoded_data = ssd_encoder.transform(ssd)
ssd_encoded_data = ssd_encoded_data.ravel()


os_encoder = OrdinalEncoder(categories=[["Prime","Chrome","DOS","Windows","Mac"]])
os_encoder.fit(os)
os_encoded = os_encoder.transform(os)
os_encoded = os_encoded.ravel()


y=df['Price']
x=pd.DataFrame({'SSD cap':df['SSD Capacity'],'encSSD':ssd_encoded_data,'OSenc':os_encoded,'RAM':df['RAM'],'Benchmark':df['GPU Benchmark']})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 28)
max_depth=math.ceil(math.log(len(x), 2))
regressor=RandomForestRegressor(n_estimators=100,max_depth= max_depth, n_jobs=  -1)

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#r2 score
r2_scr=r2_score(y_test, y_pred)
print("R2 Score", r2_scr)

#root mean square error
MSE = mean_squared_error(y_test, y_pred)
rmse = float(format(np.sqrt(MSE)))
print("\nRMSE: ", rmse)


#taking inputs to predict prices
ram_inp=input("Enter the numeric value of Ram: ")
ssd_inp=input("Do you want SSD on your laptop? Yes/No: ")
ssdCap_inp=input("Enter the SSD capactiy in GB/TB: ")
os_inp=input("Enter the OS you want eg: Mac/Windows/DOS/Chrome/Prime: ")
gpu_inp=input("Wnat GPU do you want on your device? Note: IF MAC OR if no GPU then type \"ABC\": ")

myinp=func.input_tuning(ram_inp, ssd_inp, ssdCap_inp, os_inp, gpu_inp, os_encoder,  gpu_dataset)

y_res=int(regressor.predict(myinp))
print('The approx price of the laptop with provided specification will be',y_res)
