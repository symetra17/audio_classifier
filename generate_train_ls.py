import glob,os,random
import json

inp_folder = R'C:\Users\dva\Documents\audio_design\images'
folders = glob.glob(os.path.join(inp_folder, '*'))
train_ls = []
test_ls = []
fid_train = open('train_set.json','w')
fid_test = open('test_set.json','w')

for idx, folder in enumerate(folders):
    print('Id:', idx, ' ',folder, )
    files = glob.glob(os.path.join(folder,'*.*'))    
    for fname in files:
        if random.random() > 0.3:
            train_ls.append((idx,fname))
            #fid_train.write('%d %s\n'%(idx, fname))
        else:
            test_ls.append((idx,fname))
            #fid_test.write('%d %s\n'%(idx, fname))


json.dump(train_ls, fid_train)
json.dump(test_ls, fid_test)

fid_train.close()
fid_test.close()
