import os
train=open('train_syn.txt','w')
trainlist=os.listdir('../low_high_datasets/SYN_1/train/low/')
for name in trainlist:
    train.writelines(name+'\n')
train.close()

test=open('test_syn.txt','w')
testlist=os.listdir('../low_high_datasets/SYN_1/eval/low/')
for name in testlist:
    test.writelines(name+'\n')
test.close()