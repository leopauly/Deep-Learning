
import sys
sys.stdout = open('test.lst', 'w')

for i in range (3000,3500):
    c=str(i)
    #print('train_new/input/'+c+'.jpg train_new/output_images/'+c+'.jpg')
    print('./humanoranimal/'+c+'.jpg 0')

for i in range (17000,17500):
    c=str(i)
    #print('train_new/input/'+c+'.jpg train_new/output_images/'+c+'.jpg')
    print('./humanoranimal/'+c+'.jpg 1')
