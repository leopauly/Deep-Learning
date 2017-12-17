
import sys
sys.stdout = open('fractals.lst', 'w')

for i in range (0,3000):
    c=str(i)
    #print('train_new/input/'+c+'.jpg train_new/output_images/'+c+'.jpg')
    #print('./humanoranimal/'+c+'.jpg 0')
    print('./humanoranimal/'+c+'.jpg 0 2')

for i in range (14000,17000):
    c=str(i)
    #print('train_new/input/'+c+'.jpg train_new/output_images/'+c+'.jpg')
    #print('./humanoranimal/'+c+'.jpg 1')
    print('./humanoranimal/'+c+'.jpg 0 3')
