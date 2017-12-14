# Creating image-name list for caffe programs to read from

import sys
sys.stdout = open('test_CRACK.txt', 'w') # filename

for i in range (1,57):
    c=str(i)
    #print('train_new/input/'+c+'.jpg train_new/output_images/'+c+'.PNG')  # Segmentation (input and output are images)
    print('c ('+c+').PNG 0') # Classification (input images and output labels)
    print('c ('+c+').PNG') # Testing (input images and no output)
