#for i in range(1,8):
#    for j in range(1,25):
#        for k in range(1,6):
#            print(1t4r_+"i"+_+"j"+_00+"k")
#
#            
#for i in range(1,8):
#    for j in range(1,25):
#        for k in range(1,6):
#            print('1t4r_',i,'_',j,'_00',k,)
#            
#            for i in range(1,10):
#      for j in range(1,10):
#            print(i, '*', j, '=', i*j, '\t', sep='',end='')
#
#
#


#result_txt = open('list.txt' ,'w')
#
#for j in range(1,25):
#    result_txt.write("\n")
#    for i in range(1,8):
#        result_txt.write("\n")
#        for k in range(1,6):
#            if k==1:
#               result_txt.write("[") 
#            result_txt.write("*1t4r_%d_%d_00%d*, "%(i,j,k))
#            if k==5:
#               result_txt.write("],")
#            
#result_txt.close()

result_txt = open('list.txt' ,'w')

for i in range (11, 13):
    result_txt.write("\n")
#    for m in (1):
#        result_txt.write("\n")
    for j in range(1, 9):
        result_txt.write("\n")
        for k in range(1, 21):
            if k==1:
               result_txt.write("[") 
            result_txt.write("*3t4r_1_%d_%d_00%d*, "%(i, j, k))
            if k==20:
               result_txt.write("],")
                   
result_txt.write("\n")
for i in range (11, 13):
    result_txt.write("\n")
#    for m in (1):
#        result_txt.write("\n")
    for j in range(1, 6):
        result_txt.write("\n")
        for k in range(1, 21):
            if k==1:
               result_txt.write("[") 
            result_txt.write("*3t4r_2_%d_%d_00%d*, "%(i, j, k))
            if k==20:
               result_txt.write("],")
        
result_txt.close()

#result_txt = open('list.txt' ,'w')


#for i in range(1,8):          
#    for j in [4, 7, 11, 13, 21, 23]:
#        for k in range(1,6):
#            result_txt.write("*1t4r_%d_%d_00%d*, " % (i,j,k))            

#for i in range(1,8):          
#    for j in [1, 2, 3, 5, 6, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 22, 24]:
#        for k in range(1,6):
#            result_txt.write("*1t4r_%d_%d_00%d*, " % (i,j,k)) 
#            
#result_txt = open('list2.txt' ,'w')
         

#for i in range(6,7):          
#    for j in [14, 15, 16, 17, 18, 19, 20, 22, 24]:
#        for k in range(1,6):
#            result_txt.write("*1t4r_%d_%d_00%d*, " % (i,j,k)) 


#result_txt.close()

#fp = open('list.txt, 'r')