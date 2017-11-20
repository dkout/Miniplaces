f=open('test.txt','w')
for i in range(10001):
    i = str(i)
    zeros=8-len(i)
    f.write('test/'+zeros*'0'+i+'.jpg\n')
f.close()
