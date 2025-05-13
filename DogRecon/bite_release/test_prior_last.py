import numpy as np
canine_start = 19

prior_list =[]
if canine_start>18:
    for i in range(144):
        if i <=2*canine_start-36:
            s = 2
        elif 2*canine_start-36 < i <= canine_start:
            s = 2 -(i-(2*canine_start-36))/(36-canine_start)
        elif canine_start < i <= canine_start +36:
            s = 1
        elif canine_start +36 < i <=72:
            s = 1+(i-(canine_start+36))/(36-canine_start)
        elif 72 < i <= 108-canine_start:
            s = 2 -(i-72)/(36-canine_start)

        elif 108-canine_start < i <= 144-canine_start:
            s = 1
        elif 144-canine_start < i <= 180-2*canine_start:
            s = 1+ (i-144+canine_start)/(36-canine_start)
        elif 180-2*canine_start < i<= 144:
            s = 2
        prior_list.append(s)


else:
    for i in range(144):
        if i <=canine_start:
            s = 2 - i/canine_start
        elif canine_start < i <= canine_start+36:
            s = 1
        elif canine_start+36 < i <= canine_start*2 +36:
            s = 1+ (i-canine_start)/canine_start
        elif canine_start*2 +36 < i <= 108-2*canine_start:
            s = 2
        elif 108-2*canine_start < i <= 108-canine_start:
            s = 2 -(i-(108-2*canine_start))/canine_start

        elif 108-canine_start < i <= 144-canine_start:
            s = 1
        elif 144-canine_start < i <= 144:
            s = 1+ (i-144+canine_start)/canine_start
        prior_list.append(s)
prior_npy = np.array(prior_list)/sum(prior_list)
print(prior_npy)