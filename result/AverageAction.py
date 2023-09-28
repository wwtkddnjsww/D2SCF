import numpy as np
from matplotlib import pyplot as plt
import math

#todo: Cost / 10 해야 단위 맞출 수 있음. !! Energy Consumption이랑 각각 Completion Time도!

axis_x = np.arange(1,502)



opti_graph   = np.load('DDPG_Action_List.npy',allow_pickle=True)


head_md   = np.zeros(501)
head_edge = np.zeros(501)
tail_md   = np.zeros(501)
tail_edge = np.zeros(501)

print(len(opti_graph[0]))

opti_temp = []

for i in range(len(opti_graph)):
    temp = 0
    for j in range(len(opti_graph[i])):
        if len(opti_graph[i][j]) == 1:
            print(opti_graph[i][j])
            head_edge[i]+=1
            tail_edge[i]+=8
        else:
            head_md[i]+=1
            for k in opti_graph[i][j]:
                if k == 0:
                    tail_edge[i]+=1
                elif k == 1:
                    tail_md[i]+=1
head_md = head_md/1000
head_edge = head_edge/1000
tail_md = tail_md/1000
tail_edge = tail_edge/1000


print(head_md)
print(head_edge)
print(tail_md)
print(tail_edge)
# print(opti_graph)

# for i in range(6):
#     for j in range(i):
#         np.delete(opti_graph[i],0)
#
# acc_graph = np.max(acc_graph, axis=0)
# opti_graph = np.max(opti_graph, axis=0)
#
# print('optimal_max: ', np.max(opti_graph))
# print('rand_max: ', np.max(rand_graph))
# print('static_max: ', np.max(static_graph))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"]="stix"
plt.rcParams["figure.figsize"] = (8,6)
#plt.grid(linestyle='dotted')
plt.ylabel('Reward',fontsize=18)
plt.xlabel('Iterations',fontsize=18) #수식 : r'\수식내용'
#plt.tick_params(axis='x', direction='in',top=True,bottom=True)
#plt.tick_params(axis='y', direction='in',left=True,right=True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0,500)
plt.ylim(0,9)
width = 1.2
plt.bar(axis_x,head_edge, color = '#111111', label = 'Head model at edge cloud',width = width)
plt.bar(axis_x,head_md, bottom = head_edge, color = '#444444', label = 'Head model at MD',width = width)
plt.bar(axis_x,tail_edge, bottom = head_md+head_edge,  color = '#888888', label = 'Tail model at edge cloud',width = width)
plt.bar(axis_x,tail_md, bottom = head_md+head_edge+tail_edge,  color = '#AAAAAA', label = 'Tail model at MD',width = width)


# plt.plot(axis_x, edge_graph,   label = 'EDGE-ONLY', zorder=1)
# plt.plot(axis_x, md_graph,     label = 'MD-ONLY', zorder=1)
# plt.plot(axis_x, head_graph,   label = 'STATIC', zorder=1)
# plt.plot(axis_x, random_graph,   label = 'RANDOM', zorder=1)


plt.xlabel('Episodes', labelpad=10)
plt.ylabel('Number of Models', labelpad=10)
#
# #plt.title('Client model bit')
plt.legend(loc='best',edgecolor='black',fontsize=14,fancybox=False)#bbox_to_anchor=(0.9,0.0)
plt.savefig('Models.pdf',dpi=300,transparent=True,bbox_inches='tight')
plt.savefig('Models.eps',dpi=300,transparent=True,bbox_inches='tight')

plt.show()
# target_acc=70



# for i in range(200):
#     if acc_8bit[i] > target_acc:
#         print('8bit:', i)
#         break
#
# for i in range(200):
#     if acc_16bit[i] > target_acc:
#         print('16bit: ', i)
#         break
#
# for i in range(200):
#     if acc_32bit[i] > target_acc:
#         print('32bit: ', i)
#         break

