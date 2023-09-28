import numpy as np
from matplotlib import pyplot as plt
import math

#todo: Cost / 10 해야 단위 맞출 수 있음. !! Energy Consumption이랑 각각 Completion Time도!

axis_x = np.arange(1,502)



opti_graph   = np.load('DDPG_Cost.npy')
edge_graph   = np.load('Edge_Cost.npy')
md_graph     = np.load('MD_Cost.npy')
head_graph   = np.load('Static_Cost.npy')
random_graph = np.load('RAND_Cost.npy')

opti_graph = opti_graph/50


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
plt.grid(linestyle='dotted')
plt.ylabel('Reward',fontsize=18)
plt.xlabel('Iterations',fontsize=18) #수식 : r'\수식내용'
plt.tick_params(axis='x', direction='in',top=True,bottom=True)
plt.tick_params(axis='y', direction='in',left=True,right=True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.xticks([5000,10000,15000,20000,25000], [100, 200, 300, 400, 500])

plt.plot(axis_x, -opti_graph,   label = 'D2SCM', zorder = 2)
# plt.plot(axis_x, -edge_graph,   label = 'Edge Only', zorder=1)
# plt.plot(axis_x, -md_graph,     label = 'MD Only', zorder=1)
# plt.plot(axis_x, -head_graph,   label = 'Static', zorder=1)
# plt.plot(axis_x, -random_graph,   label = 'RANDOM', zorder=1)


plt.xlabel('Episodes')
plt.ylabel('Cost')
#
# #plt.title('Client model bit')
plt.legend(loc='center right',edgecolor='black',fontsize=14,fancybox=False)#bbox_to_anchor=(0.9,0.0)
#plt.savefig('Reward.eps',dpi=200,transparent=True,bbox_inches='tight')
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

