import numpy as np
from matplotlib import pyplot as plt
import math

axis_x = np.array([4,8,12,16,20])

rand_graph =[]
static_graph = []
md_graph = []
edge_graph = []
ddpg_graph =[]

for i in range(4):
    ddpg = np.load('DDPG_TC' + str(i) + '.npy')
    ddpg_graph.append(np.mean(ddpg))
    static_graph.append(np.mean(np.load('Head_TC'+str(i)+'.npy')))
    md_graph.append(np.mean(np.load('MD_TC'+str(i)+'.npy')))
    edge_graph.append(np.mean(np.load('Edge_TC'+str(i)+'.npy')))
    rand_graph.append(np.mean(np.load('RAND_TC'+str(i)+'.npy')))
i = 4
ddpg = np.load('DDPG_TC0.5_' + str(i) + '.npy')

ddpg_graph.append(np.mean(ddpg))
static_graph.append(np.mean(np.load('Head_TC0.5_'+str(i)+'.npy')))
md_graph.append(np.mean(np.load('MD_TC0.5_'+str(i)+'.npy')))
edge_graph.append(np.mean(np.load('Edge_TC0.5_'+str(i)+'.npy')))
rand_graph.append(np.mean(np.load('RAND_TC0.5_'+str(i)+'.npy')))



# for i in range(6):
#     for j in range(i):
#         np.delete(opti_graph[i],0)
#
# acc_graph = np.max(acc_graph, axis=0)
# opti_graph = np.max(opti_graph, axis=0)


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"]="stix"
plt.rcParams["figure.figsize"] = (8,6)
plt.grid(linestyle='dotted')
plt.ylabel('Completion Time',fontsize=18)
plt.xlabel('Bandwidth',fontsize=18) #수식 : r'\수식내용'
plt.tick_params(axis='x', direction='in',top=True,bottom=True)
plt.tick_params(axis='y', direction='in',left=True,right=True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

markersize=8
plt.plot(axis_x, -np.array(ddpg_graph)/2000, label = 'DS2CF', zorder=2,marker='o',color='black',markerfacecolor='white',markersize=markersize)
plt.plot(axis_x, -np.array(static_graph)/2000, label = 'SPLIT', zorder=2,marker='*',color='black',markerfacecolor='white',markersize=markersize)
plt.plot(axis_x, -np.array(rand_graph)/2000, label = 'RANDOM', zorder=2,marker='^',color='black',markerfacecolor='white',markersize=markersize)
plt.plot(axis_x, -np.array(edge_graph)/2000, label = 'EDGE', zorder=2,marker='x',color='black',markerfacecolor='white',markersize=markersize)
plt.plot(axis_x, -np.array(md_graph)/2000, label = 'MD', zorder=2,marker='P',color='black',markerfacecolor='white',markersize=markersize)

# plt.plot(axis_x, rand_graph, label = 'RAND', zorder=1)
# plt.plot(axis_x, static_graph, label = 'STATIC', zorder=1)

# plt.xlabel('Iterations')
# plt.ylabel('Reward')
#
# #plt.title('Client model bit')
plt.xlabel('$\omega$')
plt.ylabel('Average Task Completion Time (s)')
plt.legend(loc='best',edgecolor='black',fontsize=14,fancybox=False)#bbox_to_anchor=(0.9,0.0)
plt.savefig('g.eps',dpi=200,transparent=True,bbox_inches='tight')
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

