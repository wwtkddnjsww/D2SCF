import numpy as np
from matplotlib import pyplot as plt
import math

axis_x = np.array([1,5,10,15,20,25,30,35,40,45,50])


rand_graph =[]
static_graph = []
md_graph = []
edge_graph = []
ddpg_graph =[]
ddpg_min = []
ddpg_max = []
edge_min = []
edge_max = []
md_min = []
md_max = []
static_min = []
static_max = []
rand_min = []
rand_max = []

for i in range(len(axis_x)):
    ddpg = np.load('DDPG_TC0.1_' + str(i) + '.npy')
    ddpg_graph.append(np.mean(ddpg))
    ddpg_min.append(np.min(np.load('DDPG_TC0.1_' + str(i) + '.npy')))
    ddpg_max.append(np.max(np.load('DDPG_TC0.1_' + str(i) + '.npy')))

    static_graph.append(np.mean(np.load('Head_TC0.1_'+str(i)+'.npy')))
    static_min.append(np.min(np.load('Head_TC0.1_'+str(i)+'.npy')))
    static_max.append(np.max(np.load('Head_TC0.1_' + str(i) + '.npy')))

    md_graph.append(np.mean(np.load('MD_TC0.1_'+str(i)+'.npy')))
    md_min.append(np.min(np.load('MD_TC0.1_'+str(i)+'.npy')))
    md_max.append(np.max(np.load('MD_TC0.1_'+str(i)+'.npy')))

    edge_graph.append(np.mean(np.load('Edge_TC0.1_'+str(i)+'.npy')))
    edge_min.append(np.min(np.load('Edge_TC0.1_' + str(i) + '.npy')))
    edge_max.append(np.max(np.load('Edge_TC0.1_' + str(i) + '.npy')))

    rand_graph.append(np.mean(np.load('RAND_TC0.1_'+str(i)+'.npy')))
    rand_min.append(np.min(np.load('RAND_TC0.1_' + str(i) + '.npy')))
    rand_max.append(np.max(np.load('RAND_TC0.1_' + str(i) + '.npy')))




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
plt.ylabel('Average Task Completion Time (s)',fontsize=18)
plt.xlabel('Bandwidth',fontsize=18) #수식 : r'\수식내용'
plt.tick_params(axis='x', direction='in',top=True,bottom=True)
plt.tick_params(axis='y', direction='in',left=True,right=True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

markersize=8
plt.errorbar(axis_x,-np.array(ddpg_graph), yerr=[-np.array(ddpg_graph)+np.array(ddpg_max),np.array(ddpg_graph)-np.array(ddpg_min)],
            label = 'D2SCF', zorder=2,marker='o',color='royalblue',markerfacecolor='white',markersize=markersize)

plt.errorbar(axis_x, -np.array(static_graph), yerr=[-np.array(static_graph)+np.array(static_max),np.array(static_graph)-np.array(static_min)],
             label = 'SPLIT', zorder=-1,marker='*',color='olive',markerfacecolor='white',markersize=markersize)
#
plt.errorbar(axis_x, -np.array(rand_graph),yerr=[-np.array(rand_graph)+np.array(rand_max),np.array(rand_graph)-np.array(rand_min)],
             label = 'RANDOM', zorder=-2,marker='^',color='darkorange',markerfacecolor='white',markersize=markersize)
#
plt.errorbar(axis_x, -np.array(edge_graph), yerr=[-np.array(edge_graph)+np.array(edge_max),np.array(edge_graph)-np.array(edge_min)],
             label = 'EDGE', zorder=-2,marker='x',color='indianred',markerfacecolor='white',markersize=markersize)
#
plt.errorbar(axis_x, -np.array(md_graph), yerr=[-np.array(md_graph)+np.array(md_max),np.array(md_graph)-np.array(md_min)],
             label = 'MD', zorder=-2,marker='P',color='mediumpurple',markerfacecolor='white',markersize=markersize)

# plt.plot(axis_x, rand_graph, label = 'RAND', zorder=1)
# plt.plot(axis_x, static_graph, label = 'STATIC', zorder=1)

# plt.xlabel('Iterations')
# plt.ylabel('Reward')
#
# #plt.title('Client model bit')
# plt.ylim(5,30)
plt.xlabel('$D$')
plt.ylabel('Task Completion Time (s)')
plt.legend(bbox_to_anchor=(0.3,0.4),edgecolor='black',fontsize=14,fancybox=False)
plt.savefig('../g_errorbar.eps',dpi=200,transparent=True,bbox_inches='tight')
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

