import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
fig,ax=plt.subplots(figsize=(10,3.7));ax.set_xlim(0,10);ax.set_ylim(0,3.7);ax.axis('off')
def box(x,y,w,h,title,text,color='#e6eef3'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#3f657b',lw=1.2));ax.text(x+w/2,y+h*.72,title,ha='center',va='center',fontsize=10,weight='bold');ax.text(x+w/2,y+h*.32,text,ha='center',va='center',fontsize=8)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.3,color='#43515b'))
box(.15,2,2.65,1.3,'Causal operational state','Stock, backlog, demand, orders\nTyped edges and time context')
box(3.25,2,3.0,1.3,'Relation-aware encoder','Three graph attention layers\nSeparate self-state pathway')
box(6.75,2,3.0,1.3,'Two-cycle forecasts','Occurrence probability p\nConditional shortage q')
arrow((2.85,2.65),(3.17,2.65));arrow((6.33,2.65),(6.67,2.65))
box(6.75,.12,3,.95,'Fixed emergency rule','Shared capacity; purchases arrive t+2','#f8eadf')
box(.15,.12,4.9,.95,'Material-conserving inventory system','Common demand and disruptions; cost and service accounting','#e4eee3')
arrow((8.25,1.91),(8.25,1.15));arrow((6.67,.60),(5.13,.60));arrow((1.5,1.15),(1.5,1.91))
fig.savefig('paper/architecture.pdf',bbox_inches='tight');plt.close(fig)
