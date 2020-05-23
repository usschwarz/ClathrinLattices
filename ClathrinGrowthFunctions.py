########################################################################
# Simulation of the growth of clathrin lattices 					   #
# - written by Felix Frey, Heidelberg University,  2019-2020		   #
#																	   #
# This script provides all functions that are needed to simulate and   #
# analyze the growth of clathrin lattices.							   #
########################################################################


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from concorde.tsp import TSPSolver
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import EllipseFitter as ef
import shapely.geometry 


# Function, that initializes the hexagonal lattice on
# which clathrin triskelia are placed in order to simulate
# the growth of clathrin lattices.
#
# First, an empty lattice is created.
# Second, for every lattice vertex the following attributes are initialized
# and set to false:
#
# 'node': which tells whether a hub of a triskelion is present or not
# 'clathrin': which tells whether a triskelion knee or ankle is present 
# 'gap': which tells whether a structural gap is present 
# 'edge': which tells whether the lattice vertex is at the edge of the lattice
# 'binding_site': which tells whether a new triskelion can bind to this lattice vertex 


def InitializeLattice(size_x,size_y):	
	Lattice = nx.hexagonal_lattice_graph(size_x, size_y, periodic=False, with_positions=False, create_using=None)
	
	allnodes= list(nx.nodes(Lattice))
	
	for node in allnodes:
		Lattice.node[node]['node'] = 'false'
		Lattice.node[node]['clathrin'] = 'false'
		Lattice.node[node]['gap'] = 'false'
		Lattice.node[node]['edge'] = 'false'
		Lattice.node[node]['binding_sites'] = 'false'

	return Lattice
	
	
# Function, that calculates all binding sites from a chosen triskelion
# hub in the ankle model
		
def NextNextNeighbor(node,Lattice):
	
	neighbors= list(nx.all_neighbors(Lattice,node))
	next_neighbors=[]

	#exclude node in double counting
	#idea, define a direction: axis-node-neighbor
	#take next neighbor and decide relative to the axis if of if not
		 
	for neighbor_i in neighbors:
		left_right_neighbor=[]
		for j in range(3):
			if list(nx.all_neighbors(Lattice,neighbor_i))[j]!=node:				
				left_right_neighbor.append(list(nx.all_neighbors(Lattice,neighbor_i))[j])
			
		#x-axis goes to the left then take neighbor with larger y
		if(neighbor_i[0]-node[0]<0):
			if left_right_neighbor[0][1]>left_right_neighbor[1][1]:
				next_neighbors.append(left_right_neighbor[0])
			else:
				next_neighbors.append(left_right_neighbor[1])
				
		
		#x-axis goes to the right then take neighbor with smaller y	
		if(neighbor_i[0]-node[0]>0):
			if left_right_neighbor[0][1]<left_right_neighbor[1][1]:
				next_neighbors.append(left_right_neighbor[0])
			else:
				next_neighbors.append(left_right_neighbor[1])	
		# cannot decide based on x-axis, consdier y-axis
		else:	
			#y-axis goes to the bottom then take neighbor with smaller x
			if(neighbor_i[1]-node[1]<0):
				if left_right_neighbor[0][0]>left_right_neighbor[1][0]:
					next_neighbors.append(left_right_neighbor[1])
				else:
					next_neighbors.append(left_right_neighbor[0])
					
			#y-axis goes to the top then take neighbor with larger x
			if(neighbor_i[1]-node[1]>0):
				if left_right_neighbor[0][0]<left_right_neighbor[1][0]:
					next_neighbors.append(left_right_neighbor[1])
				else:
					next_neighbors.append(left_right_neighbor[0])	


	return next_neighbors
	
	

# Function, that calculates all binding sites from a chosen triskelion
# hub in the heel model

def AntiNextNextNeighbor(node,Lattice):
	
	neighbors= list(nx.all_neighbors(Lattice,node))
	next_neighbors=[]
		 
	for neighbor_i in neighbors:
		left_right_neighbor=[]
		for j in range(3):
			if list(nx.all_neighbors(Lattice,neighbor_i))[j]!=node:				
				left_right_neighbor.append(list(nx.all_neighbors(Lattice,neighbor_i))[j])
			
		#x-axis goes to the left then take neighbor with smaller y
		if(neighbor_i[0]-node[0]<0):
			if left_right_neighbor[0][1]>left_right_neighbor[1][1]:
				next_neighbors.append(left_right_neighbor[1])
			else:
				next_neighbors.append(left_right_neighbor[0])
				
		
		#x-axis goes to the right then take neighbor with larger y	
		if(neighbor_i[0]-node[0]>0):
			if left_right_neighbor[0][1]<left_right_neighbor[1][1]:
				next_neighbors.append(left_right_neighbor[1])
			else:
				next_neighbors.append(left_right_neighbor[0])	
		# cannot decide based on x-axis, consdier y-axis
		else:	
			#y-axis goes to the bottom then take neighbor with larger x
			if(neighbor_i[1]-node[1]<0):
				if left_right_neighbor[0][0]>left_right_neighbor[1][0]:
					next_neighbors.append(left_right_neighbor[0])
				else:
					next_neighbors.append(left_right_neighbor[1])
					
			#y-axis goes to the top then take neighbor with smaller x
			if(neighbor_i[1]-node[1]>0):
				if left_right_neighbor[0][0]<left_right_neighbor[1][0]:
					next_neighbors.append(left_right_neighbor[0])
				else:
					next_neighbors.append(left_right_neighbor[1])	


	return next_neighbors


# Function, that calculates all binding sites from a chosen triskelion
# hub in the knnee-and-ankle model
		
def NextNeighborsKAM(node,Lattice):
	
	neighbors= list(nx.all_neighbors(Lattice,node))
	next_neighbors=[]

	#exclude node in double counting
	#idea, define a direction: axis-node-neighbor
	#take next neighbor and decide relative to the axis if of if not
		 
	for neighbor_i in neighbors:
		left_right_neighbor=[]
		for j in range(3):
			if list(nx.all_neighbors(Lattice,neighbor_i))[j]!=node:				
				left_right_neighbor.append(list(nx.all_neighbors(Lattice,neighbor_i))[j])
			
		#x-axis goes to the left then take neighbor with larger y
		if(neighbor_i[0]-node[0]<0):
			if left_right_neighbor[0][1]>left_right_neighbor[1][1]:
				next_neighbors.append(left_right_neighbor[0])
			else:
				next_neighbors.append(left_right_neighbor[1])
				
		
		#x-axis goes to the right then take neighbor with smaller y	
		if(neighbor_i[0]-node[0]>0):
			if left_right_neighbor[0][1]<left_right_neighbor[1][1]:
				next_neighbors.append(left_right_neighbor[0])
			else:
				next_neighbors.append(left_right_neighbor[1])	
		# cannot decide based on x-axis, consdier y-axis
		else:	
			#y-axis goes to the bottom then take neighbor with smaller x
			if(neighbor_i[1]-node[1]<0):
				if left_right_neighbor[0][0]>left_right_neighbor[1][0]:
					next_neighbors.append(left_right_neighbor[1])
				else:
					next_neighbors.append(left_right_neighbor[0])
					
			#y-axis goes to the top then take neighbor with larger x
			if(neighbor_i[1]-node[1]>0):
				if left_right_neighbor[0][0]<left_right_neighbor[1][0]:
					next_neighbors.append(left_right_neighbor[1])
				else:
					next_neighbors.append(left_right_neighbor[0])	


	return next_neighbors+neighbors



# Function, that calculates all binding sites from a chosen triskelion
# hub in the ankle-heel model, not used in the manuscript.

def AnkleAntiAnkleNeighbors(node,Lattice):
	
	neighbors= list(nx.all_neighbors(Lattice,node))
	next_neighbors=[]
	 
	for neighbor_i in neighbors:
		for j in range(3):
			if list(nx.all_neighbors(Lattice,neighbor_i))[j]!=node:				
				next_neighbors.append(list(nx.all_neighbors(Lattice,neighbor_i))[j])
	
	return next_neighbors

# Function, that calculates all binding sites from a chosen triskelion
# hub in the toe model

def ToeNeighbors(node,Lattice):
	
	neighbors= list(nx.all_neighbors(Lattice,node))
	next_neighbors=NextNextNeighbor(node,Lattice)
	toe_neighbors=[]

		 
	for i in range(3):
		node=neighbors[i]
		neighbor_i=next_neighbors[i]
		left_right_neighbor=[]
		for j in range(3):
			if list(nx.all_neighbors(Lattice,neighbor_i))[j]!=node:				
				left_right_neighbor.append(list(nx.all_neighbors(Lattice,neighbor_i))[j])
			
		#x-axis goes to the left then take neighbor with larger y
		if(neighbor_i[0]-node[0]<0):
			if left_right_neighbor[0][1]>left_right_neighbor[1][1]:
				toe_neighbors.append(left_right_neighbor[0])
			else:
				toe_neighbors.append(left_right_neighbor[1])
				
		
		#x-axis goes to the right then take neighbor with smaller y	
		if(neighbor_i[0]-node[0]>0):
			if left_right_neighbor[0][1]<left_right_neighbor[1][1]:
				toe_neighbors.append(left_right_neighbor[0])
			else:
				toe_neighbors.append(left_right_neighbor[1])	
		# cannot decide based on x-axis, consdier y-axis
		else:	
			#y-axis goes to the bottom then take neighbor with smaller x
			if(neighbor_i[1]-node[1]<0):
				if left_right_neighbor[0][0]>left_right_neighbor[1][0]:
					toe_neighbors.append(left_right_neighbor[1])
				else:
					toe_neighbors.append(left_right_neighbor[0])
					
			#y-axis goes to the top then take neighbor with larger x
			if(neighbor_i[1]-node[1]>0):
				if left_right_neighbor[0][0]<left_right_neighbor[1][0]:
					toe_neighbors.append(left_right_neighbor[1])
				else:
					toe_neighbors.append(left_right_neighbor[0])	


	return toe_neighbors


# Function that returns all nodes which belong to the same
# clathrin triskelion, input is the center of the triskelion
# nodes defined with (x,y) where x left right and y top down

def Triskelion(node,Lattice):
	Lattice.node[node]['node'] = 'true'
	Lattice.node[node]['clathrin'] = 'false'
	Lattice.node[node]['binding_sites'] = 'false'

	neighbors= list(nx.all_neighbors(Lattice,node))	
	next_neighbors=NextNextNeighbor(node,Lattice)
	triseklion_nodes=next_neighbors+neighbors
	
	return [triseklion_nodes,Lattice]

# Function that returns the legs of all triskelia that build up the
# clathrin lattice

def TriskelionEdges(clathrin_nodes,Lattice):
	triskelion_list=[]
	
	for node in clathrin_nodes:
		neighbors= list(nx.all_neighbors(Lattice,node))
		next_neighbors=NextNextNeighbor(node,Lattice)	 	
		triskelion=next_neighbors+neighbors+[node]
		ClathrinTriskelion=Lattice.subgraph(triskelion)
		triskelion_list.append(ClathrinTriskelion.edges())
		
	return triskelion_list


# Function that puts a new triskelion onto the lattice according to
# chosen binding the mode

def OccupyTriskelion(x,y,Lattice,mode):
	triskelion,Lattice=Triskelion((x,y),Lattice)
	for i in triskelion:
		if Lattice.node[i]['node']!='true':
			Lattice.node[i]['clathrin'] = 'true'

	if mode=="km":
		for i in list(nx.all_neighbors(Lattice,(x,y))):
			if Lattice.node[i]['node']!='true':
				Lattice.node[i]['binding_sites'] = 'true'
	
	if mode=="am":
		for i in NextNextNeighbor((x,y),Lattice):
			if Lattice.node[i]['node']!='true':
				Lattice.node[i]['binding_sites'] = 'true'
				
	if mode=="kam":
		for i in NextNeighborsKAM((x,y),Lattice):
			if Lattice.node[i]['node']!='true':
				Lattice.node[i]['binding_sites'] = 'true'				
				
	if mode=="hm":
		for i in AntiNextNextNeighbor((x,y),Lattice):
			if Lattice.node[i]['node']!='true':
				Lattice.node[i]['binding_sites'] = 'true'					

	if mode=="ahm":	
		for i in AnkleAntiAnkleNeighbors((x,y),Lattice):
			if Lattice.node[i]['node']!='true':
				Lattice.node[i]['binding_sites'] = 'true'

	if mode=="tm":	
		for i in ToeNeighbors((x,y),Lattice):
			if Lattice.node[i]['node']!='true':
				Lattice.node[i]['binding_sites'] = 'true'								
							
	return Lattice

# Function that defines the edges of the clathrin lattice.
#
# Only knee or ankle lattice nodes can be edge sites. Clathrin hubs
# can never be edge sites.
# A lattice node is defined edge=false, if all neighbors are either 
# a knee, ankle or hub node. Otherwise a node is defined edge=true

# (1) all nodes are set to edge=false
# (2) all ankle and knee nodes without hub are identified.
# (3) all neighbors of those nodes are identified.
# (4) all nodes with empty neighbors are called edges.

def AnalyzeEdgeLength(Lattice):
	
	allnodes= list(nx.nodes(Lattice))
	for node in allnodes:
		Lattice.node[node]['edge'] = 'false'
	
	leg_sites=[n for n,v in Lattice.nodes(data=True) if v['clathrin'] == 'true']
	for leg_site in leg_sites:
		neighbors= list(nx.all_neighbors(Lattice,leg_site))
		for neighbor in neighbors:
			if(Lattice.node[neighbor]['clathrin']=='false' and Lattice.node[neighbor]['node']=='false'):
				Lattice.node[leg_site]['edge']='true'


# Function that calculates the area of a clathrin lattice. Therefore, 
# the edge sites of the clathrin lattice are interpreted as a polygon.
# Next, the circumference of that polygon is defined by solving 
# a travelling salesman problem (TSP) for the edge sites, i.e. by traversing 
# the edge sites by the shortest route. For this purpose we use the
# python implementaion of the concorde TSP solver. 
# From this outline we then calculate the area of the clathrin lattice
# by the sektorformel of Leibniz.
	
def FastArea(Lattice):
	clathrin_edges = [n for n,v in Lattice.nodes(data=True) if v['edge'] == 'true' ]  
	ClathrinEdges=Lattice.subgraph(clathrin_edges)
	
	edge_sites = np.asarray(nx.get_node_attributes(ClathrinEdges, 'pos').values(),dtype=np.float)
	solver = TSPSolver.from_data(edge_sites[:,0],edge_sites[:,1],norm="EUC_2D")	
	tour_data = solver.solve(time_bound = 60.0, verbose = False, random_seed = 42) 
	route=tour_data.tour
	
	new_edge_sites_order = np.concatenate((np.array([edge_sites[route[i]] for i in range(len(route))]),np.array([edge_sites[0]])))
	
	x=new_edge_sites_order[:,0]
	y=new_edge_sites_order[:,1]	
	area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
	
	return np.abs(area)	
	
# Function that first calculates the area of the clathrin lattice 
# similar to the function 'FastArea'. Then, the circumference, 
# circularity, solidity and ellipse parameters of the clathrin lattice 
# are calculated.
	
def LatticeParameters(Lattice):
	clathrin_edges = [n for n,v in Lattice.nodes(data=True) if v['edge'] == 'true' ]  
	ClathrinEdges=Lattice.subgraph(clathrin_edges)
	
	edge_sites = np.asarray(nx.get_node_attributes(ClathrinEdges, 'pos').values(),dtype=np.float)
	solver = TSPSolver.from_data(edge_sites[:,0],edge_sites[:,1],norm="EUC_2D")	
	tour_data = solver.solve(time_bound = 60.0, verbose = False, random_seed = 42) 
	route=tour_data.tour
	
	new_edge_sites_order = np.concatenate((np.array([edge_sites[route[i]] for i in range(len(route))]),np.array([edge_sites[0]])))
	
	perimeter=PolygonPerimeter(route,edge_sites)
	area=PolygonArea(new_edge_sites_order)
	circularity = PolygonCircularity(perimeter,area)
	solidity=PolygonSolidity(new_edge_sites_order,area)
	
	params=PolygonEllipse(new_edge_sites_order)
	aspect_ratio=params[1]/params[2]

	return [perimeter,area,circularity,solidity,aspect_ratio]	
	
# Function that calculates the cicumference of the clathrin lattice by
# taking the sum of all polygon edges

def PolygonPerimeter(route,edge_sites):
	
	perimeter=0
	
	for i in range(len(route)):
		perimeter+=np.linalg.norm(edge_sites[route[i]]-edge_sites[route[i-1]])
	
	return perimeter

# Function that calculates the area of the clathrin lattice 
# from the outline by the sektorformel of Leibniz.	

	
def PolygonArea(edge_sites):
	x=edge_sites[:,0]
	y=edge_sites[:,1]	
	area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
	
	return np.abs(area)

# Function that calculates the circularity of the clathrin lattice.	
	
def PolygonCircularity(perimeter,area):
	return 4*np.pi*(area/(perimeter**2))	
	
# Function that calculates the solidity of the clathrin lattice. 
# Therefore, the convex hull of the polygon (defined by the edge nodes
# of the clathrin lattice) is calculated. 	

def PolygonSolidity(edge_sites,area):
	
	hull = ConvexHull(edge_sites)
	hull_vertices=np.append(hull.vertices,hull.vertices[0])
	x_hull=edge_sites[hull_vertices,0]
	y_hull=edge_sites[hull_vertices,1]
	
	convex_area=0.5*np.sum(y_hull[:-1]*np.diff(x_hull) - x_hull[:-1]*np.diff(y_hull))
	convex_area=np.abs(convex_area)

	return area/convex_area

# Function that fits an ellipse to the polygon (defined by the edge nodes
# of the clathrin lattice).
# First, the polygon is transformed into a binary array or a mask.
# Secondly, the ImageJ "EllipseFitter" method is used on that binary
# array in order to fit an ellipse to the polygon and find all ellipse
# parameters.
# Note: The center of the ellipse is shifted back to python coordinates
# from the Image J coordinates. This is only important to plot the ellipse
# correctly.
	
def PolygonEllipse(edge_sites):

	dim_x, dim_y=200,200
	mask=np.zeros((dim_x,dim_y))

	poly = shapely.geometry.Polygon(edge_sites)
	for i in range(dim_x):
		for j in range (dim_y):
			point = shapely.geometry.Point(i,j)
			if point.intersects(poly):
				mask[i,j]=True

	phi, major, minor,center_x,center_y  = ef.EllipseFitter(mask,usePrint=False)
	
	#reorient
	phi=-phi

	#transfomation python <-> ImageJ cf. line 200+201 in the source code of EllipseFitter
	center_x=center_x-0.5
	center_y=center_y-0.5
	
	return [phi, major, minor,center_x,center_y]


	
# Function that defines the structural gaps of the clathrin lattice.
#
# Only knee or ankle lattice nodes can be structural gaps sites. 
# A structural gap is a knee or ankle node which could be populated by
# a triskelion without changing the number of visible clathrin legs 
# within the lattice.			

# First, check if a edges of a questionable node share all edges
# with the already existing lattice. If all edges all already shared, the
# algorithm has found a structural gap.
# Second, sum all gaps.

def StructuralGap(Lattice):

	counter=0
	clathrin_nodes = [n for n,v in Lattice.nodes(data=True) if v['node'] == 'true']  
	clatrhin_legs=[n for n,v in Lattice.nodes(data=True) if v['clathrin'] == 'true']
	lattice_edges=TriskelionEdges(clathrin_nodes,Lattice)
	lattice_edges = [item for sublist in lattice_edges for item in sublist]
	lattice_edges_reversed = [item[::-1] for item in lattice_edges]

	for leg_site in clatrhin_legs:
		questionable_edge=TriskelionEdges([leg_site],Lattice)
		#as edges have directonality one has to check the edges and
		#a reversed version of all edges.
		questionable_edge = [item for sublist in questionable_edge for item in sublist]
		vec = [e in lattice_edges+lattice_edges_reversed for e in questionable_edge]
		#if all edges are within the lattice mark as gap
		if np.count_nonzero(vec)==6:
			counter+=1
			Lattice.node[leg_site]['gap'] = 'true'			
			
	return counter					

######################################################################## 
# 						Plot and print functions 					   # 
######################################################################## 

# Function that print the important parameters

def PrintParameters(params):
	
	perimeter=params[0]
	area=params[1]
	circularity=params[2]
	solidity=params[3]
	major=params[4]
	minor=params[5]

	print	
	print "Solidity: " + str(solidity) 
	print 
	print "Aspect ratio: " +str(major/minor)
	print 
	print "Roundness=Inverse aspect ratio: " + str(minor/major)
	print
	print "Area sektorformel: " + str(area) + " Area ellipse: " + str(np.pi*major*minor/4.)
	print
	print "Perimeter: " + str(perimeter) 
	print
	print "Circularity: " + str(circularity) 
	
# Function that plots triskelia

def PlotTriskelion(clathrin_nodes,Lattice):
	
	# Define colors 
	new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
				  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
				  '#bcbd22', '#17becf']

	# Plot triskelia
	for node in clathrin_nodes:
		neighbors= list(nx.all_neighbors(Lattice,node))
		next_neighbors=NextNextNeighbor(node,Lattice)
		 
		triskelion=next_neighbors+neighbors+[node]
		ClathrinTriskelion=Lattice.subgraph(triskelion)
		nx.draw_networkx_edges(ClathrinTriskelion, nx.get_node_attributes(ClathrinTriskelion, 'pos'),width=4 ,font_size=2,edge_color=new_colors[0])
	
# Function that plots the hexagonal lattice, all triskelia, gaps, edges,
# the fitted ellipse and the convex hull.
	
def PlotLattice(Lattice):
	
	# Define colors and figure
	new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
				  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
				  '#bcbd22', '#17becf']


	
	fig = plt.figure(figsize=(10,10))
	sub1=plt.subplot(111)
	sub1.set_aspect("equal")
	
	# Clathrin nodes
	clathrin_nodes = [n for n,v in Lattice.nodes(data=True) if v['node'] == 'true']  
	ClathrinNodes=Lattice.subgraph(clathrin_nodes)
	
	# Clathrin edges
	clathrin_edges = [n for n,v in Lattice.nodes(data=True) if v['edge'] == 'true' ]  
	ClathrinEdges=Lattice.subgraph(clathrin_edges)
	
	# Clathrin structural gaps
	clathrin_gaps = [n for n,v in Lattice.nodes(data=True) if v['gap'] == 'true']  
	ClathrinGaps=Lattice.subgraph(clathrin_gaps)
	
	# Clathrin travelling salesman problem
	edge_sites = np.asarray(nx.get_node_attributes(ClathrinEdges, 'pos').values(),dtype=np.float)
	solver = TSPSolver.from_data(edge_sites[:,0],edge_sites[:,1],norm="EUC_2D")	
	tour_data = solver.solve(time_bound = 60.0, verbose = False, random_seed = 42) 
	route=tour_data.tour
	new_edge_sites_order = np.concatenate((np.array([edge_sites[route[i]] for i in range(len(route))]),np.array([edge_sites[0]])))
	
	# Plot lattice and all structures
	plt.plot(new_edge_sites_order[:,0],new_edge_sites_order[:,1],color=new_colors[1])
	nx.draw_networkx_edges(Lattice, nx.get_node_attributes(Lattice, 'pos'), width=3,node_color="k",alpha=0.5)
	PlotTriskelion(clathrin_nodes,Lattice)
	nx.draw_networkx_nodes(ClathrinNodes, nx.get_node_attributes(ClathrinNodes, 'pos'), node_size=60,font_size=2,node_color=new_colors[0], label="clathrin")
	nx.draw_networkx_nodes(ClathrinGaps, nx.get_node_attributes(ClathrinGaps, 'pos'), node_size=60,font_size=2,node_color=new_colors[3], label="clathrin gap")
	nx.draw_networkx_nodes(ClathrinEdges, nx.get_node_attributes(ClathrinEdges, 'pos'),node_size=60,node_color=new_colors[1],label="clathrin edge")
	
	# Plot convex hull
	hull = ConvexHull(new_edge_sites_order)
	for simplex in hull.simplices:
		plt.plot(new_edge_sites_order[simplex, 0], new_edge_sites_order[simplex, 1], 'k-')

	# Perimeter, Area, Circularity, Solidity, Ellipse
	perimeter=PolygonPerimeter(route,edge_sites)
	area=PolygonArea(new_edge_sites_order)
	circularity =PolygonCircularity(perimeter,area)		
	solidity=PolygonSolidity(new_edge_sites_order,area)
	phi, major, minor,center_x,center_y	= PolygonEllipse(new_edge_sites_order)
	
	# Print parameters
	PrintParameters([perimeter,area,circularity,solidity,major,minor])

	# Ellipse
	my_ellipse = Ellipse((center_x,center_y),major,minor,phi)
	ax=plt.gca()
	ax.add_patch(my_ellipse)
	my_ellipse.set(clip_box=ax.bbox,alpha=0.5, color=new_colors[1])
	ax.add_artist(my_ellipse)
	
	
	plt.axis('off')
	plt.legend(loc="upper left",numpoints=1,scatterpoints=1,fontsize=20)
	plt.savefig("output/lattice.pdf")
	plt.waitforbuttonpress(0) 
	plt.close(fig)
	
	
