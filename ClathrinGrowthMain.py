########################################################################
# Simulation of the growth of clathrin lattices 					   #
# - written by Felix Frey, Heidelberg University,	2019-2020		   #
#																	   #
# This script provides the main function and the important growth	   # 
# parameters to simulate the growth of clathrin lattices   	           #
########################################################################



mode="km"	# Binding modes km: knee model, am: ankle model, hm: 
			# heel model, ahm: ankle-heel model, tm: toe model
			# kam: knee-ankle model
input_data="EM" # Used input area distribution:
#"EM": distribution deduced from EM, "CLEM": distribution deduced from CLEM
number_of_clusters=1000 # Number of simulation runs
show_lattice= True # Option to visualize the growing lattice 
factor=17.5 # Scaling factor: 1 python unit corresponds to 17.5 nm


import os

# Create output folder
print "Create output folder"
if not os.path.isdir("output"):
   os.makedirs("output")

# Import  
import ClathrinGrowthFunctions as CGF
import random
import numpy as np

# Initialize lists to store the data
area=[]
perimeter=[]
structural_gaps=[]
nodes=[]
solidity=[]
aspect_ratio=[]
maxsize_size=120000 # cutoff area of clathrin lattices, to ensure that
					# the algorithm will terminate

# main loop
for j in range(number_of_clusters):
	print "------------------"
	print "------------------"
	print "run number: " +str(j)
	print "------------------"
	print "------------------"
	HexLattice=CGF.InitializeLattice(10*3,10*3) # Set the size of the clathrin lattice
	random_x=5*3 # Initial x,y coordinates of the lattice 
	random_y=10*3
	i=0
	while True: # Reject very large lattices, since it is computationally too costly
		if input_data=="EM":
			sample_size=np.int(np.random.lognormal(mean=9.8,sigma=0.48)) # EM area distribution
		if input_data=="CLEM":	
			sample_size=np.int(np.random.lognormal(mean=10.4,sigma=0.51)) # CLEM area distribution
		if sample_size<maxsize_size: 
				break	
							
	while True:
		HexLattice=CGF.OccupyTriskelion(random_x,random_y,HexLattice,mode) # Occupy lattice
		i+=1
		clathrin_legs = [n for n,v in HexLattice.nodes(data=True) if v['binding_sites'] == 'true'] # Compute all binding sites
		random_choice=random.choice(clathrin_legs) # Choose next bining site
		random_x= random_choice[0]
		random_y= random_choice[1]

		CGF.AnalyzeEdgeLength(HexLattice) # Find all edge nodes of th lattice

		if(CGF.FastArea(HexLattice)*factor**2>sample_size): # Check lattice size and analyse lattice parameters
			
			measures=CGF.LatticeParameters(HexLattice) 
			area.append(measures[1]*factor**2)
			perimeter.append(measures[0]*factor)
			structural_gaps.append(CGF.StructuralGap(HexLattice))	
			nodes.append(i)
			solidity.append(measures[3])
			aspect_ratio.append(measures[4])
			if show_lattice:
				CGF.PlotLattice(HexLattice)	
			break
				
		#CGF.PlotLattice(HexLattice) # show growth of lattices
# Save data		
np.savetxt('output/Histo'+str(mode)+str(input_data)+'.csv', (area,perimeter,structural_gaps,nodes,solidity,aspect_ratio))
