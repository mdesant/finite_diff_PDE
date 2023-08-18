#Finite difference method (The Journal of Chemical Physics 47, 454 (1967)) for 
#two electrons in an infinite square well
import numpy as np
import argparse



def main(num,a,h,eigvec,debug=False):
   #diagonal of the H matrix
   Hdiag=[]
   gridlist=[]
   idcol= []
   x2=h

   for k in range(2,num-1):
       idcol.append(k)

   mesh=np.zeros((num,num))

   p=0  # row index
   node=1
   
   for i in range(-2,-num+1,-1):
       #print(i)
       #print(mesh[i,:])
    
       idx=np.array(idcol)[p:]
       p+=1
       for j in idx:
           mesh[i,j]=node
           node+=1
           x1=j*h
           gridlist.append((x1,x2))   # only internal points 
           Hdiag.append(-4.0 -2.0*h*h/abs(x1-x2))
       x2+=h
   
   if debug:    
       print("show mesh")
       print(mesh)
   #counting the 1 entries (non-zeros nodes)
   counter=0
   for i in range(num):
       for j in range(num):
         if mesh[i,j] != 0:
             counter+=1
   print("number of interior points: %i\n" % counter)

   H=np.zeros((counter,counter))

   row=np.zeros(counter)
   ####################################


   #off diagonal elements

   #restore idcol and p index
   idcol=[]
   p=0
   #print("p :%i" % p)
   for k in range(2,num-1):
       idcol.append(k)
   for i in range(-2,-num+1,-1):
   #    print(i)
       idx=np.array(idcol)[p:]
       p+=1
       #print(idx) uncomment for debug
       for j in idx:
           if debug:
               print("I am node %i" % mesh[i,j])
           #four neighbour n1,n2,n3,n4
           n1=int(mesh[i,j+1])
           n2=int(mesh[i-1,j])
           n3=int(mesh[i,j-1])
           n4=int(mesh[i+1,j])
           l=int(mesh[i,j])
           if debug:
               print("I have neighbour %i,%i,%i,%i" % (n1,n2,n3,n4))
               print()
           #explict for clarity
           for m in [n1,n2,n3,n4]:
               if m !=0:
                 row[m-1]=1
           if debug:
               print(row)
           H[l-1,:]=row
           #reinitialize row
           row=np.zeros(counter)
   #complete the hamiltonian
   for k in range(counter):
       H[k,k]=Hdiag[k]
   np.savetxt("lowerH.txt",H)
   print("Hamiltonian is symm?")
   print(np.allclose(H,H.T))
   try:
       w,v=np.linalg.eigh(H)
   except np.linalg.LinAlgError:
       print("Error in numpy.linalg.eigh of inputted matrix")
   #reordering
   idw = w.argsort()[::-1]
   w = w[idw]
   v = v[:,idw]
   print("Eigenvalues e1, e2 ..")
   print(-w[:3]/(2*h*h))
   ########################################################################
   print("doing the upper half part...")
   uH=np.zeros((counter,counter)) # H mapped in the upper half mesh
   row=np.zeros(counter)           # re initialized
   ##reflect the mesh through the anti-diagonal: get the upper half -> umesh
   ## the diagonal of H is unchanged
   umesh=mesh[::-1,::-1].T
   #show the mesh
   if debug:
       print("showing the upper mesh")
       print(umesh)

   idrow=[]
   for k in range(1,num-2):
       idrow.append(k)
#
   for i in range(1,num-2):
       for j in idrow:
           if debug:
               print("I am node %i" % umesh[i,j])
           l=int(umesh[i,j])
           #four neighbour n1,n2,n3,n4
           n1=int(umesh[i,j+1])
           n2=int(umesh[i-1,j])
           n3=int(umesh[i,j-1])
           n4=int(umesh[i+1,j])
#
           if debug:
               print("I have neighbour %i,%i,%i,%i" % (n1,n2,n3,n4))
               print()
           for m in [n1,n2,n3,n4]:
               if m !=0:
                 row[m-1]=1
           if debug:
               print(row)
           uH[l-1,:]=row
           #reinitialize row
           row=np.zeros(counter)
           
       idrow.pop()     
   np.savetxt("upperH.txt",uH)
   ##complete the hamiltonian
   for k in range(counter):
       uH[k,k]=Hdiag[k]
   print("Hamiltonian is symm?")
   print(np.allclose(uH,uH.T))
   try:
       w_u,v_u=np.linalg.eigh(uH)
   except np.linalg.LinAlgError:
       print("Error in numpy.linalg.eigh of inputted matrix")
   #reordering
   idw_u = w_u.argsort()[::-1]
   w_u = w_u[idw_u]
   v_u = v_u[:,idw_u]    
   print("Eigenvalues e1, e2 ..")
   print(-w_u[:3]/(2*h*h))

   ######

   #np.savetxt("internals.txt",np.array(gridlist))

   #internal point grid
   #half lower part
   #check

   # generate the grid point for printing
   
   
   lgridlist=np.array(gridlist)
   print("n. points of lower half grid: %i\n" % lgridlist.shape[0])
   ugridlist=np.empty((counter,2))
   ugridlist[:,0]=np.array(gridlist)[:,1]
   ugridlist[:,1]=np.array(gridlist)[:,0]
   print("n. points of upper half grid: %i\n" % ugridlist.shape[0])
   internalgrid=np.append(lgridlist,ugridlist,axis=0)
   print("n. int. grid point: %i\n" % internalgrid.shape[0])
   #complete with boundary points
   tmp = []
   #diagonal points
   for i in range(num):
       tmp.append((i*h,i*h))
   # (x1,0) points \ (0,0)

   for i in range(1,num):
       tmp.append((i*h,0.0))
   # (a,x2) points \ (a,0) and (a,a)
   for i in range(1,num-1):
       tmp.append((a,i*h))
   # (x1,a) points \ (a,a)

   for i in range(num-1):
       tmp.append((i*h,a))
   # (0,x2) points \ (0,0) and (0,a)
   for i in range(1,num-1):
       tmp.append((0,i*h))
   totalgrid=np.append(internalgrid,np.array(tmp),axis=0)    
   #check
   print("total grid length: %i" % totalgrid.shape[0])


   #total number of grid points
   totpoints=num*num
   #number of bounday points 
   nzero = totpoints -2*counter
   pbound=np.zeros(nzero)
   #ground state wfn
   psi=np.append(v[:,eigvec],v_u[:,eigvec])
   totalpsi = np.append(psi,pbound)

   np.savetxt("mesh.txt",totalgrid)
   np.savetxt('wavefunction'+str(eigvec) +'.txt', np.c_[np.array(totalgrid),np.array(totalpsi)], fmt='%.12e')
   
   return np.array(totalgrid),np.array(totalpsi)
if __name__ == '__main__':
 parser = argparse.ArgumentParser()

 parser.add_argument("-d", "--debug", help="Debug on, prints some infos", required=False,
         default=False, action="store_true")
 parser.add_argument("--nsamples" , help="N. of sampling points n= L/dn", required=False,
         default=7,type = int)
 parser.add_argument("-a", "--margin" , help="Set the margin of the grid L=a", required=False,
         default=4.0, type = float)
 parser.add_argument("--state" , help="Eingenstate to dump", required=False,
         default=0,type = int)
 args = parser.parse_args()

 debug = args.debug 
 eigvec = args.state

 num=args.nsamples # test differnt values, e.g 20
 a=args.margin
 h=a/(num-1)
 print("h : %4.4f" %h)
 print("grid points %i x %i \n" % (num,num))
 if eigvec == 0:
     print("Dumping groud state\n")
 else:
     print("Dumping excited state %1\n" % eigvec)   

 main(num,a,h,eigvec,debug)
#############################################################################
##
##np.savetxt("internals.txt",np.array(gridlist))

##complete with boundary points
##diagonal points
#for i in range(num):
#    gridlist.append((i*h,i*h))
## (x1,0) points \ (0,0)

#for i in range(1,num):
#    gridlist.append((i*h,0.0))
## (a,x2) points \ (a,0) and (a,a)
#for i in range(1,num-1):
#    gridlist.append((a,i*h))
##check
#gridlen=len(gridlist)
#nzero=gridlen-counter
#print("Lower half grid length: %i" % gridlen)
#pbound=np.zeros(nzero)
##ground state wfn
#psi=np.append(v[:,-2],pbound)
#ugridlist=np.empty((gridlen,2))
#ugridlist[:,0]=np.array(gridlist)[:,1]
#ugridlist[:,1]=np.array(gridlist)[:,0]

##assemble the total mesh avoiding duplicated points
#totlist=np.concatenate((np.array(gridlist),ugridlist[:counter],ugridlist[counter+num:]),axis=0)
#print("total grid length: %i" % totlist.shape[0])
#totpsi=np.append(psi,psi[:gridlen-num])
#np.savetxt("mesh.txt",totlist)
#np.savetxt('res.txt', np.c_[np.array(totlist),np.array(totpsi)], fmt='%.12e')
