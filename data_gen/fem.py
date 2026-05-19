import sys         
from ngsolve import *
#from ngsolve.webgui import Draw
#import netgen.gui
from netgen.geom2d import SplineGeometry
import numpy as np
import ngsolve_special_functions as ngs

### Correct Born. 
# functions for near field data
def Phi(kappa,rrad):
    return CoefficientFunction((1j/4)*ngs.hankel1(v=0,z=kappa*rrad))

def GradPhi(kappa,rrad,xp,yp):
    return CoefficientFunction((-1j*kappa/4)*ngs.hankel1(v=1,z=kappa*rrad)*CoefficientFunction(((x-xp)/rrad,(y-yp)/rrad)))

def nearf2d(u,mesh,farp,kappa,porder):
    # Compute near field pattern at given points by integrating -u*dPhi/dnu + du/dnu*Phi over the scatterer boundary
    # Points must lie outside the scatterer
    ntheta=farp["n"]
    ctheta=farp["cent"]
    apptheta=farp["app"]
    RM=farp["RM"]
    points=np.zeros((ntheta,2))
    theta=np.zeros(ntheta)
    for jp in range(0,ntheta):
        theta[jp]=(ctheta-apptheta/2)+apptheta*jp/(ntheta-1)
    if RM==np.inf:
        points[:,0]=np.cos(theta)
        points[:,1]=np.sin(theta)
    else:
        points[:,0]=RM*np.cos(theta)
        points[:,1]=RM*np.sin(theta)
    nv=specialcf.normal(mesh.dim)
    unear=np.zeros(farp["n"],dtype=complex)
    fesa=H1(mesh, order=porder, complex=True, definedon=mesh.Materials("air"))
    for ipnt in range(0,farp["n"]):
        xp,yp=points[ipnt,:]
        rad=CoefficientFunction(sqrt((x-xp)**2+(y-yp)**2))
        # Compute u*dPhi/dnu on the scatterer boundary (nu outward)
        func1=CoefficientFunction((GradPhi(kappa,rad,xp,yp)*nv) * u)
        with TaskManager():
            unear1=Integrate(tuple(func1),mesh,order=porder+1,definedon=
                            mesh.Boundaries("scatterer"))
        # Integrate du/dnu*Phi over the scatterer boundary by lifting to the air region
        vv=GridFunction(fesa)
        vv.vec[:]=0
        vv.Set(Phi(kappa,rad),BND,definedon=mesh.Boundaries("scatterer"))
        func2=CoefficientFunction(grad(vv)*grad(u)-kappa*kappa*vv*u)
        with TaskManager():
            unear2=Integrate(tuple(func2),mesh,order=porder+1,definedon=mesh.Materials("air"))
        unear[ipnt]=(unear1+unear2)
    return(unear,theta) 

# FEM Functions to generate far field data
def farf2d(u,mesh,farp,kappa,porder):
    ntheta=farp["n"]
    ctheta=farp["cent"]
    apptheta=farp["app"]
    nv=specialcf.normal(mesh.dim)
    theta=np.zeros(ntheta)
    uinf=np.zeros(ntheta,dtype=complex)
    fesa=H1(mesh, order=porder, complex=True, definedon=mesh.Materials("air"))
    for jp in range(0,ntheta):
        theta[jp]=(ctheta-apptheta/2)+apptheta*jp/(ntheta-1)
        # phi[jp]=2*np.pi*jp/(nphi-1)  ## Special for testing
        xhat=CoefficientFunction((np.cos(theta[jp]),np.sin(theta[jp])))
        Eout = exp(-1J*kappa*(x*xhat[0]+y*xhat[1]))
        func1=CoefficientFunction(-1j*kappa*(xhat*nv)*Eout * u)
        uinf1=Integrate(tuple(func1),mesh,order=porder+1,definedon=
                            mesh.Boundaries("scatterer"))
        vv=GridFunction(fesa)
        vv.vec[:]=0
        vv.Set(Eout,BND,definedon=mesh.Boundaries("scatterer"))
        fvv=CoefficientFunction(grad(vv)*grad(u)-kappa*kappa*vv*u)
        uinf2=Integrate(tuple(fvv),mesh,order=porder+1,definedon=mesh.Materials("air"))
        uinf[jp]=exp(1J*np.pi/4)/np.sqrt(8*np.pi*kappa)*(uinf1+uinf2)
    return(uinf,theta)

def helmsol(mesh,porder,ncoef,kappa,incp,farp):
    fes = H1(mesh, order=porder, complex=True)
    u = fes.TrialFunction()
    v = fes.TestFunction()
    a = BilinearForm(fes)
    a += SymbolicBFI(grad(u)*grad(v) - kappa**2*ncoef*u*v)
    a += SymbolicBFI(-1j*kappa*u*v,definedon=mesh.Boundaries("outerbnd"))
    print('Number of DoFs: ',fes.ndof)
    gfu = GridFunction(fes)
    #Draw(gfu,mesh,'us')
    with TaskManager():
        a.Assemble()
        Ainv=a.mat.Inverse()
    umeas=np.zeros((farp["n"],incp["n"]),dtype=complex)
    theta=np.zeros(incp["n"]);
    center=incp["cent"]
    app=incp["app"]
    RT=incp["RT"]
    for ip in range(0,incp["n"]):
        if ip%10==0:
            print("Done ip = ", ip,' of ',incp["n"])
        if incp["n"]==1:
            theta[0]=0.0
        else:
            theta[ip]=(center-app/2)+app*ip/(incp["n"]-1)
        if RT==np.inf:
            d=np.array([np.cos(theta[ip]),np.sin(theta[ip])])
        else:
            zp=np.array([RT*np.cos(theta[ip]),RT*np.sin(theta[ip])])
        with TaskManager():
            b = LinearForm(fes)
            if RT==np.inf:
                ui=exp(1J*kappa*(d[0]*x+d[1]*y))
            else:
                rrad=CoefficientFunction(sqrt((x-zp[0])**2+(y-zp[1])**2))
                ui=(1j/4.)*ngs.hankel1(v=0,z=kappa*rrad)
            b += SymbolicLFI(kappa*kappa*(ncoef-1)*ui * v)
            b.Assemble()
            gfu.vec.data =  Ainv * b.vec
            #Redraw()
            if farp["RM"]==np.inf:
                umeas[:,ip],phi=farf2d(gfu,mesh,farp,kappa,porder)
            else:
                umeas[:,ip],phi=nearf2d(gfu,mesh,farp,kappa,porder)
    return(umeas,phi,theta)
    
# Discretize the born operator
def discretize_born(k, xlim, phi, Ngrid, theta, RT, RM):
    from scipy.special import hankel1
    vert_step = 2 * xlim / Ngrid  # Vertical discretization step size
    hor_step = 2 * xlim / Ngrid  # Horizontal discretization step size
    if RM==np.inf:
        Cfac = vert_step * hor_step * np.exp(1j * np.pi / 4) * np.sqrt(k**3 / (np.pi * 8))  # Compute constant factor
    else:
        Cfac=vert_step * hor_step
    y1 = np.linspace(-xlim, xlim, Ngrid)  # Discretize horizontal domain
    y2 = np.linspace(-xlim, xlim, Ngrid)  # Discretize vertical domain
    Y1, Y2 = np.meshgrid(y1, y2)  # Produce matrices of coordinates on square domain
    grid_points = np.column_stack((Y1.ravel(), Y2.ravel()))  # Combine matrices into N^2 x 2 (matrix of coordinate pairs)
    if RM==np.inf:
        xhat = np.array([np.cos(theta), np.sin(theta)]).T # Measurement angle values to evaluate far field at
    else:
        xM=RM*np.array([np.cos(theta), np.sin(theta)]).T # Measurement points
    if RT==np.inf:
        d = np.array([np.cos(phi), np.sin(phi)]) # Incident wave directions
    else:
        zT = RT*np.array([np.cos(phi), np.sin(phi)]) # Incident source points
    ntheta=len(theta)
    Exp=np.zeros((ntheta,Ngrid*Ngrid),dtype=complex)
    if RM==np.inf and RT==np.inf:
        diff = xhat - d
        dot_products = np.dot(diff, grid_points.T)
        Exp = np.exp(1j * k * dot_products)
    elif RM==np.inf and RT!=np.inf:# far field measurments, incident point source
        # normalization likely wrong for mixed case  (not used in paper)
        for iy in range(0,Ngrid*Ngrid):
            src=(1j/4)*hankel1(0,k*np.linalg.norm(grid_points[iy,:]-zT))
            for iz in range(0,ntheta):
                Exp[iz,iy]=np.exp(1j*k*np.dot(xhat[iz,:],grid_points[iy,:]))*src
    elif RM!=np.inf and RT==np.inf: # near field measurements and plane wave incidence   
        # normalization likely wrong for mixed case  (not used in paper
        for iy in range(0,Ngrid*Ngrid):
            src=np.exp(-1j*k*np.dot(d,grid_points[iy,:]))
            for iz in range(0,ntheta):
                Exp[iz,iy]=(1j/4)*hankel1(0,k*np.linalg.norm(grid_points[iy,:]-xM[iz,:]))*src
    elif RM!=np.inf and RT!=np.inf: # near field source and reciever
        for iy in range(0,Ngrid*Ngrid):
            src=(1j/4)*hankel1(0,k*np.linalg.norm(grid_points[iy,:]-zT))
            for iz in range(0,ntheta):
                Exp[iz,iy]=k*k*(1j/4)*hankel1(0,k*np.linalg.norm(grid_points[iy,:]-xM[iz,:]))*src 
    A = Cfac * Exp # Discretize born operator
    return A