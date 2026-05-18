import sys         
from ngsolve import *
#from ngsolve.webgui import Draw
#import netgen.gui
from netgen.geom2d import SplineGeometry
import numpy as np

# functions for near field data
def Phi(kappa,rrad):
    return CoefficientFunction((1j/4)*ngs.hankel1(v=0,z=kappa*rrad))

def GradPhi(kappa,rrad,xp,yp):
    return CoefficientFunction((-1j*kappa/4)*ngs.hankel1(v=1,z=kappa*rrad)*CoefficientFunction(((x-xp)/rrad,(y-yp)/rrad)))

def nearf2d(u,mesh,points,npoints,kappa,porder):
    # Compute near field pattern at given points by integrating -u*dPhi/dnu + du/dnu*Phi over the scatterer boundary
    # Points must lie outside the scatterer
    nv=specialcf.normal(mesh.dim)
    unear=np.zeros(npoints,dtype=complex)
    fesa=H1(mesh, order=porder, complex=True, definedon=mesh.Materials("air"))
    for ipnt in range(0,npoints):
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
    return(unear) 

# FEM Functions to generate far field data
def farf2d(u,mesh,farp,kappa,porder):
    ntheta=farp["n"]
    ctheta=farp["cent"]
    apptheta=farp["app"]
    nv=specialcf.normal(mesh.dim)
    theta=np.zeros(ntheta)
    uinf=np.zeros(ntheta,dtype=complex)
    fesa=H1(mesh, order=porder, complex=True, definedon=mesh.Materials("air"))
    #print(Integrate(CoefficientFunction(nv[0]*(x-x0[0])+nv[1]*(y-x0[1])),mesh,order=porder+1,definedon=mesh.Boundaries("scatterer")))
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
    uinf=np.zeros((farp["n"],incp["n"]),dtype=complex)
    phi=np.zeros(incp["n"]);
    center=incp["cent"]
    app=incp["app"]
    for ip in range(0,incp["n"]):
        if ip%10==0:
            print("Done ip = ", ip,' of ',incp["n"])
        if incp["n"]==1:
            phi[0]=0.0
        else:
            phi[ip]=(center-app/2)+app*ip/(incp["n"]-1)
        d=[np.cos(phi[ip]),np.sin(phi[ip])]
        with TaskManager():
            b = LinearForm(fes)
            ui=exp(1J*kappa*(d[0]*x+d[1]*y))
            b += SymbolicLFI(kappa*kappa*(ncoef-1)*ui * v)
            b.Assemble()
            gfu.vec.data =  Ainv * b.vec
            Redraw()
        uinf[:,ip],theta=farf2d(gfu,mesh,farp,kappa,porder)
    return(uinf,phi,theta)
    
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
    
    if RM==np.inf and RT==np.inf:
        diff = xhat - d
        dot_products = np.dot(diff, grid_points.T)
        Exp = np.exp(1j * k * dot_products)
    elif RM==np.inf and RT!=np.inf:# far field measurments, ins=cident point source
        src=np.array((Ngrid*Ngrid,len(phi)),dtype=complex)
        rcv=np.array((len(theta),Ngrid*Ngrid),dtype=complex)
        for iy in range(0,Ngrid*Ngrid):
            for ir in range(0,len(phi)):
                src=1j/4*hankel1(0,k*np.linalg.norm(grid_points[iy,:]-zT[ir,:]))
            for iz in range(0,len(theta)):
                rcv=np.exp(1j*k*np.dot(xhat,grid_points[iy,:]))
        Exp=rcv*src
    elif RM!=np.inf and RT==np.inf: # near field measurements and plane wave incidence
        src=np.array((Ngrid*Ngrid,len(phi)),dtype=complex)
        rcv=np.array((len(theta),Ngrid*Ngrid),dtype=complex)
        for iy in range(0,Ngrid*Ngrid):
            for ir in range(0,len(phi)):
                src=np.exp(1j*k*np.dot(-d,grid_points[iy,:]))
            for iz in range(0,len(theta)):
                rcv=1j/4*hankel1(0,k*np.linalg.norm(grid_points[iy,:]-xM[iz,:]))
        Exp=rcv*src
    elif RM!=np.inf and RT!=np.inf: # near field source and reciever
        src=np.array((Ngrid*Ngrid,len(phi)),dtype=complex)
        rcv=np.array((len(theta),Ngrid*Ngrid),dtype=complex)
        for iy in range(0,Ngrid*Ngrid):
            for ir in range(0,len(phi)):
                src=1j/4*hankel1(0,k*np.linalg.norm(grid_points[iy,:]-zT[ir,:]))
            for iz in range(0,len(theta)):
                rcv=1j/4*hankel1(0,k*np.linalg.norm(grid_points[iy,:]-xM[iz,:]))
        Exp=rcv*src
    A = Cfac * Exp # Discretize born operator
    print("A shape:", A.shape)
    return A
