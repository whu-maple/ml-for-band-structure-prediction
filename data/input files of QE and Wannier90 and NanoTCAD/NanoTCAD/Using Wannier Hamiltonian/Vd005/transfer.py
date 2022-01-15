from NanoTCAD_ViDES import *
import numpy as np

a=loadtxt("wannier.H")
h2=array(a,dtype="complex")

h2[:,2]=h2[:,2]+h2[:,3]*1j
h=h2[:,0:3]

a=loadtxt('n_Nc.dat');
n=int(a[0]);
Nc=int(a[1]);
nnn=zeros(Nc);
for i in range(0,Nc):
    nnn[i]=n;
del a

MOSFET=Hamiltonian(n,Nc,nnn);
MOSFET.H=h;

MOSFET.dE =0.001;
MOSFET.Eupper = 2;
MOSFET.Elower =-2;
NE=(int)((MOSFET.Eupper-MOSFET.Elower)/MOSFET.dE+1);
MOSFET.Phi=zeros(n*Nc);
MOSFET.Ei=zeros(n*Nc);
MOSFET.LDOS_source=zeros(NE*Nc);
MOSFET.LDOS_drain=zeros(NE*Nc);


[x,y,z]=get_xyz_from_file("DEVICE.xyz");

MOSFET.x=around(x/10.0,2)
MOSFET.y=around(y/10.0,2)
MOSFET.z=around(z/10.0,2)

X=nonuniformgrid(array([-0.5,0.4,0.5,0.1,1.5,0.4]));
y1=nonuniformgrid(array([min(MOSFET.y)-1.0,0.4,min(MOSFET.y),0.1]));
y2=nonuniformgrid(array([max(MOSFET.y),0.1,max(MOSFET.y)+1.0,0.4]));
Y=concatenate((y1,y2));
grid=grid3D(X,Y,MOSFET.z,MOSFET.x,MOSFET.y,MOSFET.z);

Vdrain =0.05;
Workfunction=0;

MOSFET.mu1=0;
MOSFET.mu2=-Vdrain;

Vgsmin= 0.0;
Vgsmax= 0.7;
Vgstep= 0.1;
num=int(round((Vgsmax-Vgsmin)/Vgstep)+1);

SiO2=region("hex",-0.5,1.5,grid.ymin,grid.ymax,grid.zmin,grid.zmax);
SiO2.eps=3.9;
z3=grid.zmin + 5.112 - 0.07;
z4=grid.zmax - 5.112 + 0.07;
gate_up=gate("hex",grid.xmax,grid.xmax,grid.ymin,grid.ymax,z3,z4);
gate_down=gate("hex",grid.xmin,grid.xmin,grid.ymin,grid.ymax,z3,z4);

gate_up.Ef=-Workfunction;
gate_down.Ef=-Workfunction;

p=interface3D(grid,SiO2,gate_up,gate_down,NE,Nc);

source_dope=region("hex",grid.xmin,grid.xmax,grid.ymin,grid.ymax,grid.zmin,z3);
drain_dope=region("hex",grid.xmin,grid.xmax,grid.ymin,grid.ymax,z4,grid.zmax);
dope_reservoir(grid,p,MOSFET,5e-3,source_dope);
dope_reservoir(grid,p,MOSFET,5e-3,drain_dope);


v=zeros(num);
current=zeros(num);

counter=0;
Vgs = 0.0;
Vgsmax=Vgsmax+0.0001;
string="./IdVg_Vd=%s" %Vdrain;

if (not os.path.exists(string)):
    os.system("mkdir %s" %string);


p.normpoisson = 0.1;   
p.normd = 0.04;

string="Phi0.00.out";
a=loadtxt(string);
b=array(a);
p.Phi=b[:,3];
del a;
del b;
flag = 0;

while (Vgs<=Vgsmax+0.001):

    # I set the Fermi level of the gate
    gate_up.Ef=-Vgs-Workfunction;
    gate_down.Ef=-Vgs-Workfunction;
    set_gate(p,gate_up);
    set_gate(p,gate_down);

    print "calculating Id at Vg=%.2f" %Vgs;
    # print "Normd=%s" %p.normd;
    solve_self_consistent_TB(grid,p,MOSFET,y,9000);
    v[counter]=Vgs;
    #computing the current
    vt=kboltz*300.0/q;
    E=array(MOSFET.E);
#    savetxt("E",E);
    T=array(MOSFET.T);
#    savetxt("T",T);
    arg=2*q*q/(2*pi*hbar)*T*(Fermi((E-MOSFET.mu1)/vt)-Fermi((E-MOSFET.mu2)/vt))*MOSFET.dE
    id=sum(arg);
    current[counter]=id;     
    if ((Vgs-0)**2<1e-9):
	    Vgs=0;
    a=[grid.x3D,grid.y3D,grid.z3D,p.Phi];
    string="./IdVg_Vd=%s/Phi%.2f.out" % (Vdrain,Vgs);
    savetxt(string,transpose(a));
    del a;
    a=[MOSFET.x,MOSFET.y,MOSFET.z,MOSFET.phi_atoms];
    string="./IdVg_Vd=%s/Phi_atoms%.2f.out" % (Vdrain,Vgs);
    savetxt(string,transpose(a));
    del a;
    a=[grid.x3D,grid.y3D,grid.z3D,p.free_charge];
    string="./IdVg_Vd=%s/ncar%.2f.out" % (Vdrain,Vgs);
    savetxt(string,transpose(a));
    del a;
    a=[MOSFET.x,MOSFET.y,MOSFET.z,MOSFET.free_charge_atoms];
    string="./IdVg_Vd=%s/ncar_atoms%.2f.out" % (Vdrain,Vgs);
    savetxt(string,transpose(a));
    del a;
    a=[MOSFET.E,MOSFET.T];
    string="./IdVg_Vd=%s/T%.2f.out" % (Vdrain,Vgs);
    savetxt(string,transpose(a));
    del a;
    a=[MOSFET.E,arg];
    string="./IdVg_Vd=%s/I%.2f.out" % (Vdrain,Vgs);
    savetxt(string,transpose(a));
    del a;
    del arg;
    E_final=zeros(NE*Nc);
    for i in range(0,NE):
        for j in range(0,Nc):
            E_final[j+i*Nc]=E[i];
                
    Z_final_tmp=zeros(Nc);
    for j in range(0,Nc):
        Z_final_tmp[j]=(sum(z[sum(nnn[0:j]):sum(nnn[0:j+1])])/nnn[j])/10
        # Z_final_tmp=unique(z)/10;
        #savetxt("Z_final_tmp",Z_final_tmp);
    Z_final=zeros(NE*Nc);
    for i in range(0,NE):
	    for j in range(0,Nc):
	        Z_final[j+i*Nc]=Z_final_tmp[j];
    a=[E_final,Z_final,p.LDOS_source];
    string="./IdVg_Vd=%s/LDOS_source%.2f.out" % (Vdrain,Vgs);
    savetxt(string,transpose(a));
    del a;
    a=[E_final,Z_final,p.LDOS_drain];
    string="./IdVg_Vd=%s/LDOS_drain%.2f.out" % (Vdrain,Vgs);
    savetxt(string,transpose(a));
    del a;
    del E_final;
    del Z_final;

    counter=counter+1;
    Vgs=Vgs+Vgstep;


tempo=[v,current]
string="./IdVg_Vd=%s/idvds_Vd=%.2f.out" % (Vdrain,Vdrain);
savetxt(string,transpose(tempo));






