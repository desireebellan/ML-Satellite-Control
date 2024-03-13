from kinematics import Accelerations, DiffKinematics, Kinematics
from attitude_transformations import quat_DCM
from utils import SkewSym, transpose, AttributeDict
from numpy import zeros, eye
import numpy as np

def I_I(R0,RL,robot):
    # Projects the link inertias in the inertial CCS.
    #
    # [I0,Im]=I_I(R0,RL,robot)
    #
    # :parameters: 
    #   * R0 -- Rotation matrix from the base-link CCS to the inertial CCS -- [3x3].
    #   * RL -- Links CCS 3x3 rotation matrices with respect to the inertial CCS -- as a [3x3xn] matrix.
    #   * robot -- Robot model (see :doc:`/Tutorial_Robot`).
    #
    # :return: 
    #   * I0 -- Base-link inertia matrix, projected in the inertial CCS -- as a [3x3] matrix.
    #   * Im -- Links inertia matrices, projected in the inertial CCS -- as a [3x3xn] matrix.
    #
    # See also: :func:`src.kinematics_dynamics.MCB`. 

    '''{  
        LICENSE

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY  without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Lesser General Public License for more details.

        You should have received a copy of the GNU Lesser General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
    }'''

    #=== CODE ===#

    #Base-link inertia
    I0 = R0 @ robot.base_link.inertia @ transpose(R0)

    #Pre-allocate inertias
    Im=zeros((3,3,robot.n_links_joints),dtype=R0.dtype)

    #Inertias of the links
    for i in range(robot.n_links_joints):
        Im[:,:,i]=RL[:,:,i] @ robot.links[i].inertia @ transpose(RL[:,:,i])
    
    return I0, Im


def ID(wF0,wFm,t0,tL,t0dot,tLdot,P0,pm,I0,Im,Bij,Bi0,robot):
    # This function solves the inverse dynamics (ID) problem (it obtains the
    # generalized forces from the accelerations) for a manipulator.
    #
    # [tau0,taum] = ID(wF0,wFm,t0,tL,t0dot,tLdot,P0,pm,I0,Im,Bij,Bi0,robot)
    # 
    # :parameters: 
    #   * wF0 -- Wrench acting on the base-link center-of-mass [n,f], projected in the inertial CCS -- as a [6x1] matrix.
    #   * wFm -- Wrench acting on the links center-of-mass  [n,f], projected in the inertial CCS -- as a [6xn] matrix.
    #   * t0 -- Base-link twist [\omega,rdot], projected in the inertial CCS -- as a [6x1] matrix.
    #   * tL -- Manipulator twist [\omega,rdot], projected in the inertial CCS -- as a [6xn] matrix.
    #   * t0dot -- Base-link twist-rate vector \omegadot,rddot], projected in inertial frame -- as a [6x1] matrix.
    #   * tLdot -- Manipulator twist-rate vector \omegadot,rddot], projected in inertial frame -- as a [6xn] matrix.
    #   * P0 -- Base-link twist-propagation "vector" -- as a [6x6] matrix.
    #   * pm -- Manipulator twist-propagation "vector" -- as a [6xn] matrix.
    #   * I0 -- Base-link inertia matrix, projected in the inertial CCS -- as a [3x3] matrix.
    #   * Im -- Links inertia matrices, projected in the inertial CCS -- as a [3x3xn] matrix.
    #   * Bij -- Twist-propagation matrix (for manipulator i>0 and j>0) -- as a [6x6xn] matrix.
    #   * Bi0 -- Twist-propagation matrix (for i>0 and j=0) -- as a [6x6xn] matrix.
    #   * robot -- Robot model (see :doc:`/Tutorial_Robot`).
    #
    # :return: 
    #   * tau0 -- Base-link forces [n,f]. The torque n is projected in the body-fixed CCS, while the force f is projected in the inertial CCS -- [6x1].
    #   * taum -- Joint forces/torques -- as a [n_qx1] matrix.
    #
    # See also: :func:`src.kinematics_dynamics.Floating_ID` and :func:`src.kinematics_dynamics.FD`. 


    '''{  
        LICENSE

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY  without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Lesser General Public License for more details.

        You should have received a copy of the GNU Lesser General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
    }'''

    #=== CODE ===#

    #--- Number of links and Joints ---#
    n_q = robot.n_q
    n=robot.n_links_joints

    #--- Mdot ---#
    #base-link Mdot
    Mdot0 = np.block([[SkewSym(t0[:3]) @ I0, zeros((3,3))], [zeros([3,3]), zeros([3,3])]])

    #Pre-allocate
    Mdot=zeros((6,6,n),dtype=wF0.dtype)

    #Manipulator Mdot
    for i in range(n):
        Mdot[:,:,i]=np.block([[SkewSym(tL[:3,i]) @ Im[:,:,i], zeros((3,3))], [zeros((3,3)), zeros((3,3))]])

    #--- Forces ---#

    #Base-link
    wq0=np.block([[I0,zeros((3,3))],[zeros((3,3)),robot.base_link.mass*eye(3)]])@t0dot + Mdot0@t0 - wF0

    #Pre-allocate
    wq=zeros((6,n),dtype=wF0.dtype)

    #Manipulator
    for i in range(n):
        wq[:,i]=np.block([[Im[:,:,i],zeros((3,3))],[zeros((3,3)),robot.links[i].mass*eye(3)]])@tLdot[:,i]\
            + Mdot[:,:,i]@tL[:,i] - wFm[:,i]

    #Pre-allocate
    wq_tilde=zeros((6,n),dtype=wF0.dtype)

    #Backwards recursion
    for i in range(n-1, -1, -1):
        #Initialize wq_tilde
        wq_tilde[:,i]=wq[:,i]
        #Add children contributions
        child = np.nonzero(robot.con.child[:,i])[0].tolist()
        for j in child:
            wq_tilde[:, i]=wq_tilde[:, i] + transpose(Bij[:,:,j,i])@wq_tilde[:,j]

    #Base-link
    wq_tilde0=wq0
    #Add children contributions
    child = np.nonzero(robot.con.child_base)[0].tolist()
    for j in child:
        wq_tilde0 += transpose(Bi0[:,:,j])@wq_tilde[:,j:j+1]

    #---- Joint forces ---#
    #Base-link
    tau0=transpose(P0)@wq_tilde0

    #Pre-allocate
    taum=zeros((n_q,1),dtype=wF0.dtype)

    #Manipulator joint forces.
    for i in range(n):
        if robot.joints[i].type!=0:
            taum[robot.joints[i].q_id,0]=transpose(pm[:,i:i+1])@wq_tilde[:,i]
            
    return tau0, taum

def FD(tau0,taum,wF0,wFm,t0,tm,P0,pm,I0,Im,Bij,Bi0,u0,um,robot):
    # This function solves the forward dynamics (FD) problem (it obtains the
    # acceleration from  forces).
    #
    # [u0dot,umdot] = FD(tau0,taum,wF0,wFm,t0,tm,P0,pm,I0,Im,Bij,Bi0,u0,um,robot)
    #
    # :parameters: 
    #   * tau0 -- Base-link forces [n,f]. The torque n is projected in the body-fixed CCS, while the force f is projected in the inertial CCS -- [6x1].
    #   * taum -- Joint forces/torques -- as a [n_qx1] matrix.
    #   * wF0 -- Wrench acting on the base-link center-of-mass [n,f], projected in the inertial CCS -- as a [6x1] matrix.
    #   * wFm -- Wrench acting on the links center-of-mass  [n,f], projected in the inertial CCS -- as a [6xn] matrix.
    #   * t0 -- Base-link twist [\omega,rdot], projected in the inertial CCS -- as a [6x1] matrix.
    #   * tL -- Manipulator twist [\omega,rdot], projected in the inertial CCS -- as a [6xn] matrix.
    #   * P0 -- Base-link twist-propagation "vector" -- as a [6x6] matrix.
    #   * pm -- Manipulator twist-propagation "vector" -- as a [6xn] matrix.
    #   * I0 -- Base-link inertia matrix, projected in the inertial CCS -- as a [3x3] matrix.
    #   * Im -- Links inertia matrices, projected in the inertial CCS -- as a [3x3xn] matrix.
    #   * Bij -- Twist-propagation matrix (for manipulator i>0 and j>0) -- as a [6x6xn] matrix.
    #   * Bi0 -- Twist-propagation matrix (for i>0 and j=0) -- as a [6x6xn] matrix.
    #   * u0 -- Base-link velocities [\omega,rdot]. The angular velocity is projected in the body-fixed CCS, while the linear velocity is projected in the inertial CCS -- [6x1].
    #   * um -- Joint velocities -- [n_qx1].
    #   * robot -- Robot model (see :doc:`/Tutorial_Robot`).
    #
    # :return: 
    #   * u0dot -- Base-link accelerations [\omegadot,rddot]. The angular acceleration is projected in a body-fixed CCS, while the linear acceleration is projected in the inertial CCS -- [6x1].
    #   * umdot -- Manipulator joint accelerations -- [n_qx1].
    #
    # See also: :func:`src.kinematics_dynamics.ID` and :func:`src.kinematics_dynamics.I_I`. 

    '''{  
        LICENSE

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY  without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Lesser General Public License for more details.

        You should have received a copy of the GNU Lesser General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
    }'''
    #=== CODE ===#

    #--- Number of links and Joints ---#
    n=robot.n_links_joints 
    n_q=robot.n_q 

    #---- Inverse Dynamics with 0 accelerations ---#
    #Recompute Accelerations with u0dot=umdot=0
    t0dot,tmdot=Accelerations(t0,tm,P0,pm,Bi0,Bij,u0,um,zeros((6,1)),zeros((n_q,1)),robot)
    
    
    #Use the inverse dynamics
    tau0_0ddot,taum_0ddot = ID(wF0,wFm,t0,tm,t0dot,tmdot,P0,pm,I0,Im,Bij,Bi0,robot)

    #--- Forward Dynamics ---#

    #Initialize solution
    phi0=tau0-tau0_0ddot
    phi=taum-taum_0ddot

    #--- M hat, psi hat and psi  ---#
    #Pre-allocate
    M_hat=zeros((6,6,n),dtype = tau0.dtype)
    psi_hat=zeros((6,n),dtype = tau0.dtype)
    psi=zeros((6,n),dtype = tau0.dtype)

    #Backwards recursion
    for i in range(n-1, -1, -1):
        #Initialize
        M_hat[:,:,i]=np.block([[Im[:,:,i],zeros((3,3))],[zeros((3,3)),robot.links[i].mass*eye(3)]])
        #Add children contributions
        child = np.nonzero(robot.con.child[:,i])[0].tolist()
        for j in child:
            M_hatii=M_hat[:,:,j]-psi_hat[:,j:j+1]@transpose(psi[:,j:j+1])
            M_hat[:,:,i] += transpose(Bij[:,:,j,i])@M_hatii@Bij[:,:,j,i]
            
        if robot.joints[i].type==0:
            psi_hat[:,i:i+1]=zeros((6,1)) 
            psi[:,i:i+1]=zeros((6,1)) 
        else:
            psi_hat[:,i]=M_hat[:,:,i]@pm[:,i] 
            psi[:,i]=psi_hat[:,i]/(transpose(pm[:,i:i+1])@psi_hat[:,i:i+1]) 
            

    #Base-link
    M_hat0=np.block([[I0,zeros((3,3))], [zeros((3,3)),robot.base_link.mass*eye(3)]])
    #Add children contributions
    child = np.nonzero(robot.con.child_base)[0].tolist()
    for j in child:
        M_hat0ii=M_hat[:,:,j] - psi_hat[:,j]@transpose(psi[:,j])
        M_hat0 += transpose(Bi0[:,:,j])@M_hat0ii@Bi0[:,:,j] 

    psi_hat0=M_hat0@P0 

    #--- eta ---#
    #Pre-allocate and initialize
    eta=zeros((6,n),dtype=P0.dtype) 
    phi_hat=zeros((6,n),dtype=P0.dtype) 
    phi_tilde=zeros((n_q),dtype=P0.dtype) 

    #Backwards recursion
    for i in range(n-1, -1, -1):
        #Initialize
        eta[:,i:i+1]=zeros((6,1)) 
        #Add children contributions
        child = np.nonzero(robot.con.child[:,i])[0].tolist()
        for j in child:
            eta[:,i:i+1]+=transpose(Bij[:,:,j,i])@(psi[:,j:j+1]*phi_hat[j,0]+eta[:,j:j+1]) 

        phi_hat[i]=-transpose(pm[:,i:i+1])@eta[:,i:i+1] 
        if robot.joints[i].type!=0:
            phi_hat[i] += phi[robot.joints[i].q_id]         
            phi_tilde[robot.joints[i].q_id] = phi_hat[i, 0]/(transpose(pm[:,i:i+1])@psi_hat[:,i:i+1]) 

    #Base-link
    eta0=zeros((6,1)) 
    #Add children contributions
    child = np.nonzero(robot.con.child_base)[0].tolist()
    for k in child:
        eta0+=transpose(Bi0[:,:,j])@(psi[:,j:j+1]*phi_hat[j, 0]+eta[:,j:j+1]) 

    phi_hat0=phi0-transpose(P0)@eta0 
    phi_tilde0=np.linalg.solve((transpose(P0)@psi_hat0),phi_hat0)


    #--- Base-link acceleration ---#
    u0dot=phi_tilde0 

    #--- Manipulator acceleration (and mu) ---#

    #Pre-allocate
    mu=zeros((6,n),dtype = P0.dtype) 
    umdot=zeros((n_q,1), dtype = P0.dtype) 


    #Forward recursion
    for i in range(n):
        
        if robot.joints[i].parent_link==0:
            #First joint
            mu[:,i:i+1]=Bi0[:,:,i]@(P0@u0dot) 
        else:
            #Rest of the links
            if robot.joints[robot.joints[i].parent_link].type!=0:
                mu_aux=(pm[:,robot.joints[robot.joints[i].parent_link].id]*umdot[robot.joints[i-1].q_id] \
                    + mu[:,robot.joints[robot.joints[i].parent_link].id]) 
            else:
                mu_aux=mu[:,robot.joints[robot.joints[i].parent_link].id]
            mu[:,i]=Bij[:,:,i,i-1]@mu_aux 
        
        #Initialize
        if robot.joints[i].type!=0:
            umdot[robot.joints[i].q_id,0]=phi_tilde[robot.joints[i].q_id]-transpose(psi[:,i:i+1])@mu[:,i:i+1] 

    return u0dot,umdot

def MCB(I0,Im,Bij,Bi0,robot):
    # Computes the Mass Composite Body Matrix (MCB) of the multibody system.
    #
    # [M0_tilde,Mm_tilde]=MCB(I0,Im,Bij,Bi0,robot)
    #
    # :parameters: 
    #   * I0 -- Base-link inertia matrix, projected in the inertial CCS -- as a [3x3] matrix.
    #   * Im -- Links inertia matrices, projected in the inertial CCS -- as a [3x3xn] matrix.
    #   * Bij -- Twist-propagation matrix (for manipulator i>0 and j>0) -- as a [6x6xn] matrix.
    #   * Bi0 -- Twist-propagation matrix (for i>0 and j=0) -- as a [6x6xn] matrix.
    #   * robot -- Robot model (see :doc:`/Tutorial_Robot`).
    #
    # :return: 
    #   * M0_tilde -- Base-link mass composite body matrix -- as a [6x6] matrix .
    #   * Mm_tilde -- Manipulator mass composite body matrix -- as a [6x6xn] matrix.
    #
    # See also: :func:`src.kinematics_dynamics.I_I`. 

    '''{  
        LICENSE

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY  without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Lesser General Public License for more details.

        You should have received a copy of the GNU Lesser General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
    }'''

    #=== CODE ===#

    #Number of links and Joints
    n=robot.n_links_joints

    #Pre-allocate
    Mm_tilde=zeros((6,6,n),dtype=I0.dtype)

    #Backwards recursion
    for i in range(n-1, -1, -1):
        Mm_tilde[:,:,i]=np.block([[Im[:,:,i],zeros((3,3))],[zeros((3,3)),robot.links[i].mass*eye(3)]])
        #Add children contributions
        child = np.nonzero(robot.con.child[:,i])[0].tolist()
        for j in child:
            Mm_tilde[:,:,i]+=transpose(Bij[:,:,j,i])@Mm_tilde[:,:,j]@Bij[:,:,j,i]

    #Base-link M tilde
    M0_tilde=np.block([[I0,zeros((3,3))],[zeros((3,3)),robot.base_link.mass*eye(3)]])
    #Add children contributions
    child = np.nonzero(robot.con.child_base)[0].tolist()
    for j in child:
        M0_tilde+=transpose(Bi0[:,:,j])@Mm_tilde[:,:,j]@Bi0[:,:,j]
        
    return M0_tilde,Mm_tilde


def GIM(q:np.ndarray, robot:AttributeDict):
    # Computes the Generalized Inertia Matrix (GIM) H of the multibody vehicle.
    #
    # This function uses a recursive algorithm.
    #
    # [H0, H0m, Hm] = GIM(M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot)
    #
    # :parameters: 
    #   * M0_tilde -- Base-link mass composite body matrix -- as a [6x6] matrix .
    #   * Mm_tilde -- Manipulator mass composite body matrix -- as a [6x6xn] matrix.
    #   * Bij -- Twist-propagation matrix (for manipulator i>0 and j>0) -- as a [6x6xn] matrix.
    #   * Bi0 -- Twist-propagation matrix (for i>0 and j=0) -- as a [6x6xn] matrix.
    #   * P0 -- Base-link twist-propagation "vector" -- as a [6x6] matrix.
    #   * pm -- Manipulator twist-propagation "vector" -- as a [6xn] matrix.
    #   * robot -- Robot model (see :doc:`/Tutorial_Robot`).
    #
    # :return: 
    #   * H0 -- Base-link inertia matrix -- as a [6x6] matrix.
    #   * H0m -- Base-link -- manipulator coupling inertia matrix -- as a [6xn_q] matrix.
    #   * Hm -- Manipulator inertia matrix -- as a [n_qxn_q] matrix.
    #   
    # To obtain the full generalized inertia matrix H:
    #
    # .. code-block:: matlab
    #   
    #   #Compute H
    #   [H0, H0m, Hm] = GIM(M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot) 
    #   H=[H0,H0m H0m' Hm] 
    #
    # See also: :func:`src.kinematics_dynamics.CIM`.


    '''{  
        LICENSE

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY  without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Lesser General Public License for more details.

        You should have received a copy of the GNU Lesser General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
    }'''
    # Get Needed parameters
    q0, qm = q[:7], q[7:]
    R0 = transpose(quat_DCM(transpose(q0[:4]).squeeze()))
    r0 = q0[4:]
    
    _, RL, _, rL, e, g = Kinematics(R0=R0, r0=r0, qm=qm, robot = robot)
    I0, Im = I_I(R0, RL, robot)
    Bij, Bi0, P0, pm = DiffKinematics(R0=R0, r0=r0, rL=rL, e=e, g=g)
    M0_tilde, Mm_tilde = MCB(I0,Im,Bij,Bi0,robot)
    

    #=== CODE ===#

    #--- Number of links and Joints ---#
    n_q=robot.n_q
    n=robot.n_links_joints

    #--- H matrix ---#

    #Base-link inertia matrix
    H0 = transpose(P0)@M0_tilde@P0

    #Pre-allocate Hm
    Hm=zeros((n_q,n_q),dtype=M0_tilde.dtype)

    #Manipulator inertia matrix Hm
    for j in range(n):
        for i in range(n):
            if robot.joints[i].type!=0 and robot.joints[j].type!=0:
                Hm[robot.joints[i].q_id,robot.joints[j].q_id] = transpose(pm[:,i:i+1])@Mm_tilde[:,:,i]@Bij[:,:,i,j]@pm[:,j:j+1]
                Hm[robot.joints[j].q_id,robot.joints[i].q_id] = Hm[robot.joints[i].q_id,robot.joints[j].q_id]

    #Pre-allocate H0m
    H0m=zeros((6,n_q),dtype=M0_tilde.dtype)

    #Coupling inertia matrix
    for i in range(n):
        if robot.joints[i].type!=0:
            H0m[:,robot.joints[i].q_id]=transpose(transpose(pm[:,i:i+1])@Mm_tilde[:,:,i]@Bi0[:,:,i]@P0)

    return H0, H0m, Hm

def CIM(q:np.ndarray, qdot:np.ndarray, robot:AttributeDict):
    # Computes the Generalized Convective Inertia Matrix C of the multibody system.
    #
    # :parameters: 
    #   * t0 -- Base-link twist [\omega,rdot], projected in the inertial CCS -- as a [6x1] matrix.
    #   * tL -- Manipulator twist [\omega,rdot], projected in the inertial CCS -- as a [6xn] matrix.
    #   * I0 -- Base-link inertia matrix, projected in the inertial CCS -- as a [3x3] matrix.
    #   * Im -- Links inertia matrices, projected in the inertial CCS -- as a [3x3xn] matrix.
    #   * M0_tilde -- Base-link mass composite body matrix -- as a [6x6] matrix .
    #   * Mm_tilde -- Manipulator mass composite body matrix -- as a [6x6xn] matrix.
    #   * Bij -- Twist-propagation matrix (for manipulator i>0 and j>0) -- as a [6x6xn] matrix.
    #   * Bi0 -- Twist-propagation matrix (for i>0 and j=0) -- as a [6x6xn] matrix.
    #   * P0 -- Base-link twist-propagation "vector" -- as a [6x6] matrix.
    #   * pm -- Manipulator twist-propagation "vector" -- as a [6xn] matrix.
    #   * robot -- Robot model (see :doc:`/Tutorial_Robot`).
    #
    # :return: 
    #   * C0 -> Base-link convective inertia matrix -- as a [6x6] matrix.
    #   * C0m -> Base-link - manipulator coupling convective inertia matrix -- as a [6xn_q] matrix.
    #   * Cm0 -> Manipulator - base-link coupling convective inertia matrix -- as a [n_qx6] matrix.
    #   * Cm -> Manipulator convective inertia matrix -- as a [n_qxn_q] matrix.
    #
    # To obtain the full convective inertia matrix C:
    #
    # .. code-block:: matlab
    #   
    #   #Compute the Convective Inertia Matrix C
    #   [C0, C0m, Cm0, Cm] = CIM(t0,tL,I0,Im,M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot)
    #   C=[C0,C0m Cm0,Cm] 
    #
    # See also: :func:`src.kinematics_dynamics.GIM`.

    '''{  
        LICENSE

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY  without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Lesser General Public License for more details.

        You should have received a copy of the GNU Lesser General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
    }'''

    #=== CODE ===#
    
    q0, qm = q[:7], q[7:]
    R0 = transpose(quat_DCM(transpose(q0[:4]).squeeze()))
    r0 = q0[4:]
    u0, um = qdot[:6], qdot[6:]
    
    RJ, RL, rJ, rL, e, g = Kinematics(R0, r0, qm, robot)
    Bij, Bi0, P0, pm = DiffKinematics(R0, r0, rL, e, g, robot)
    t0, tL = Velocities(Bij, Bi0, P0, pm, u0, um, robot)
    I0, Im = I_I(R0, RL, robot)
    M0_tilde, Mm_tilde = MCB(I0, Im, Bij, Bi0, robot)

    #--- Number of links and Joints ---#
    n_q=robot.n_q
    n=robot.n_links_joints

    #--- Omega ---#
    #Base-link Omega
    Omega0=np.block([[SkewSym(t0[:3]), zeros((3,3)) ],
        [zeros((3,3)), zeros((3,3))]] )

    #Pre-allocate Omega
    Omega=zeros((6,6,n),dtype = t0.dtype) 

    #Compute Omega
    for i in range(n):
        Omega[:,:,i]=np.block([[SkewSym(tL[:3,i]), zeros((3,3)) ]
            [zeros((3,3)), SkewSym(tL[:3,i])]] )

    #--- Mdot ---#
    #Base-link Mdot
    Mdot0=np.block([[Omega0[:3,:3]@I0, zeros((3,3))] , zeros((3,6))])

    #Pre-allocate
    Mdot=zeros((6,6,n),dtype=t0.dtype) 

    #Compute Mdot
    for i in range(n):
        Mdot[:,:,i]=np.block([[Omega[:3,:3,i]@Im[:,:,i], zeros((3,3)) ], zeros((3,6))])

    #--- Mdot tilde ---#
    #Pre-Allocate
    Mdot_tilde=zeros((6,6,n),dtype=t0.dtype) 

    #Backwards recursion
    for i in range(n-1, -1, -1):
        #Initialize
        Mdot_tilde[:,:,i]=Mdot[:,:,i]
        #Add children contributions
        child = np.nonzero(robot.con.child[:,i])[0].tolist() 
        for j in child:
            Mdot_tilde[:,:,i]+=Mdot_tilde[:,:,j]

    #Base-link
    Mdot0_tilde = Mdot0 
    #Add children contributions
    child = np.nonzero(robot.con.child_base)[0].tolist() 
    for j in child:
        Mdot0_tilde += Mdot_tilde[:,:,j]

    #--- Bdot ---#

    #Pre-allocate Bdotij
    Bdotij=zeros((6,6,n,n),dtype=t0.dtype) 

    #Compute Bdotij
    for j in range(n):
        for i in range(n):
            if robot.con.branch[i,j]==1:
                #Links are in the same branch
                Bdotij[:,:,i,j]=np.block([[zeros((3,3)), zeros((3,3))] , 
                                          [SkewSym(tL[3:,j]-tL[3:,i]), zeros((3,3))]] )
            else:
                #Links are not in the same branch
                Bdotij[:,:,i,j]=zeros((6,6)) 


    #--- Hij tilde ---#
    #Pre-allocate Hij_tilde
    Hij_tilde=zeros((6,6,n,n),dtype=t0.dtype) 

    #Hij_tilde
    for i in range(n-1, -1, -1):
        for j in range(n-1, -1, -1):
            Hij_tilde[:,:,i,j]=Mm_tilde[:,:,i,j]@Bdotij[:,:,i,j]
            #Add children contributions
            child = np.nonzero(robot.con.child[:,i])[0].tolist() 
            for k in child:
                Hij_tilde[:,:,i,j]+=transpose(Bij[:,:,k,i])@Hij_tilde[:,:,k,i]

    #Pre-allocate Hi0_tilde and H0j_tilde
    Hi0_tilde=zeros((6,6,n),dtype=t0.dtype) 

    #Hi0_tilde
    for i in range(n-1, -1, -1):
        Bdot=np.block([[zeros((3,3)), zeros((3,3))] , 
                       [SkewSym(t0[3:]-tL[3:,i]), zeros((3,3))]] )
        Hi0_tilde[:,:,i]=Mm_tilde[:,:,i]@Bdot 
        #Add children contributions
        child = np.nonzero(robot.con.child[:,i])[0].tolist() 
        for k in child:
            Hi0_tilde[:,:,i]+=transpose(Bij[:,:,k,i])@Hij_tilde[:,:,k,i]

    #--- C Matrix ---#
    #Pre-allocate
    Cm=zeros((n_q,n_q),dtype=t0.dtype) 
    C0m=zeros((6,n_q),dtype=t0.dtype) 
    Cm0=zeros((n_q,6),dtype=t0.dtype) 

    #Cm Matrix
    for j in range(n):
        for i in range(n):
            #Joints must not be fixed and links on the same branch
            if (robot.joints[i].type!=0 and robot.joints[j].type!=0) and (robot.con.branch[i,j]==1 or robot.con.branch[j,i]==1):
                #Compute Cm matrix
                if i<=j : 
                    #Add children contributions
                    child_con=zeros((6,6)) 
                    child = np.nonzero(robot.con.child[:,j])[0].tolist() 
                    for k in child:
                        child_con+=transpose(Bij[:,:,k,i])@Hij_tilde[:,:,k,j]
                    Cm[robot.joints[i].q_id,robot.joints[i].q_id] = transpose(pm[:,i:i+1])\
                        @(transpose(Bij[:,:,j,i])@Mm_tilde[:,:,j]@Omega[:,:,j+child_con] \
                            +Mdot_tilde[:,:,j])@pm[:,j]
                else:
                    Cm[robot.joints[i].q_id,robot.joints[i].q_id] = transpose(pm[:,i:i+1])\
                        @(Mm_tilde[:,:,i] @ Bij[:,:,i,j]@Omega[:,:,j]\
                            +Hij_tilde[:,:,i,j] + Mdot_tilde[:,:,i]) @ pm[:,j]

    #C0 matrix
    #Add children contributions
    child_con=zeros((6,6)) 
    child = np.nonzero(robot.con.child_base)[0].tolist() 
    for k in child:
        child_con+=transpose(Bi0[:,:,k])@Hi0_tilde[:,:,k]
    C0 = transpose(P0)@(M0_tilde@Omega0 + child_con + Mdot0_tilde)@P0 
    
    #C0m
    for j in range(n):
        if  robot.joints[j].type!=0:
            if j==n:
                C0m[:,robot.joints[j].q_id]=transpose(P0) @ \
                    (transpose(Bi0[:,:,j])@Mm_tilde[:,:,j]@Omega[:,:,j]\
                        + Mdot_tilde[:,:,j])@pm[:,j]
            else:
                #Add children contributions
                child_con=zeros(6,6) 
                child = np.nonzero(robot.con.child[:,j])[0].tolist() 
                for k in child:
                    child_con+=transpose(Bi0[:,:,k])@Hij_tilde[:,:,k,j]

                C0m[:,robot.jointsj.q_id]=transpose(P0)@(transpose(Bi0[:,:,j])@Mm_tilde[:,:,j]@Omega[:,:,j]\
                    +child_con + Mdot_tilde[:,:,j])@pm[:,j]
    #Cm0
    for i in range(n):
        if robot.joints[i].type!=0:
            Cm0[robot.joints[i].q_id,:]=transpose(pm[:,i])@(Mm_tilde[:,:,i]@Bi0[:,:,i]@Omega0 \
                + Hi0_tilde[:,:,i] + Mdot_tilde[:,:,i])@P0 

    return C0, C0m, Cm0, Cm


if __name__ == "__main__":
    from robot_model import urdf2robot
    from kinematics import Velocities
    
    robot, _ = urdf2robot("./matlab/SC_3DoF.urdf")
    q0 = transpose(np.array([0,0,0,1], dtype = np.float32))
    R0 = transpose(quat_DCM(transpose(q0).squeeze()))
    r0 = zeros((3,1))
    qm = zeros((robot.n_q, 1))
    u0 = np.ones((6, 1))
    um = np.ones((robot.n_q, 1))
    u0dot = np.ones((6, 1))
    umdot = np.ones((robot.n_q, 1))
    RJ,RL,rJ,rL,e,g = Kinematics(R0, r0, qm, robot) 
    Bij, Bi0, P0, pm = DiffKinematics(R0, r0, rL, e, g, robot)
    t0, tL = Velocities(Bij, Bi0, P0, pm, u0, um, robot)
    t0dot, tLdot = Accelerations(t0, tL, P0, pm, Bi0, Bij, u0, um, u0dot, umdot, robot)
    I0, Im = I_I(R0, RL, robot)
    wf0 = np.ones((6,1))
    wfm = np.ones((6, robot.n_links_joints))
    tau0 = zeros((6, 1))
    taum = np.ones((robot.n_q, 1))
    u0dot, umdot = FD(tau0, taum, wf0, wfm, t0, tL, P0, pm, I0, Im, Bij, Bi0, u0, um, robot)
    
    q = np.block([[q0], [qm]])
    qdot = np.block([[u0], [um]])
    C0, C0m, Cm0, Cm = CIM(q, qdot, robot)
    H0, H0m, Hm = GIM(q, robot)
    
    print("C0", C0)
    print("C0m", C0m)
    print("Cm0", Cm0)
    print("Cm", Cm)
    print("H0", H0)
    print("H0m", H0m)
    print("Hm", Hm)


