from casadi import *
from SPART_CASADI.attitude_transformations import *
from SPART_CASADI.utils import transpose, SkewSym, cross
eye, zeros = SX.eye, SX.zeros

def Kinematics(R0,r0,qm,robot):
    # Computes the kinematics -- positions and orientations -- of the multibody system.
    #
    # [RJ,RL,rJ,rL,e,g]=Kinematics(R0,r0,qm,robot)
    #
    # :parameters: 
    #   * R0 -- Rotation matrix from the base-link CCS to the inertial CCS -- [3x3].
    #   * r0 -- Position of the base-link center-of-mass with respect to the origin of the inertial frame, projected in the inertial CCS -- [3x1].
    #   * qm -- Displacements of the active joints -- [n_qx1].
    #   * robot -- Robot model (see :doc:`/Tutorial_Robot`).
    #
    # :return: 
    #   * RJ -- Joints CCS 3x3 rotation matrices with respect to the inertial CCS  -- as a [3x3xn] matrix.
    #   * RL -- Links CCS 3x3 rotation matrices with respect to the inertial CCS -- as a [3x3xn] matrix.
    #   * rJ -- Positions of the joints, projected in the inertial CCS -- as a [3xn] matrix.
    #   * rL -- Positions of the links, projected in the inertial CCS -- as a [3xn] matrix.
    #   * e -- Joint rotation/sliding axes, projected in the inertial CCS -- as a [3xn] matrix.
    #   * g -- Vector from the origin of the ith joint CCS to the origin of the ith link CCS, projected in the inertial CCS -- as a [3xn] matrix.
    #
    # Remember that all the ouput magnitudes are projected in the **inertial frame**.
    #
    # Examples on how to retrieve the results from a specific link/joint:
    #
    #   To retrieve the position of the ith link: ``rL(1:3,i)``.
    #
    #   To retrieve the rotation matrix of the ith joint: ``RJ(1:3,1:3,i)``.   
    #
    # See also: :func:`src.robot_model.urdf2robot` and :func:`src.robot_model.DH_Serial2robot`.

    '''{  
        LICENSE

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Lesser General Public License for more details.

        You should have received a copy of the GNU Lesser General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
    }'''

    #=== CODE ===#

    #--- Number of links and joints ---#
    n=robot.n_links_joints

    #--- Homogeneous transformation matrices ---#

    #Pre-allocate homogeneous transformations matrices
    TJ=[SX.zeros(4,4) for i in range(n)]
    TL=[SX.zeros(4,4) for i in range(n)]

    #--- Base-link ---#
    T0=blockcat([[R0,r0], [SX.zeros(1,3),1]])

    #--- Forward kinematics recursion ---#

    #Obtain the joints and links kinematics
    for i in range(n):
        
        #Get child joint
        cjoint=robot.joints[i]
        
        #Joint kinematics (homogeneous transformation matrix)
        if cjoint.parent_link==-1:
            #Parent link is the base-link
            TJ[cjoint.id][:,:]=T0 @ cjoint.T
            
        else:
            #Parent link is not the base-link
            TJ[cjoint.id][:,:]=TL[cjoint.parent_link][:4,:4] @ cjoint.T
        
        #Transformation due to current joint variable
        if cjoint.type==1:
            #Revolute
            T_qm=blockcat([[transpose(Euler_DCM(cjoint.axis,qm[cjoint.q_id])), zeros((3,1))], [zeros((1,3)),1]])
        elif cjoint.type==2:
            #Prismatic
            T_qm=blockcat([[eye(3),cjoint.axis*qm(cjoint.q_id)],[zeros((1,3)),1]])
        else:
            #Fixed
            T_qm=blockcat([[eye(3),zeros((3,1))],[zeros((1,3)),1]])

        #Link Kinematics (homogeneous transformation matrix)
        clink=robot.links[cjoint.child_link]
        TL[clink.id][:,:]=TJ[clink.parent_joint][:,:]@T_qm@clink.T
        

    #--- Rotation matrices, translation, position and other geometric quantities ---#

    #Pre-allocate rotation matrices, translation and positions
    RJ=[zeros(3,3) for i in range(n)]
    RL=[zeros(3,3) for i in range(n)]
    rJ=zeros((3,n))
    rL=zeros((3,n))
    #Pre-allocate rotation/sliding axis
    e=zeros((3,n))
    #Pre-allocate other geometric quantities
    g=zeros((3,n))

    #Format rotation matrices, link positions, joint axis and other geometric
    #quantities

    #Joint associated quantities
    for i in range(n):
        RJ[i][:3,:3]=TJ[i][:3,:3]
        rJ[:3,i]=TJ[i][:3,3]
        e[:3,i:i+1]=RJ[i][:3,:3]@robot.joints[i].axis

    #Link associated quantities
    for i in range(n):
        RL[i][:3,:3]=TL[i][:3,:3]
        rL[:3,i]=TL[i][:3,3]
        g[:3,i]=rL[:3,i]-rJ[:3,robot.links[i].parent_joint]

    return RJ,RL,rJ,rL,e,g 

def DiffKinematics(R0,r0,rL,e,g,robot):
    # Computes the differential kinematics of the multibody system.
    #
    # [Bij,Bi0,P0,pm]=DiffKinematics(R0,r0,rL,e,g,robot)
    # 
    # :parameters:
    #   * R0 -- Rotation matrix from the base-link CCS to the inertial CCS -- [3x3].
    #   * r0 -- Position of the base-link center-of-mass with respect to the origin of the inertial frame, projected in the inertial CCS -- [3x1].
    #   * rL -- Positions of the links, projected in the inertial CCS -- as a [3xn] matrix.
    #   * e -- Joint rotation/sliding axes, projected in the inertial CCS -- as a [3xn] matrix.
    #   * g -- Vector from the origin of the ith joint CCS to the origin of the ith link CCS, projected in the inertial CCS -- as a [3xn] matrix.
    #   * robot -- Robot model (see :doc:`/Tutorial_Robot`).
    #
    # :return:
    #   * Bij -- Twist-propagation matrix (for manipulator i>0 and j>0) -- as a [6x6xnxn] matrix.
    #   * Bi0 -- Twist-propagation matrix (for i>0 and j=0) -- as a [6x6xn] matrix.
    #   * P0 -- Base-link twist-propagation "vector" -- as a [6x6] matrix.
    #   * pm -- Manipulator twist-propagation "vector" -- as a [6xn] matrix.
    #
    # Use :func:`src.kinematics_dynamics.Kinematics` to compute ``rL,e``, and ``g``.
    #
    # See also: :func:`src.kinematics_dynamics.Kinematics` and :func:`src.kinematics_dynamics.Jacob`. 

    '''{  
        LICENSE

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Lesser General Public License for more details.

        You should have received a copy of the GNU Lesser General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
    }'''
    
    #=== CODE ===#

    #--- Number of links  ---#
    n=robot.n_links_joints

    #--- Twist-propagation matrix ---#

    #Pre-allocate Bij
    Bij=[[zeros(6,6) for j in range(n)] for i in range(n)]

    #Compute Bij
    for j in range(n):
        for i in range(n):
            if robot.con.branch[i,j]==1:
                #Links are in the same branch
                Bij[i][j][:,:]=blockcat([[eye(3), zeros((3,3))], [SkewSym(rL[:,j]-rL[:,i]), eye(3)]])
            else:
                #Links are not in the same branch
                Bij[i][j][:,:]=zeros((6,6))

    #Pre-allocate Bi0
    Bi0=[zeros((6,6)) for i in range(n)]

    #Compute Bi0
    for i in range(n):
        Bi0[i]=blockcat([[eye(3), zeros((3,3))], [SkewSym(r0-rL[:,i:i+1]), eye(3)]])

    #--- Twist-propagation "vector" ---#

    #Pre-allocate pm
    pm=zeros((6,n))

    #Base-link
    P0=blockcat([[R0,zeros((3,3))], [zeros((3,3)), eye(3)]])

    #Forward recursion to obtain the twist-propagation "vector"
    for i in range(n):
        if robot.joints[i].type==1:
            #Revolute joint
            pm[:,i:i+1]=blockcat([[e[:,i:i+1]], [cross(e[:,i],g[:,i])]])
        elif robot.joints[i].type==2:
            #Prismatic joint
            pm[:,i]=blockcat([[zeros((3,1))], [e[:,i]]])
        elif robot.joints[i].type==0:
            #Fixed joint
            pm[:,i:i+1]=zeros((6,1))
    return Bij,Bi0,P0,pm

def Velocities(Bij,Bi0,P0,pm,u0,um,robot):
    # Computes the operational-space velocities of the multibody system.
    #
    # [t0,tL]=Velocities(Bij,Bi0,P0,pm,u0,um,robot)
    #
    # :parameters:
    #   * Bij -- Twist-propagation matrix (for manipulator i>0 and j>0) -- as a [6x6xn] matrix.
    #   * Bi0 -- Twist-propagation matrix (for i>0 and j=0) -- as a [6x6xn] matrix.
    #   * P0 -- Base-link twist-propagation "vector" -- as a [6x6] matrix.
    #   * pm -- Manipulator twist-propagation "vector" -- as a [6xn] matrix.
    #   * u0 -- Base-link velocities [\omega,rdot]. The angular velocity is projected in the body-fixed CCS, while the linear velocity is projected in the inertial CCS -- [6x1].
    #   * um -- Joint velocities -- [n_qx1].
    #   * robot -- Robot model (see :doc:`/Tutorial_Robot`).
    #
    # :return:
    #   * t0 -- Base-link twist [\omega,rdot], projected in the inertial CCS -- as a [6x1] matrix.
    #   * tL -- Manipulator twist [\omega,rdot], projected in the inertial CCS -- as a [6xn] matrix.
    #
    # Use :func:`src.kinematics_dynamics.DiffKinematics` to compute ``Bij``, ``Bi0``, ``P0``, and ``pm``.
    #
    # See also: :func:`src.kinematics_dynamics.Jacob`


    '''{  
        LICENSE

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Lesser General Public License for more details.

        You should have received a copy of the GNU Lesser General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
    }'''

    #=== CODE ===#

    #--- Number of links and joints ---#
    n=robot.n_links_joints

    #Pre-allocate
    tL=zeros((6,n))

    #Base-link
    t0=P0 @ u0

    #Forward recursion to obtain the twist
    for i in range(n):     
        if robot.joints[i].parent_link==-1:
            #First link
            tL[:,i:i+1]=Bi0[i] @ t0
        else:
            #Rest of the links
            tL[:,i]=Bij[i][i-1] @ tL[:,i-1]
        
        #Add joint contribution
        if robot.joints[i].type!=0:
            tL[:,i:i+1]=tL[:,i:i+1]+pm[:,i:i+1] @ um[robot.joints[i].q_id:robot.joints[i].q_id+1]
    
    return t0, tL

def Accelerations(t0,tL,P0,pm,Bi0,Bij,u0,um,u0dot,umdot,robot):
    # Computes the operational-space accelerations (twist-rate) of the multibody system.
    #
    # [t0dot,tLdot]=Accelerations(t0,tL,P0,pm,Bi0,Bij,u0,um,u0dot,umdot,robot)
    #
    # :parameters: 
    #   * t0 -- Base-link twist [\omega,rdot], projected in the inertial CCS -- as a [6x1] matrix.
    #   * tL -- Manipulator twist [\omega,rdot], projected in the inertial CCS -- as a [6xn] matrix.
    #   * Bij -- Twist-propagation matrix (for manipulator i>0 and j>0) -- as a [6x6xn] matrix.
    #   * Bi0 -- Twist-propagation matrix (for i>0 and j=0) -- as a [6x6xn] matrix.
    #   * P0 -- Base-link twist-propagation "vector" -- as a [6x6] matrix.
    #   * pm -- Manipulator twist-propagation "vector" -- as a [6xn] matrix.
    #   * u0 -- Base-link velocities [\omega,rdot]. The angular velocity is projected in the body-fixed CCS, while the linear velocity is projected in the inertial CCS -- [6x1].
    #   * um -- Joint velocities -- [n_qx1].
    #   * u0dot -- Base-link accelerations [\omegadot,rddot]. The angular acceleration is projected in a body-fixed CCS, while the linear acceleration is projected in the inertial CCS -- [6x1].
    #   * umdot -- Manipulator joint accelerations -- [n_qx1].
    #   * robot -- Robot model (see :doc:`/Tutorial_Robot`).
    #
    # :return: 
    #   * t0dot -- Base-link twist-rate vector \omegadot,rddot], projected in inertial frame -- as a [6x1] matrix.
    #   * tLdot -- Manipulator twist-rate vector \omegadot,rddot], projected in inertial frame -- as a [6xn] matrix.
    #
    # See also: :func:`src.kinematics_dynamics.Jacobdot`. 

    '''{  
        LICENSE

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Lesser General Public License for more details.

        You should have received a copy of the GNU Lesser General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
    }'''

    #=== CODE ===#

    #--- Number of links and Joints ---#
    n=robot.n_links_joints

    #--- Omega matrices ---#
    #Base-link
    Omega0=blockcat([[SkewSym(t0[:3]), zeros((3,3))],
                    [zeros((3,3)), zeros((3,3))]])

    #Pre-allocate
    Omegam=[zeros((6,6)) for i in range(n)]

    #Compute Omega for manipulator
    for i in range(n):
        Omegam[i]=blockcat([[SkewSym(tL[:3,i]), zeros((3,3))],
                        [zeros((3,3)), SkewSym(tL[:3,i])]])

    #--- Twist Rate ---#
    #Base-link
    t0dot = Omega0@P0@u0+P0@u0dot

    #Pre-allocate
    tLdot=zeros((6,n))

    #Forward recursion
    for i in range(n):
        
        if robot.joints[i].parent_link==-1:
            #First Link
            tLdot[:,i:i+1] = Bi0[i] @ t0dot + blockcat([[zeros((3,6))],[SkewSym(t0[3:]-tL[3:,i:i+1]),zeros((3,3))]]) @ t0
        else:
            #Rest of the links.
            tLdot[:,i] = Bij[i][robot.joints[i].parent_link] @ tLdot[:,robot.joints[i].parent_link] \
                + blockcat([[zeros((3,6))], [SkewSym(tL[3:,robot.joints[i].parent_link]-tL[3:,i]), zeros((3,3))]]) @ tL[:,robot.joints[i].parent_link]
        
        #Add joint contribution
        if robot.joints[i].type!=0:
            tLdot[:,i] += Omegam[i] @ pm[:,i] * um[robot.joints[i].q_id] + pm[:,i] * umdot[robot.joints[i].q_id]
    
    return t0dot, tLdot

def Joints2EE(qm, q0, robot, u0 = None, um = None):
    # return the position and velocity of the end-effector with respect to the inertial CSS
    #R0 = transpose(quat_DCM(transpose(q0).squeeze()))
    #r0 = q0[4:, :]
    
    R0 = transpose(Angles321_DCM(transpose(q0[:3,:].reshape((-1,1)))))
    r0 = q0[3:, :]
    RJ,RL,_, rL,e,g = Kinematics(R0, r0, qm, robot)
    if u0 is None or um is None:
        return rL[:,-1]
    else:
        Bij, Bi0, P0, pm = DiffKinematics(R0, r0, rL, e, g, robot)
        _, tL = Velocities(Bij, Bi0, P0, pm, u0, um, robot)
        return rL[:,-1], tL[:,-1]


if __name__ == "__main__":
    from robot_model import urdf2robot
    
    
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
    

