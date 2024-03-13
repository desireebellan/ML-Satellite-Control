from math import cos, sin 
from casadi import *

# SPART LIBRARY
# Matlab library ported into python
# Attitude Transformations

def Angles321_DCM(Angles:np.ndarray):
    # Convert the Euler angles (321 sequence), x-phi, y-theta, z-psi to its DCM equivalent.
    #
    # DCM = Angles321_DCM(Angles)
    #
    # :parameters: 
    #   * Angles -- Euler angles [x-phi, y-theta, z-psi] -- [3x1].
    #
    # :return: 
    #   * DCM -- Direction Cosine Matrix -- [3x3].
    #
    # See also: :func:`src.attitude_transformations.Angles123_DCM` and :func:`src.attitude_transformations.DCM_Angles321`.

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

    phi, theta, psi = Angles[0], Angles[1], Angles[2]

    DCM=blockcat([[cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta)],
            [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), sin(phi)*cos(theta)],
            [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi), cos(phi)*cos(theta)]])
    return DCM

def Euler_DCM(e,alpha): #codegen
    #Provides the Direction Cosine Matrix (DCM) from a Euler axis e=[e1,e2,e3]
    #and angle alpha.

    #Create quaternion
    q=blockcat([[e*sin(alpha/2)],[cos(alpha/2)]])

    #Convert quaternion to DCM
    DCM = quat_DCM(q)
    return DCM

def quat_DCM(q): ##codegen
    #Provides the Direction Cosine Matrix (DCM) from a quaterionion (q)
    #
    # q = [q1;q2;q3;q4] -> With q4 being the scalar part of the quaternion.
    DCM = blockcat([ [1-2*(q[1]**2+q[2]**2), 2*(q[0]*q[1]+q[2]*q[3]), 2*(q[0]*q[2]-q[1]*q[3])],
            [2*(q[1]*q[0]-q[2]*q[3]), 1-2*(q[0]**2+q[2]**2), 2*(q[1]*q[2]+q[0]*q[3])],
            [2*(q[2]*q[0]+q[1]*q[3]), 2*(q[2]*q[1]-q[0]*q[3]), 1-2*(q[0]**2+q[1]**2)]])
    return DCM

def DCM_Angles321(DCM):
    #Convert the DCM to Euler angles (321 sequence), x-phi, y-theta, z-psi.
    #
    # This solution generates angle theta between -pi/2 and pi/2,
    # and angles phi and psi between -pi and pi.
    
    Angles = blockcat([[np.arctan2(DCM[1,2],DCM[2,2])],
                       [np.arcsin(-DCM[0,2])],
                       [np.arctan2(DCM[0,1],DCM[0,0])]])
    
    return Angles

def quat_Angles321(q):
    # Convert a quaternion to Euler angles (321 sequence), x-phi, y-theta, z-psi.
    DCM = quat_DCM(q)
    Angles = DCM_Angles321(DCM)
    return Angles

def Euler_Angles321(e, alpha):
    DCM = Euler_DCM(e, alpha)
    return DCM_Angles321(DCM)
