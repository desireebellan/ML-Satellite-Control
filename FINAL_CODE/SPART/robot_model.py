from lxml import etree as ET
from SPART.attitude_transformations import Angles321_DCM
from utils import transpose, AttributeDict
import numpy as np
from casadi.casadi import SX, Sparsity

# SPART LIBRARY
# Matlab library ported into python
# Robot Model

def ConnectivityMap(robot):
    # Produces the connectivity map for a robot model.
    #
    # [branch,child,child_base]=ConnectivityMap(robot)
    #
    # :parameters:
    #   * robot -- Robot model (see :doc:`/Tutorial_Robot`).
    #
    # :return:
    #   * branch -- Branch connectivity map. This is a [nxn] lower triangular matrix. If the i,j element is 1 it means that the ith and jth link are on the same branch. 
    #   * child -- A [nxn] matrix. If the i,j element is 1, then the ith link is a child of the jth link.
    #   * child_base -- A [nx1] matrix. If the ith element is 1, the ith link is connected to the base-link.
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

    #Pre-allocate branch connectivity map
    branch=np.zeros((robot.n_links_joints,robot.n_links_joints))

    #Populate branch connectivity map
    for i in range(robot.n_links_joints-1, -1, -1):
        last_parent_link=i
        branch[i,i]=1
        #Descent through the branch until the base, populating the connectivity map
        while True:
            parent_link=robot.joints[robot.links[last_parent_link].parent_joint].parent_link
            if parent_link==-1: break
            branch[i,parent_link]=1
            last_parent_link = parent_link

    #Populate child map
    child=np.zeros((robot.n_links_joints,robot.n_links_joints), dtype = np.int32)
    child_base=np.zeros((robot.n_links_joints,1), dtype = np.int32)
    for i in range(robot.n_links_joints-1, -1, -1):
        parent_link=robot.joints[robot.links[i].parent_joint].parent_link
        if parent_link!=-1: child[i,parent_link]=1
        else : child_base[i]=1
        
    return branch,child,child_base 

def urdf2robot(filename,verbose_flag = False):   

    # Creates a SPART robot model from a URDF file.
    #
    # [robot,robot_keys] = urdf2robot(filename,verbose_flag)
    # 
    # :parameters: 
    #   * filename -- Path to the URDF file.
    #   * verbose_flag -- True for verbose output (default False).
    #
    # :return:
    #   * robot -- Robot model (see :doc:`/Tutorial_Robot`).
    #   * robot_keys -- Links/Joints name map (see :doc:`/Tutorial_Robot`).
    #
    # This function was inspired by:
    # https://github.com/jhu-lcsr/matlab_urdf/blob/master/load_ne_id.m

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
    parser = ET.XMLParser()
    tree = ET.parse(filename, parser=parser)
    robot_urdf = tree.getroot()
    
    robot = AttributeDict()
    #Get robot name
    robot.name = robot_urdf.attrib['name']
    
    # Get links and joints from URDF file
    links_urdf = robot_urdf.findall("link")
    joints_urdf = robot_urdf.findall("joint")
    
    #Create number of joint variables (to be poulated later)
    robot.n_q=0

    #Count links and joints
    robot.n_links_joints = len(links_urdf) 

    #Create temporary link and joint maps
    links = dict()
    joints = dict()
    
    #Display data
    if verbose_flag: print('Number of links: {} (including the base link)\n'.format(robot.n_links_joints))
    
    #Iterate over links
    for k in range(robot.n_links_joints):
        #Create basic structure with default values
        link = AttributeDict()
        link_xml = links_urdf[k]
        link.name = link_xml.attrib['name'] 
        link.T = np.block([[np.eye(3), np.zeros((3,1))],[np.zeros((1,3)), 1]])
        link.parent_joint = []
        link.child_joint = []
        
        #Grab inertial properties
        inertial = link_xml.find("inertial")
        
        #Grab origin properties
        origin = inertial.find("origin")
        
        xyz = origin.attrib["xyz"]
        rpy = origin.attrib["rpy"]

        if xyz:
            link.T[:3,3] = np.array([eval(x) for x in xyz.split(" ")])
        if rpy:
            link.T[:3,:3] = Angles321_DCM(transpose(np.array([eval(x) for x in rpy.split(" ")])))   
        
        #Mass
        mass = inertial.find("mass")
        link.mass = eval(mass.attrib["value"])
        
        #Inertia
        inertia = inertial.find("inertia")
        ixx = eval(inertia.attrib['ixx'])
        iyy = eval(inertia.attrib['iyy'])
        izz = eval(inertia.attrib['izz'])
        ixy = eval(inertia.attrib['ixy'])
        iyz = eval(inertia.attrib['iyz'])
        ixz = eval(inertia.attrib['ixz'])
        link.inertia = np.block([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        
        #Store this link in the links map
        links[link.name]=link
        
    #Iterate over joints
    for k in range(robot.n_links_joints-1):
        #Create basic structure with default values
        joint = AttributeDict()
        joint_xml = joints_urdf[k]
        joint.name = joint_xml.attrib["name"]
        joint.type_name = joint_xml.attrib["type"]
        joint.parent_link = ""
        joint.child_link = ""
        joint.T = np.block([[np.eye(3),np.zeros((3,1))], [np.zeros((1,3)),1]])
        
        if joint.type_name in {"revolute", "continuous"}: joint.type = 1
        elif joint.type_name == "prismatic": joint.type = 2 
        elif joint.type_name == "fixed":
            joint.type = 0
            joint.axis = np.zeros((3,1))
        else: raise ValueError("Joint type {} not supported.".format(joint.type_name))
        
        #Get origin properties
        origin = joint_xml.find("origin")
        xyz = origin.attrib["xyz"]
        rpy = origin.attrib["rpy"]
        if xyz: joint.T[:3,3] = np.array([eval(x) for x in xyz.split(" ")])
        if rpy: joint.T[:3,:3] = Angles321_DCM(transpose(np.array([eval(x) for x in rpy.split(" ")])))
        
        #Get rotation/sliding axis
        axis = joint_xml.find("axis")
        if axis is not None: 
            xyz = axis.attrib["xyz"]
            joint.axis = transpose(np.array([eval(x) for x in xyz.split(" ")]))
        elif axis is None and joint.type !=0:
            #Moving joints need a rotation/sliding axis.
            raise Exception(joint.name + " is a moving joint and requires a joint axis.")
        
        #Get parent link name
        parent = joint_xml.find("parent") 
        if parent is not None:
            joint.parent_link = parent.attrib["link"]
            
            #Store the joint name in the parent link
            parent=links[joint.parent_link]
            parent.child_joint.append(joint.name)
            links[joint.parent_link] = parent
            
        #Get child link name
        child = joint_xml.find("child")
        if child is not None:
            joint.child_link = child.attrib["link"]
            
            #Store the joint name in the child link
            child =links[joint.child_link]
            child.parent_joint.append(joint.name)
            links[joint.child_link] = child

        #Correct homogeneous transformation so that it is from previous link
        #inertial
        joint.T = np.linalg.solve(parent.T, joint.T)
        
        #Store this joint in the joints map
        joints[joint.name]=joint
        
    # Find the base link
    exist_base = False
    for link_name in links.keys():
        if not links[link_name].parent_joint:
            exist_base = True
            base_link = link_name
            if verbose_flag: print('Base link:',base_link)

    #There needs to be a root link
    if not exist_base:
        raise Exception('Robot has no single base link!')

    #Structure links and joints map into a structure and create a map with
    #names and IDs.

    #Create ID maps
    robot_keys = AttributeDict()
    robot_keys.link_id={}
    robot_keys.joint_id={}
    robot_keys.q_id={}

    #Remove base link from the number of total links
    robot.n_links_joints=robot.n_links_joints-1

    #Create links and joints stucture
    if robot.n_links_joints>0:
        robot.links = [AttributeDict() for n in range(robot.n_links_joints)]
        robot.joints = [AttributeDict() for n in range(robot.n_links_joints)]

    #Save base link on its own structure
    clink=links[base_link]
    robot.base_link = AttributeDict()
    robot.base_link.mass=clink.mass 
    robot.base_link.inertia=clink.inertia

    #Assign base ID
    robot_keys.link_id[base_link]=-1

    #Add links and joints into the structure with the standard numbering
    nl=-1  #Link index
    nj=-1  #Joint index
    nq=0  #Joint variable index 

    #--- Recursive function ---#
    def urdf2robot_recursive(robot,robot_keys,links,joints,child_joint,nl,nj,nq):
        #Copy the elements of child joint
        robot.joints[nj].id=nj
        robot.joints[nj].type=child_joint.type
        #Assign joint variable if joint is revolute or prismatic
        if child_joint.type:
            robot.joints[nj].q_id=nq 
            robot_keys.q_id[child_joint.name]=nq
            nq=nq+1
        else:
            #Fixed joint assign -1
            robot.joints[nj].q_id=-1

        robot.joints[nj].parent_link=robot_keys.link_id[child_joint.parent_link]
        robot.joints[nj].child_link=nl
        robot.joints[nj].axis=child_joint.axis 
        robot.joints[nj].T=child_joint.T 

        #Copy elements of child link
        clink=links[child_joint.child_link]
        robot.links[nl].id=nl 
        robot.links[nl].parent_joint=nj 
        robot.links[nl].T=clink.T 
        robot.links[nl].mass=clink.mass 
        robot.links[nl].inertia=clink.inertia 

        #Assign ID
        robot_keys.joint_id[child_joint.name]=nj 
        robot_keys.link_id[clink.name]=nl 

        #Recursively scan through the tree structure
        for n in clink.child_joint:
            robot,robot_keys,nl,nj,nq=urdf2robot_recursive(robot,robot_keys,links,joints,joints[n],nl+1,nj+1,nq)     
        
        return robot,robot_keys,nl,nj,nq
    
    #Recursively scan through the tree structure
    for n in clink.child_joint:
        robot,robot_keys,nl,nj,nq = urdf2robot_recursive(robot,robot_keys,links,joints,joints[n],nl+1,nj+1,nq)

    #Populate number of joint variables
    robot.n_q=nq
    if verbose_flag: print('Number of joint variables:',robot.n_q)

    #--- Add Conectivity Map ---#
    branch,child,child_base=ConnectivityMap(robot)
    robot.con = AttributeDict()
    robot.con.branch=branch
    robot.con.child=child
    robot.con.child_base=child_base
    
    return robot, robot_keys

def setParams(params:dict, robot:AttributeDict):
    if "mp" not in params.keys() or "rp" not in params.keys() or "Ip" not in params.keys():
        raise Exception("Params in the wrong format!")
    robot.links[-1].mass = params["mp"]
    robot.links[-1].inertia = params["Ip"]
    robot.links[-1].T = np.block([[np.eye(3), params["rp"].reshape((3, 1))],[np.zeros((1,3)), 1]])
    return robot

if __name__ == "__main__":
    from numpy import zeros
    filename = "./matlab/SC_3DoF.urdf"
    robot, robot_keys = urdf2robot(filename)
    params = {"mp": 10.0 , "rp": zeros((3,1)).astype(np.float32), "Ip": zeros((3,3)).astype(np.float32)}
    robotp = setParams(params, robot)









