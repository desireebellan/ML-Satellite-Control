function rJ = joint2ee(q0, qm, robot)
    % base rotation and position w.r.t. the inertial frame
    R0=quat_DCM([q0(1:4)]')';
    r0 = q0(5:7)';
    qm = qm';
    %Kinematics
    [~,~,rJ,~,~,~]=Kinematics(R0,r0,qm,robot);
end