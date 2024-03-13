function H = getInertiaMatrix(q, robot)
    %--- Parameters ---%
    %Base and rotation
    R0=quat_DCM([q(1:4)]')';
    r0 = q(5:7);

    %Joint variables
    qm = q(8:10);

    %--- Kinematics ---%
    %Kinematics
    [~,RL,~,rL,e,g]=Kinematics(R0,r0,qm,robot);
    %Diferential Kinematics
    [Bij,Bi0,P0,pm]=DiffKinematics(R0,r0,rL,e,g,robot);
    %--- Dynamics ---%
    %Inertias in inertial frames
    [I0,Im]=I_I(R0,RL,robot);
    %Mass Composite Body matrix
    [M0_tilde,Mm_tilde] = MCB(I0,Im,Bij,Bi0,robot);
    %Generalized Inertia matrix
    [H0, H0m, Hm] = GIM(M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot);

    H = [H0, H0m; H0m', Hm];
end 