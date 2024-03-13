function C = getCouplingMatrix(q, u, robot)
    %--- Parameters ---%
    %Base position
    R0=quat_DCM([q(1:4)]')';
    r0 = q(5:7);


    %Joint variables
    qm=q(8:7 + robot.n_q);

    %Velocities
    u0=u(1:6);
    um=u(7:6 + robot.n_q);

    %--- Kinematics ---%
    %Kinematics
    [~,RL,~,rL,e,g]=Kinematics(R0,r0,qm,robot);
    %Diferential Kinematics
    [Bij,Bi0,P0,pm]=DiffKinematics(R0,r0,rL,e,g,robot);
    %Velocities
    [t0,tm]=Velocities(Bij,Bi0,P0,pm,u0,um,robot);
    %--- Dynamics ---%
    %Inertias in inertial frames
    [I0,Im]=I_I(R0,RL,robot);
    %Mass Composite Body matrix
    [M0_tilde,Mm_tilde] = MCB(I0,Im,Bij,Bi0,robot);

    %Generalized Convective Inertia matrix
    [C0, C0m, Cm0, Cm] = CIM(t0,tm,I0,Im,M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot);

    C = [C0, C0m; Cm0, Cm];
    C = C * u; 
end