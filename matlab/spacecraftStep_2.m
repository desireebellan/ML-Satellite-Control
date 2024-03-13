function [umdot, u0dot, taum] = spacecraftStep_2(q0, qm, u0, um, robot)

    %--- Parameters ---%
    %Base position
    R0=quat_DCM([q0(1:4)]')';
    r0 = q0(5:7);

    %--- Kinematics ---%
    %Kinematics
    [~,RL,~,rL,e,g]=Kinematics(R0,r0,qm,robot);

    %Diferential Kinematics
    [Bij,Bi0,P0,pm]=DiffKinematics(R0,r0,rL,e,g,robot);

    %Velocities
    [t0,tL]=Velocities(Bij,Bi0,P0,pm,u0,um,robot);

    %--- Dynamics ---%
    %Inertias in inertial frames
    [I0,Im]=I_I(R0,RL,robot);

    %Mass Composite Body matrix
    [M0_tilde,Mm_tilde] = MCB(I0,Im,Bij,Bi0,robot);

    %Generalized Inertia matrix
    [H0, H0m, Hm] = GIM(M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot);
    H = [H0, H0m; H0m', Hm];
    
    %Generalized Convective Inertia matrix
    [C0, C0m, Cm0, Cm] = CIM(t0,tL,I0,Im,M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot);
    C = [C0, C0m; Cm0, Cm];

    qd = [0, 0, 0, 1]'
    qe = quat_error(q0, qd)
    u0dot_ = Kp*qe(1:3) - Kd*u0;
    umdot = pinv(H0m) * (-H0 * u0dot_ - C0 * u0 - C0m*um);

    wF0=zeros(6,1);
    wFm=zeros(6,robot.n_links);

    [taum, u0dot] = Floating_ID(wF0,wFm,Mm_tilde,H0,t0,tL,P0,pm,I0,Im,Bij,Bi0,u0,um,umdot,robot);

end

function qe = quat_error(q, qd)
    q_ = [-q(1:3); q(4)];
    qe = quaternion_multiply(q_, qd);
    if (q(4) < 0) 
        qe = -qe 
    end 
end

function q_result = quaternion_multiply(q1, q2)
    % Multiply two quaternions
    
    w1 = q1(4);
    v1 = q1(1:3);
    
    w2 = q2(4);
    v2 = q2(1:3);
    
    q_result = [
        w1*v2 + w2*v1 + cross(v1, v2);
        w1*w2 - dot(v1, v2);
    ];
end