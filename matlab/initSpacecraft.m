function [H, C] = initSpacecraft(robot)
%--- Parameters ---%
%Base position
q0 = sym('q0', [7, 1], 'real');
R0=quat_DCM([q0(1:4)]')';
r0 = q0(5:7);


%Joint variables
qm=sym('qm',[robot.n_q,1],'real');

%Velocities
u0=sym('u0',[6,1],'real');
um=sym('um',[robot.n_q,1],'real');

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
%Generalized Inertia matrix
[H0, H0m, Hm] = GIM(M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot);

H = [H0, H0m; H0m', Hm];
H = subs(H, [sin(qm./2.0).*cos(qm./2.0), sin(qm./2.0).^2.*2 - 1.0], [1.0/2.0*sin(qm), -cos(qm)]);

% inertial coefficients
a0 = sym("a0", [1,3], "real");
H = subs(H, (q0(1).^2.*2.0+q0(3).^2.*2.0-1.0), a0(1));

%Generalized Convective Inertia matrix
[C0, C0m, Cm0, Cm] = CIM(t0,tm,I0,Im,M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot);

C = [C0, C0m; Cm0, Cm];

end