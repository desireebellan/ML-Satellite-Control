function data = spacecraftStep(taum, data, robot, dt)

robot = jsondecode(robot);
data = jsondecode(data);

%% forward dynamics
% base torque
tau0 = zeros(6,1);

%--- Kinematics ---%
R0=quat_DCM([data.q0(1:4)]')';
r0 = data.q0(5:7);

[~,RL,~,rL,e,g]=Kinematics(R0,r0,data.qm,robot);

%--- Differential Kinematics ---%
%Differential kinematics
[Bij,Bi0,P0,pm]=DiffKinematics(R0,r0,rL,e,g,robot);
%Velocities
[t0,tm]=Velocities(Bij,Bi0,P0,pm,data.u0,data.um,robot);

%--- Inertia Matrices ---%
%Inertias in inertial frames
[I0,Im]=I_I(R0,RL,robot);
%Mass Composite Body matrix
[M0_tilde,Mm_tilde]=MCB(I0,Im,Bij,Bi0,robot);
%Generalized Inertia matrix
[H0, H0m, Hm] = GIM(M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot);
%Generalized Convective Inertia matrix
[C0, C0m, Cm0, Cm] = CIM(t0,tm,I0,Im,M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot);

H = [H0, H0m ; H0m', Hm];
C = [C0 C0m; Cm0 Cm];

udot = H\([tau0; taum] - C * [data.u0;data.um]);

data.u0dot = udot(1:6);
data.umdot = udot(7:end);

%--- Integration ---%
% manipulator
data.um = data.um + data.umdot * dt;
data.qm = data.qm + data.um * dt;

% base
data.u0 = data.u0 + data.u0dot * dt;
q0dot = 1/2 * QuaternionProduct([data.u0(1:3);0], data.q0(1:4));
data.q0(1:4) = data.q0(1:4) + q0dot * dt;  % angular velocities
data.q0(5:7) = data.q0(5:7) + data.u0(4:6) * dt; % linear velocities

data = jsonencode(data);
end