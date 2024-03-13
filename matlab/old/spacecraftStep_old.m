function data = spacecraftStep_old(H_, C_, tau0, taum, data, robot, dt)

    % symbolic data
    %Base position
    q0_ = sym('q0', [7, 1], 'real');
    
    %Joint variables
    qm_=sym('qm', [robot.n_q,1], 'real');
    
    %Velocities
    u0_=sym('u0',[6,1],'real');
    um_=sym('um',[robot.n_q,1],'real');

    % substitute
    H = subs(H_, q0_, data.q0);
    H = subs(H, qm_, data.qm);

    C = subs(C_, q0_, data.q0);
    C = subs(C, qm_, data.qm);
    C = subs(C, u0_, data.u0);
    C = subs(C, um_, data.um);

    H = double(H);
    C = double(C);


    udot = H\([tau0; taum] - C * [data.u0;data.um]);

    data.u0dot = udot(1:6);
    data.umdot = udot(7:9);

    %--- Integration ---%
    % manipulator
    data.um = data.um + data.umdot * dt;
    data.qm = data.qm + data.um * dt;

    % base
    data.u0 = data.u0 + data.u0dot * dt;
    q0dot = 1/2 * QuaternionProduct([data.u0(1:3);0], data.q0(1:4));
    data.q0(1:4) = data.q0(1:4) + q0dot * dt;  % angular velocities
    data.q0(5:7) = data.q0(5:7) + data.u0(4:6) * dt; % linear velocities

end