function data = integrate(H, C, tau0, taum, data, dt)

    data.tau = [tau0;taum];

    udot = H\([tau0; taum] - C * [data.u0;data.um]);

    data.u0dot = udot(1:6);
    data.umdot = udot(7:9);

    %--- Integration ---%
    % manipulator
    data.um = data.um + data.umdot * dt;
    data.qm = data.qm + data.um * dt;
    % normalize
    data.qm = mod(data.qm, 2*pi);

    % base
    data.u0 = data.u0 + data.u0dot * dt;

    %attitude
    %q0dot = 1/2 * QuaternionProduct([data.u0(1:3);0], data.q0(1:4));
    q0dot = 1/2 * quaternion_multiply([data.u0(1:3);0], data.q0(1:4));
    q0 = data.q0(1:4) + q0dot * dt;
    data.q0(1:4) = q0/norm(q0);
    %data.q0(1:4) = apply_angular_velocity(data.q0(1:4), data.u0(1:3), dt);

    % position
    data.q0(5:7) = data.q0(5:7) + data.u0(4:6) * dt; % linear velocities

function q_new = apply_angular_velocity(q, w, dt)
    % Apply angular velocity to compute the updated quaternion
    
    % Normalize the quaternion
    q = q / norm(q);
    
    % Extract the components of the quaternion
    q0 = q(4);
    q_vec = q(1:3);
    
    % Compute the rotation angle and axis
    angle = norm(w) * dt;
    axis = w / norm(w);
    
    % Compute the quaternion exponential
    quat_exp = [axis*sin(angle/2); cos(angle/2)];
    
    % Multiply the quaternions
    q_new = quaternion_multiply(quat_exp, q);
    
    % Normalize the resulting quaternion
    q_new = q_new / norm(q_new);
end

function q_result = quaternion_multiply(q1, q2)
    % Multiply two quaternions
    
    w1 = q1(4);
    v1 = q1(1:3);
    
    w2 = q2(4);
    v2 = q2(1:3);
    
    q_result = [
        w1*v2 + w2*v1 - cross(v1, v2);
        w1*w2 - dot(v1, v2);
    ];
end

function x = clip(x, min, max)
    x(x>max) = max;
    x(x<min) = min;
end


end 