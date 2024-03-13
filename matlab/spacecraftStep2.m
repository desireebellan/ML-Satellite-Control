function data = spacecraftStep2(tau, data, dt, n)

    robot = evalin('base','robotp');
    
    for i = 1:n
        [umdot, u0dot, taum] = spacecraftStep_2(data.q0, data.qm, data.u0, data.um, robot);
        data = integrate2(data, umdot, u0dot, taum, dt);
    end
end