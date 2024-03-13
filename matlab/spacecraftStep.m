function data = spacecraftStep(tau0, taum, data, dt, n)

    robot = evalin('base','robotp');
    
    for i = 1:n
        [H,C] = spacecraftStep_(data.q0, data.qm, data.u0, data.um, robot);
        data = integrate(H, C, tau0, taum, data, dt);
    end
end
