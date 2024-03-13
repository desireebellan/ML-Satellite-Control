function data = planarStep(m, rc, Ic, data, H,C,tau, dt)
    data.qdotdot = H(Ic, m, data.q(2), rc(1), rc(2))\(tau - C(m, data.q(2), data.qdot(1), data.qdot(2), rc(1), rc(2)));
    data.qdot = min(max(data.qdot + data.qdotdot*dt, -200*dt), 200*dt);
    idx = find(data.qdot == 200*dt | data.qdot == -200*dt);
    data.qdotdot(idx) = 0.0;
    data.q = min(max(data.q + data.qdot*dt, -pi/2), pi);
    idx = find(data.q == pi | data.q == -pi/2);
    data.qdot(idx) = 0.0;

end