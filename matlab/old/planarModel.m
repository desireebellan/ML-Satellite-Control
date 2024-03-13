function [H, C] = planarModel(m, l, rc, Ic)

    q = sym("q", [2,1], "real");
    qdot = sym("qdot", [2,1], "real");
    m_load = sym("ml", "real");
    I_load = sym("Il", "real");
    rc_load = sym("rcl", [2,1], "real");

    Izz = zeros(2);
    Izz(1) = Ic(1) + m(1)*((l(1) + rc(1,1))^2 + rc(1,2)^2);
    Izz(2) = Ic(2) + m(2)*((l(2) + rc(2,1))^2 + rc(2,2)^2);
    Izz_load = I_load + m_load*((l(2) + rc_load(1))^2 + rc_load(2)^2);

    a = sym(zeros(4));

    a(1) = Izz(1) + (m(2)+m_load)*l(1)^2;
    a(2) = Izz(2) + Izz_load;
    a(3) = l(1)*(m(2)*(l(2) + rc(2,1)) + m_load*(l(2) + rc_load(1)));
    a(4) = l(1) * (m(2)*rc(2,2) + m_load*rc_load(2));


    m11 = a(1) + a(2) + 2*a(3)*cos(q(2)) - 2*a(4)*sin(q(2));
    m12 = a(2) + a(3)*cos(q(2)) - a(4) * sin(q(2));
    m22 = a(2);

    H = [m11 m12; m12 m22];
    H = matlabFunction(H);

    c1 = -(a(3)*sin(q(2)) + a(4)*cos(q(2)))*(2*qdot(1)*qdot(2) + qdot(2)^2);
    c2 = (a(3)*sin(q(2)) + a(4)*cos(q(2)))*qdot(1)^2;

    C = [c1;c2];
    C = matlabFunction(C);

end 