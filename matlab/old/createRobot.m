function robot = createRobot()

    % base

    r01 = sym("L0", [3,1], 'real');

    data.base.T_L0_J1 = sym([eye(3), r01; zeros(1,3),1]);
    data.base.mass = sym('m_base', 'real');
    data.base.I = diag(sym('I_base', [3,1], 'real'));
    

    data.n = 3;

    d = sym('d', [1, data.n], 'real');
    q = sym('qm', [1, data.n], 'real');
    b = sym("b", [3, data.n], 'real');
    m = sym('m', [1,data.n], 'real');
    I = sym('I', [3, data.n], 'real');

    mp = sym('mp', 'real');
    rp = sym('rp', [3,1], 'real');
    Ip = sym('Ip', [3,1], 'real');

    % first joint, rotational
    data.man(1).type = 1;
    data.man(1).DH.d = d(1);
    data.man(1).DH.alpha = sym(pi/2);
    data.man(1).DH.a = 0;
    data.man(1).DH.theta = q(1);
    data.man(1).b = b(1:3,1);
    data.man(1).I = diag(I(1:3,1));
    data.man(1).mass = m(1);

    % second joint, rotational
    data.man(2).type = 1;
    data.man(2).DH.d = 0;
    data.man(2).DH.alpha = 0;
    data.man(2).DH.a = d(2);
    data.man(2).DH.theta = q(2);
    data.man(2).b = b(1:3,2);
    data.man(2).I = diag(I(1:3,2));
    data.man(2).mass = m(2);

    % third joint, rotational
    data.man(3).type = 1;
    data.man(3).DH.d = 0;
    data.man(3).DH.alpha = 0;
    data.man(3).DH.a = d(3);
    data.man(3).DH.theta = q(3);
    data.man(3).b = b(1:3,3);
    data.man(3).I = diag(I(1:3,3)) + diag(Ip);
    data.man(3).mass = m(3) + mp;

    % fourt joint - ee wrist 
    %End-Effector
    data.EE.theta=sym(-pi/2);
    data.EE.d=rp;

    dhparams = [data.man(1).DH.a   data.man(1).DH.alpha	    data.man(1).DH.d   data.man(1).DH.theta;
                data.man(2).DH.a   data.man(2).DH.alpha	    data.man(2).DH.d   data.man(2).DH.theta;
                data.man(3).DH.a   data.man(3).DH.alpha	    data.man(3).DH.d   data.man(3).DH.theta];


    [robot,~] = DH_Serial2robot(data);


end