function [robot, robot_sym] = symbolicRobot(filename)
    [robot, ~] = urdf2robot(filename);

    % symbolic payload
    mp = sym("mp", "real");
    assume (mp>0);
    Ip = sym("Ip", [3,1], "real");
    % assume(I) ?
    rp = sym("rp", [3,1], "real");

    robot.links(robot.n_links_joints).mass = mp;
    robot.links(robot.n_links_joints).inertia = diag(Ip);
    robot.links(robot.n_links_joints).T = [[eye(3), rp];[zeros(1,3), 1]];

    robot_sym = robot;

    % symbolic robot
    m = sym("m", [robot.n_links_joints-1, 1], "real");
    assume (m>0);
    I = sym("I", [robot.n_links_joints-1, 3], "real");
    r = sym("r", [robot.n_links_joints-1, 3], "real");

    for i=1:robot.n_links_joints-1
        robot_sym.links(i).mass = m(i,1);
        robot_sym.links(i).inertia = diag(I(i,1:3));
        robot_sym.links(i).T = [[eye(3), r(i, 1:3)'];[zeros(1,3), 1]];        

    end

    robot_sym.base_link.mass = sym("M0", "real");
    robot_sym.base_link.inertia = diag(sym("I0", [3,1], "real"));

end





