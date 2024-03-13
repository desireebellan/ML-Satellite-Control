function robot = setParam(robot, mp, rp, Ip)

    robot.links(robot.n_links_joints).mass = mp;
    %robot.links(robot.n_links_joints).inertia = diag(Ip);
    robot.links(robot.n_links_joints).inertia = Ip;
    robot.links(robot.n_links_joints).T = [[eye(3), rp];[zeros(1,3), 1]];

    robot_sym = robot;
end