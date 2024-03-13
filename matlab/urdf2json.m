function [robot, robot_keys] = urdf2json(filename)
    % read and encode urdf
    [robot, robot_keys] = urdf2robot(filename);
    robot = jsonencode(robot);
    robot_keys = jsonencode(robot_keys);
end