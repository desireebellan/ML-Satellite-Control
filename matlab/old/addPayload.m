function [H, C] = addPayload(H, C, mp, rp, Ip)

    % symbolic payload
    m = sym("mp", "real");
    assume (m>0);
    I = sym("Ip", [3,1], "real");
    % assume(I) ?
    r = sym("rp", [3,1], "real");

    % substitute
    H = subs(H, m, mp);
    H = subs(H, r, rp);
    H = subs(H, I, Ip);

    C = subs(C, m, mp);
    C = subs(C, r, rp);
    C = subs(C, I, Ip);
end