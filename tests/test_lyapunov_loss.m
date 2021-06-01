function [] = test_lyapunov_loss()
    
% AXM + MXA = C
% A and M are SPD > 0.
% C is symmetric, but low-rank.
% Loss is the one proposed by Vandereycken and Vandewalle
% Paper on the cost function to use is at
% https://epubs.siam.org/doi/abs/10.1137/090764566?journalCode=sjmael

    %rng('default');
    %rng(0);
    
    
    
    symm = @(Z) 0.5*(Z+Z');
    
    data_choice = 'Ex2Low'; % 'Ex1Full', 'Ex1Low', 'Ex2Full', 'Ex2Low';
    solver_choice = 'TR';
    
    r = 10;
    switch data_choice
        % Example 1 with full rank
        case 'Ex1Full'
            m = 49;
            A = sp_laplace(sqrt(m));
            A = sparse(A);
            Xorg = randn(m);
            Xorg = Xorg'*Xorg; 
            C = A*Xorg + Xorg*A;
            M = eye(m);
        case 'Ex1Low'
            m = 49;
            A = sp_laplace(sqrt(m));
            A = sparse(A);
            Xorg = zeros(1, m); Xorg(m-r+1:m) = 1;
            Xorg = diag(Xorg);
            C = A*Xorg + Xorg*A;
            M = eye(m);
        case 'Ex2Full'
            m = 50;
            Ac = [2.5, -1, zeros(1,m-2)];
            Ar = [2.5, 1, 1, 1, 1, zeros(1, m-5)];
            A = sparse(toeplitz(Ac, Ar));
            A = symm(A);
            Xorg = randn(m);
            Xorg = Xorg'*Xorg;
            C = A*Xorg + Xorg*A;
            M = eye(m);
        case 'Ex2Low'
            m = 50;
            Ac = [2.5, -1, zeros(1,m-2)];
            Ar = [2.5, 1, 1, 1, 1, zeros(1, m-5)];
            A = sparse(toeplitz(Ac, Ar));
            A = symm(A);
            Xorg = zeros(1, m); Xorg(m-r+1:m) = 1;
            Xorg = diag(Xorg);
            C = A*Xorg + Xorg*A;
            M = eye(m);
    end
    
    
    
    % Problem define
    problemAI.M = sympositivedefinitefactory(m);    
    problemAI.cost = @cost;
    problemAI.egrad = @egrad;
    problemAI.ehess = @ehess;
    
    problemBW.M = sympositivedefiniteBWfactory(m);    
    problemBW.cost = @cost;
    problemBW.egrad = @egrad;
    problemBW.ehess = @ehess;
   
    
    % for BW and AI
    function f = cost(X)
        f = trace(X*A*X*M) - trace(X*C);
    end
    
    function g = egrad(X)
        g = A*X*M + M*X*A - C;
    end
    
    function gdot = ehess(X, Xdot)
        gdot = A*Xdot*M + M*Xdot*A;
    end

    function mydist = disttosol(X)
        mydist = norm(X - Xorg, 'fro');
    end
    function loss = myloss(X);
        loss = norm(A*X*M + M*X*A -C, 'fro');
    end
    
    function stats = computestats(problem, X, stats)
        stats.disttosol = disttosol(X);
        stats.loss = myloss(X);
    end

    % for LE
    problemLE.M = symmetricfactory(m);
    problemLE.cost = @costLE;
    problemLE.egrad = @egradLE;
    
    function f = costLE(S)
        f = cost(expm(S)); 
    end

    function g = egradLE(S)
        R = egrad(expm(S)); % (P.*expm(S) - PA) .* P; % derivative wrt X or expm(S).
        g = symm(dexpm(S, R)); % derivatie wrt S.
    end

    function stats = computestatsLE(problem, S, stats)
        stats.disttosol = disttosol(expm(S));
        stats.loss = myloss(expm(S));
    end

    
    % solve 
    Xinitial = eye(m);
    Sinitial = logm(Xinitial);
    options.maxiter = 100;
    options.tolgradnorm = 1e-8;
    options.statsfun = @computestats;
    
    switch solver_choice
        case 'TR'
            [M_AI, ~, info_AI, ~] = trustregions(problemAI, Xinitial, options);
            [M_BW, ~, info_BW, ~] = trustregions(problemBW, Xinitial, options);
            options.statsfun = @computestatsLE;
            [M_LE, ~, info_LE, ~] = trustregions(problemLE, Sinitial, options);
            
        case 'SD'
            [M_AI, ~, info_AI, ~] = steepestdescent(problemAI, Xinitial, options);
            [M_BW, ~, info_BW, ~] = steepestdescent(problemBW, Xinitial, options);  
            options.statsfun = @computestatsLE;
            [M_LE, ~, info_LE, ~] = steepestdescent(problemLE, Sinitial, options);
            
        case 'CG'
            [M_AI, ~, info_AI, ~] = conjugategradient(problemAI, Xinitial, options);
            [M_BW, ~, info_BW, ~] = conjugategradient(problemBW, Xinitial, options);
            options.statsfun = @computestatsLE;
            [M_LE, ~, info_LE, ~] = conjugategradient(problemLE, Sinitial, options);
        
    end    

    
    % plot
    lw = 4.0;
    ms = 7.0;
    fs = 25;
    
    if strcmp(solver_choice, 'TR')
        h2 = figure();
        plot(221);
        semilogy(cumsum([info_BW.numinner]), [info_BW.disttosol], '-o', 'color', 'b', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
        semilogy(cumsum([info_AI.numinner]), [info_AI.disttosol], '-+', 'color', 'r', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
        semilogy(cumsum([info_LE.numinner]), [info_LE.disttosol], '-x', 'color', 'm', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
        hold off;
        ax1 = gca;
        set(ax1,'FontSize', fs);
        set(h2,'Position',[100 100 600 500]);
        ax1.XAxis.Exponent = 0;
        xtickformat(ax1,'%,5.0g');
        xlabel('Inner iterations (cumsum)', 'fontsize', fs);
        ylabel('Distance to solution', 'fontsize', fs);
        legend('BW', 'AI', 'LE');
        
    else
        h2 = figure();
        plot(221);
        semilogy([info_BW.iter], [info_BW.disttosol], '-o', 'color', 'b', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
        semilogy([info_AI.iter], [info_AI.disttosol], '-+', 'color', 'r', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
        semilogy([info_LE.iter], [info_LE.disttosol], '-x', 'color', 'm', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
        hold off;
        hold off;
        ax1 = gca;
        set(ax1,'FontSize', fs);
        set(h2,'Position',[100 100 600 500]);
        xlabel('Iterations', 'fontsize', fs);
        ylabel('Distance to solution', 'fontsize', fs);
        legend('BW', 'AI', 'LE');
    end
    
    % helper function
    function A = sp_laplace(n)
        % Generates sparse 2D discrete Laplacian matrix of dimension n^2.
        r = zeros(1,n); %
        r(1:2) = [2, -1];
        T = toeplitz(r);
        E = speye(n); %
        A = kron(T, E) + kron(E, T);
    end
    
end
