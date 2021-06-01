function [] = test_linear_loss()
    
    % loss f(A) = || P.*A - PA  ||_F^2

    %rng('default');
    %rng(0);
    
    symm = @(Z) 0.5*(Z+Z');
    
    % parameters
    n = 50;
    CN = 10;
    data_choice = 'full'; % 'full' or 'sparse'
    solver_choice = 'TR';
        
    % generate true A
    D = 1000*diag(logspace(-log10(CN), 0, n)); fprintf('Exponential decay of singular values with CN %d.\n \n\n', CN);
    [Q, R] = qr(randn(n)); %#ok
    A = Q*D*Q';
    
    % generate P
    switch data_choice
        case 'sparse'            
            fraction = 0.5;
            P = rand(n,n);
            P = 0.5*(P+P');
            P = (P <= fraction);            
        case 'full'
            P = ones(n);
    end
    
    % Hence, we know the nonzero entries in PA:
    PA = P.*A;
    
   
    % Problem define
    problemAI.M = sympositivedefinitefactory(n);    
    problemAI.cost = @cost;
    problemAI.egrad = @egrad;
    problemAI.ehess = @ehess;
    
    problemBW.M = sympositivedefiniteBWfactory(n);    
    problemBW.cost = @cost;
    problemBW.egrad = @egrad;
    problemBW.ehess = @ehess;
    
        
    % for AI and BW
    function f = cost(X)
        f = 0.5*norm(P.*X - PA, 'fro')^2;
    end
    
    function g = egrad(X)
        g = (P.*X - PA).*P;
    end
    
    function gdot = ehess(X, Xdot)
        gdot = P.*Xdot .*P;
    end

    function mydist = disttosol(X)
        mydist = norm(X - A, 'fro');
    end
    
    function stats = computestats(problem, X, stats)
        stats.disttosol = disttosol(X);
    end
    

    % for LE
    problemLE.M = symmetricfactory(n);
    problemLE.cost = @costLE;
    problemLE.egrad = @egradLE;

    function f = costLE(S)
        f = cost(expm(S)); %0.5*norm(P.*expm(S) - PA, 'fro')^2;
    end

    function g = egradLE(S)
        R = egrad(expm(S)); % (P.*expm(S) - PA) .* P; % derivative wrt X or expm(S).
        g = symm(dexpm(S, R)); % derivatie wrt S.
    end

    function stats = computestatsLE(problem, S, stats)
        stats.disttosol = disttosol(expm(S));
    end
  

    
    
    % solve
    Xinitial = eye(n);
    Sinitial = logm(Xinitial);
    options.maxiter = 200;
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
        h2 = figure(2);
        plot(221);
        semilogy(cumsum([info_BW.numinner]), [info_BW.disttosol], '-o', 'color', 'b', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
        semilogy(cumsum([info_AI.numinner]), [info_AI.disttosol], '-+', 'color', 'r', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
        semilogy(cumsum([info_LE.numinner]), [info_LE.disttosol], '-x', 'color', 'm', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
        hold off;
        ax1 = gca;
        set(ax1,'FontSize', fs);
        set(h2,'Position',[100 100 600 500]);
        xlabel('Inner iterations (cumsum)', 'fontsize', fs);
        ylabel('Distance to solution', 'fontsize', fs);
        legend('BW', 'AI', 'LE');        
    else
        h2 = figure(2);
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
