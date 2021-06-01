function [] = test_trace_regression()

    % Paper: https://arxiv.org/pdf/1504.06305.pdf

    %rng('default');
    %rng(0);
    
    
    symm = @(Z) 0.5*(Z+Z');
    
    % parameters
    data_choice = 'SynFull'; % 'SynFull' or 'SynLow' or 
    solver_choice = 'TR';
    myeps = 1e-1;
    n = 1000;
    m = 50;
    
    switch data_choice
        case 'SynFull'
            r = m;
            
            x = zeros(m, 1, n);
            y = zeros(n, 1);

            Sigmaorg = randn(m, r);
            Sigmaorg = Sigmaorg*Sigmaorg';

            for kk = 1:n
                x(:,:,kk) = randn(m,1); % Rank 1 measurements
                y(kk) =  x(:,:,kk)'*(Sigmaorg*x(:,:,kk)) + myeps*randn(1);
            end
        case 'SynLow'
            r = 10;
            
            x = zeros(m, 1, n);
            y = zeros(n, 1);

            Sigmaorg = randn(m, r);
            Sigmaorg = Sigmaorg*Sigmaorg';

            for kk = 1:n
                x(:,:,kk) = randn(m,1); % Rank 1 measurements
                y(kk) =  x(:,:,kk)'*(Sigmaorg*x(:,:,kk)) + myeps*randn(1);
            end
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
    
    function f = cost(Sigma)
        f = 0;
        for kk = 1:n
        	xkk = x(:,:,kk);
        	ykk = y(kk);
        	f = f + (xkk'*Sigma*xkk - ykk)^2;
        end
        f = f/(2*n);
    end
    
    function g = egrad(Sigma)
        g = 0;
        for kk = 1:n
        	xkk = x(:,:,kk);
        	ykk = y(kk);
        	g = g + (xkk'*Sigma*xkk - ykk)*xkk*xkk';
        end
        g = g/n;
    end
    
    function gdot = ehess(Sigma, Sigmadot)
        gdot = 0;
        for kk = 1:n
        	xkk = x(:,:,kk);
        	ykk = y(kk);
        	gdot = gdot + (xkk'*Sigmadot*xkk)*xkk*xkk';
        end
        gdot = gdot/n;
    end

    function mydist = disttosol(Sigma)
        mydist = norm(Sigma - Sigmaorg, 'fro');
    end
    
    function stats = computestats(problem, Sigma, stats)
    	stats.disttosol = disttosol(Sigma);
        
        g = symm(problem.egrad(Sigma));
        stats.egradnorm = norm(Sigma*g, 'fro');
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
    end
    

    % Solve
    Xinitial = eye(m);
    Sinitial = logm(Xinitial);
    options.maxiter = 40;
    options.tolgradnorm = 1e-7;
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
        xlabel('Inner iterations (cumsum)', 'fontsize', fs);
        ylabel('Distance to solution', 'fontsize', fs);
        
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
    end 
end
