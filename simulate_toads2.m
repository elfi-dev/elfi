function[X] = simulate_toads2(theta, ntoads, ndays, model)
% simulate toads location for each toad and day
% INPUT:
% theta - model parameters, alpha, gamma and p0 (the probability of returning to a previous refuge)
% ntoads - the number of individual toads
% ndays - the number of days for simulation
% model - indicator for model version in the paper Marchand, et al (2017),
%         1 stands for random return
%         2 stands for nearest return
% OUTPUT:
% X - a ndays by ntoads matrix contains toads location for each toad and

alpha = theta(1);
gamma = theta(2);
p0 = theta(3);
X = zeros(ndays, ntoads);

% if (model == 3)
%     X_uniq = zeros(ndays, ntoads); % locations of all unique refuge sites
%     n_rs = ones(1,ntoads); % initial number of refuge sites
%     pi_ret = 0;
% end

for i = 2:ndays
    if (model == 1) % random return
        % toads that stay at new location
        ind = rand(1,ntoads) >= p0;
        deltax = rndlas(sum(ind),gamma,alpha)' ; % distance of move (only generate distance for toads that stay at new location)
        X(i,ind) = X(i-1,ind) + deltax;
        
        % return to one of the previous refuge sites
        % multiple visits to a refuge site increases the weighting
        ind_refuge = randsample(i-1,ntoads-sum(ind),true)';
        idx = sub2ind(size(X),ind_refuge,find(~ind));
        X(i,find(~ind)) = X(idx);
        
    elseif (model == 2) % nearest return
        % generate displacements
        deltax = rndlas(ntoads,gamma,alpha)';
        X(i,:) = X(i-1,:) + deltax;
        
        % return to the closest refuge site with probability p0
        ind = rand(1,ntoads) >= p0;
        if (i == 2)
            X(i,find(~ind)) = zeros(1,length(find(~ind)));
        else
            [~,ind_refuge] = min(abs(X(i,find(~ind))-X(1:i-1,find(~ind))));
            idx = sub2ind(size(X),ind_refuge,find(~ind));
            X(i,find(~ind)) = X(idx);
        end
        
     else % (model == 3)
%         % generate displacements
%         deltax = rndlas(ntoads,gamma,alpha)';
%         X(i,:) = X(i-1,:) + deltax;
%         
%         dtemp = abs(xtemp-
     end
end














% previous code for model 1
% for j = 1:ntoads
%     for i = 2:ndays
%         deltax = rndlas(1, gamma, alpha); % distance of the potential move
%         
%         if rand >= p0 % take refuge here
%             X(i,j) = X(i-1,j) + deltax;
%         else % return to one of the previous refuge sites
%             ind_refuge = randsample(i-1,1);
%             X(i,j) = X(ind_refuge,j);
%         end
%         
%     end
% end

